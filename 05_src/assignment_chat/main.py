from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from assignment_chat.prompts import return_instructions_root
#from research_api_calls import get_arxiv_info
import operator

import requests
import json
import xmltodict


from dotenv import load_dotenv
import json
import requests
from utils.logger import get_logger
import os


_logs = get_logger(__name__)

load_dotenv(".env")
load_dotenv(".secrets")



@tool
def get_arxiv_info(search_query = 'all:electron',start = 0, max_results = 5):# search for electron in all fields # retreive the first 5 results
    """
    Returns max_results papers from the arxiv API, each paper contains the author name, titles affiliations and a paper summary.
    """

    def xml_to_json_from_text(xml_text: str) -> dict:
        """
        Converts XML text (e.g., from an API response) into a JSON-compatible dict.
        Automatically handles XML namespaces and nested structures.
        """
        # Parse XML into Python dict
        data_dict = xmltodict.parse(xml_text, process_namespaces=True)

        # Convert dict to JSON string (pretty-printed)
        json_str = json.dumps(data_dict, indent=2)

        # Convert back to dict (optional, if you prefer working with a Python object)
        return json.loads(json_str)

    def extract_arxiv_papers(arxiv_dict):
        feed_key = "http://www.w3.org/2005/Atom:feed"
        entry_key = "http://www.w3.org/2005/Atom:entry"
        author_key = "http://www.w3.org/2005/Atom:author"
        
        entries = arxiv_dict.get(feed_key, {}).get(entry_key, [])
        if isinstance(entries, dict):
            entries = [entries]  # single entry case

        papers = []
        for entry in entries:
            # Title and abstract
            title = entry.get("http://www.w3.org/2005/Atom:title", "").strip()
            abstract = entry.get("http://www.w3.org/2005/Atom:summary", "").strip()

            # DOI
            doi = entry.get("http://arxiv.org/schemas/atom:doi", {}).get("#text")

            # Journal reference / publication info
            journal_ref = entry.get("http://arxiv.org/schemas/atom:journal_ref", {}).get("#text")

            # Authors
            authors_data = entry.get(author_key, [])
            if isinstance(authors_data, dict):
                authors_data = [authors_data]  # single author case
            authors = []
            for a in authors_data:
                name = a.get("http://www.w3.org/2005/Atom:name")
                affiliation = a.get("http://arxiv.org/schemas/atom:affiliation", {}).get("#text")
                authors.append({"name": name, "affiliation": affiliation})

            # Categories
            primary_category = entry.get("http://arxiv.org/schemas/atom:primary_category", {}).get("@term")
            categories_data = entry.get("http://www.w3.org/2005/Atom:category", [])
            if isinstance(categories_data, dict):
                categories_data = [categories_data]
            categories = [c.get("@term") for c in categories_data if "@term" in c]

            # PDF link
            links = entry.get("http://www.w3.org/2005/Atom:link", [])
            if isinstance(links, dict):
                links = [links]
            pdf_link = None
            for l in links:
                if l.get("@title") == "pdf":
                    pdf_link = l.get("@href")
                    break

            papers.append({
                "title": title,
                "abstract": abstract,
                "doi": doi,
                "journal_reference": journal_ref,
                "authors": authors,
                "primary_category": primary_category,
                "categories": categories,
                "pdf_link": pdf_link
            })

        return papers



    # Base api query url
    base_url = 'http://export.arxiv.org/api/query?';

    query = 'search_query=%s&start=%i&max_results=%i' % (search_query,
                                                        start,
                                                        max_results)


    # # perform a GET request using the base_url and query
    # response = urllib.urlopen(base_url+query).read()
    response = requests.get(base_url, params=query)
    #print(response.text)

    feed_info = xml_to_json_from_text(response.text)
    #print(feed_info)

    papers = extract_arxiv_papers(feed_info)
    print(json.dumps(papers, indent=2))
    return papers



def get_model_with_tools():
    model = init_chat_model(
        "openai:gpt-4o-mini",
        temperature=0.7
    )
    # Augment the LLM with tools
    tools = [get_arxiv_info]
    model_with_tools = model.bind_tools(tools)
    return model_with_tools

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    model_with_tools = get_model_with_tools()
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful research assistant tasked with looking up scientific papers about topics and returning with a concice summary of the abstracts of this papers."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }



def tool_node(state: dict):
    """Performs the tool call — expands user query, retrieves arXiv papers, and summarizes them."""

    tools = [get_arxiv_info]
    tools_by_name = {tool.name: tool for tool in tools}
    result = []

    model_with_tools = get_model_with_tools()

    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        _logs.info("No new tool calls found. Skipping tool_node execution.")
        return {"messages": []}  # prevents recursion

    user_query = ""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and msg.content:
            user_query = msg.content
            break

 # Expanding the user query capabilities to change the number of papers and durations etc.
    search_query_prompt = f"""
    You are a research assistant helping to find relevant papers on arXiv.

    Based on the user's message below, create a concise search string for the arXiv API:
    "{user_query}"

    - Use the format: all:keyword1+keyword2+...
    - Include specific years (e.g., 2025) if mentioned.
    - Exclude filler words like "find", "show me", or "paper about".
    - Example: all:time+series+forecasting+transformer+2025
    Return ONLY the search query string, nothing else.
    """

    query_generation = model_with_tools.invoke([
        SystemMessage(content="You generate precise arXiv API query strings."),
        HumanMessage(content=search_query_prompt)
    ])
    search_query_str = query_generation.content.strip()
    _logs.info(f"Generated arXiv search query: {search_query_str}")

    for tool_call in last_msg.tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        if not tool:
            _logs.warning(f"Unknown tool: {tool_call['name']}")
            continue

        papers = tool.invoke({
            "search_query": search_query_str,
            "start": 0,
            "max_results": 10
        })

        papers_str_list = []
        for paper in papers:
            authors_str = ", ".join(
                f"{a['name']} ({a['affiliation']})" if a.get("affiliation") else a['name']
                for a in paper.get("authors", [])
            )
            categories_str = ", ".join(paper.get("categories", []))
            paper_text = (
                f"Title: {paper.get('title')}\n"
                f"Authors & Affiliations: {authors_str}\n"
                f"DOI / Journal: {paper.get('doi') or paper.get('journal_reference')}\n"
                f"Primary Category / Keywords: {paper.get('primary_category')} / {categories_str}\n"
                f"Summary: {paper.get('abstract')}\n"
            )
            papers_str_list.append(paper_text)

        papers_str = "\n\n".join(papers_str_list)

        summary_response = model_with_tools.invoke([
            SystemMessage(content=(
                "You are Jarvis, a research assistant. "
                f"The following search was performed on arXiv: {search_query_str}\n\n"
                "Summarize the following retrieved papers into a coherent mini literature review."
                "It should be in one paragraph and tell a story based on the abstracts and name the authors and the date for each paper referenced."
                "Identify trends, key contributions, and connections between the works. "
                "Avoid repetition and citation-like output. Keep it academic and concise."
            )),
            HumanMessage(content=papers_str)
        ])

        result.append(
            ToolMessage(
                content=summary_response.content,
                tool_call_id=tool_call["id"]
            )
        )

    last_msg.tool_calls = []

    return {"messages": result}



def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

def get_assignment_chat_agent():
    """Returns the assignment chat agent"""    
    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")
    return agent_builder.compile()