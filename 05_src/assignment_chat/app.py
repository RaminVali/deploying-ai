from assignment_chat.main import get_assignment_chat_agent
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr
from dotenv import load_dotenv
import os

from utils.logger import get_logger

_logs = get_logger(__name__)

llm = get_assignment_chat_agent()

load_dotenv('.secrets')  #When we are doing dot-env then we have to be in the right pATH, make sure your path is good. When you run the app then go to the top folder of the APP.

#Check environment and API key
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY environment variable")


def assignment_chat(message: str, history: list[dict]) -> str:
    langchain_messages = []
    n = 0
    _logs.debug(f"History: {history}")
    for msg in history:
        if msg['role'] == 'user':
            langchain_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            langchain_messages.append(AIMessage(content=msg['content']))
            n += 1
    langchain_messages.append(HumanMessage(content=message))

    state = {
        "messages": langchain_messages,
        "llm_calls": n
    }

    response = llm.invoke(state)
    return response['messages'][len(response['messages']) - 1].content

# Add initial message so the user can see examples of using the product
initial_message = (""" Hi, I am Jarvis your research assistant! \n
                   I can help you with:\n
                   - Searching papers directly from arXiv API and giving literature reviews\n
                   - Performing a semantic search using my local arXiv database (15000 papers) you should mention my local db if you want searches performed there. \n
                   - I can query the web and summarise the results I am getting for scientific and research areas.\n
                   What would you like to do? """)

chat = gr.ChatInterface(
    fn=assignment_chat,
    type="messages",
    chatbot=gr.Chatbot(placeholder= initial_message, type='messages'),
)

if __name__ == "__main__":
    _logs.info('Starting Assignment Chat App...')
    chat.launch()
