
import requests
import json
import xmltodict


def get_arxiv_info(search_query = 'all:electron',start = 0, max_results = 5):# search for electron in all fields # retreive the first 5 results

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

    #resp_dict = json.loads(response.text)
    # facts_list = resp_dict.get("data", [])
    # facts = "\n".join([f"{i+1}. {fact}\n" for i, fact in enumerate(facts_list)])

    #print(response.text)

    feed_info = xml_to_json_from_text(response.text)
    #print(feed_info)

    papers = extract_arxiv_papers(feed_info)
    print(json.dumps(papers, indent=2))
    return papers