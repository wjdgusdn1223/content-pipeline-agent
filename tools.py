import os, re

from crewai.tools import tool
import requests


@tool
def web_search_tool(query: str):
    """
    Web Search Tool.
    Args:
        query: str
            The query to search the web for.
    Returns
        A list of search results with the website content in Markdown format.
    """
    api_key = os.getenv("FIRECRAWL_API_KEY")

    url = "https://api.firecrawl.dev/v2/search"

    payload = {
        "query": query,
        "sources": ["web"],
        "categories": [],
        "limit": 5,
        "scrapeOptions": {
            "onlyMainContent": True,
            "maxAge": 172800000,
            "parsers": ["pdf"],
            "formats": ["markdown"],
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers)
    response = response.json()

    if not response["success"]:
        return "Error using tool."

    cleaned_chunks = []

    for result in response["data"]["web"]:

        title = result["title"]
        url = result["url"]
        markdown = result["markdown"]

        cleaned_markdown = re.sub(r"\\+|\n+", "", markdown).strip()
        cleaned_markdown = re.sub(
            r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned_markdown
        )

        cleaned_result = {
            "title": title,
            "url": url,
            "markdown": cleaned_markdown,
        }

        cleaned_chunks.append(cleaned_result)

    return cleaned_chunks
