from langchain_core.tools import tool
from duckduckgo_search import DDGS

@tool
def web_search_tool(query: str) -> str:
    """Search the web for real-time information like traffic, weather, or route conditions."""
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
            return "\n".join(results) if results else "No relevant results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"