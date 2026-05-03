from langchain_core.tools import tool
import chromadb

# Initialize Chroma (shared across tools)
chroma_client = chromadb.PersistentClient(path="events")
collection = chroma_client.get_or_create_collection(name="bus_events")

@tool
def event_search_tool(query: str) -> str:
    """Search the logged events database semantically using natural language."""
    try:
        results = collection.query(query_texts=[query], n_results=10)
        if not results['documents'][0]:
            return "No matching events found in the log."
        return "LOGGED EVENTS:\n" + "\n".join(results['documents'][0])
    except Exception as e:
        return f"Event search error: {str(e)}"