from langchain_core.messages import AIMessage
from tools.web_search_tool import web_search_tool

def operations_agent(state):
    """Operations Agent - Handles route adherence, load optimization, and external data."""
    # Example: Search for traffic or route status
    search_result = web_search_tool.invoke({"query": "current traffic conditions"})
    return {"messages": state["messages"] + [AIMessage(content=f"Operations Agent: Route is on schedule. {search_result}")]}