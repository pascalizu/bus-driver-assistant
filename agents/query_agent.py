from langchain_core.messages import AIMessage
from tools.event_search_tool import event_search_tool

def query_agent(state):
    """Query Agent - Handles natural language queries over logged events."""
    query = state["messages"][-1].content
    result = event_search_tool.invoke({"query": query})
    return {"messages": state["messages"] + [AIMessage(content=f"Query Agent:\n{result}")]}