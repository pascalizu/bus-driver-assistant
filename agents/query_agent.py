from langchain_core.messages import AIMessage
from tools.event_search_tool import event_search_tool

def query_agent(state):
    """Query Agent - Handles questions about past events"""
    query = state["messages"][-1].content
    result = event_search_tool.invoke({"query": query})
    
    if "No matching events" in result:
        response = "Query Agent: No events found in the log yet. Start real-time monitoring to generate logs."
    else:
        response = f"Query Agent:\n{result}"
    
    return {"messages": state["messages"] + [AIMessage(content=response)]}