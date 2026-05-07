from langchain_core.messages import AIMessage
from tools.web_search_tool import web_search_tool

def operations_agent(state):
    """Operations Agent - Only handles route, traffic, and operational queries"""
    query = state["messages"][-1].content
    search_result = web_search_tool.invoke({"query": query})
    
    response = f"Operations Agent: Route and operations status checked.\n{search_result[:200]}..."  # Limit length
    return {"messages": state["messages"] + [AIMessage(content=response)]}