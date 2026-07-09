from langchain_core.messages import AIMessage
from tools.web_search_tool import web_search_tool

def operations_agent(state):
    """Operations Agent - Specialized in route, traffic and operational efficiency"""
    result = web_search_tool.invoke({"query": "current traffic conditions for buses"})
    return {"messages": state["messages"] + [AIMessage(content=f"🛣️ Operations Agent: Route & operations status checked.\n{result[:150]}...")]}