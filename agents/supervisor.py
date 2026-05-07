from langchain_core.messages import AIMessage

def supervisor_agent(state):
    """Supervisor Agent - Intelligent Router"""
    query = state["messages"][-1].content.lower()
    
    if any(word in query for word in ["drowsy", "fatigue", "tired", "eyes", "sleepy"]):
        route = "driver_agent"
        msg = "Supervisor: Routing to Driver Agent (Fatigue detected in query)"
    elif any(word in query for word in ["passenger", "hat", "people", "person", "count"]):
        route = "passenger_agent"
        msg = "Supervisor: Routing to Passenger Agent"
    elif any(word in query for word in ["when", "log", "history", "event", "record", "before"]):
        route = "query_agent"
        msg = "Supervisor: Routing to Query Agent"
    else:
        route = "operations_agent"
        msg = "Supervisor: Routing to Operations Agent"

    return {"messages": state["messages"] + [AIMessage(content=msg)]}