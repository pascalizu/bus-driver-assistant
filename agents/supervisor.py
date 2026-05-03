from langchain_core.messages import AIMessage

def supervisor_agent(state):
    """Supervisor Agent - Central coordinator that routes tasks to specialized agents."""
    last_message = state["messages"][-1].content.lower()
    
    if "fatigue" in last_message or "drowsy" in last_message:
        return {"messages": state["messages"] + [AIMessage(content="Supervisor: Routing to Driver Agent")]}
    elif "passenger" in last_message or "hat" in last_message:
        return {"messages": state["messages"] + [AIMessage(content="Supervisor: Routing to Passenger Agent")]}
    elif "route" in last_message or "traffic" in last_message or "operation" in last_message:
        return {"messages": state["messages"] + [AIMessage(content="Supervisor: Routing to Operations Agent")]}
    else:
        return {"messages": state["messages"] + [AIMessage(content="Supervisor: Routing to Query Agent")]}