from langchain_core.messages import AIMessage
from tools.vision_tool import computer_vision_tool

def passenger_agent(state):
    """Passenger Agent - Specialized in passenger safety, counting and behaviour"""
    result = computer_vision_tool.invoke({"task": "passengers"})
    return {"messages": state["messages"] + [AIMessage(content=f"🧍 Passenger Agent: {result}")]}