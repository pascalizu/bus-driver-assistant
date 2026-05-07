from langchain_core.messages import AIMessage
from tools.vision_tool import computer_vision_tool

def passenger_agent(state):
    """Passenger Agent - Monitors passengers and objects"""
    vision_result = computer_vision_tool.invoke({"task": "passengers"})
    response = f"Passenger Agent: {vision_result}"
    return {"messages": state["messages"] + [AIMessage(content=response)]}