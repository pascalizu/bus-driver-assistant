from langchain_core.messages import AIMessage
from tools.vision_tool import computer_vision_tool

def driver_agent(state):
    """Driver Agent - Focuses on fatigue and driver safety"""
    vision_result = computer_vision_tool.invoke({"task": "fatigue"})
    response = f"Driver Agent: {vision_result}"
    return {"messages": state["messages"] + [AIMessage(content=response)]}