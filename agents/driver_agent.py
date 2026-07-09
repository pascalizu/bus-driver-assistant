from langchain_core.messages import AIMessage
from tools.vision_tool import computer_vision_tool

def driver_agent(state):
    """Driver Agent - Specialized in fatigue, distraction and driver safety"""
    result = computer_vision_tool.invoke({"task": "fatigue"})
    return {"messages": state["messages"] + [AIMessage(content=f"🚨 Driver Agent: {result}")]}