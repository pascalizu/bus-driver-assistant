from langchain_core.messages import AIMessage
from tools.vision_tool import computer_vision_tool

def driver_agent(state):
    """Driver Agent - Specializes in driver fatigue and distraction monitoring."""
    # Call vision tool for real analysis
    vision_result = computer_vision_tool.invoke({"task": "fatigue"})
    return {"messages": state["messages"] + [AIMessage(content=f"Driver Agent: {vision_result}")]}