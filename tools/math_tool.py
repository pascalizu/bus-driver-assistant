from langchain_core.tools import tool

@tool
def math_calculation_tool(expression: str) -> str:
    """Perform mathematical calculations such as averages, sums, or trends."""
    try:
        result = eval(expression)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Invalid mathematical expression. Error: {str(e)}"