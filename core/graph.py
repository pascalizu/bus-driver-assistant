from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

# Import all agents
from agents.supervisor import supervisor_agent
from agents.driver_agent import driver_agent
from agents.passenger_agent import passenger_agent
from agents.operations_agent import operations_agent
from agents.query_agent import query_agent

# Import tools for binding (if needed in supervisor)
from tools.vision_tool import computer_vision_tool
from tools.math_tool import math_calculation_tool
from tools.web_search_tool import web_search_tool
from tools.event_search_tool import event_search_tool

class AgentState(dict):
    """Shared state for all agents"""
    messages: list

# Build the LangGraph workflow
workflow = StateGraph(AgentState)

# Add all nodes (agents)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("driver_agent", driver_agent)
workflow.add_node("passenger_agent", passenger_agent)
workflow.add_node("operations_agent", operations_agent)
workflow.add_node("query_agent", query_agent)

# Set entry point
workflow.set_entry_point("supervisor")

# Define routing logic from supervisor
workflow.add_conditional_edges(
    "supervisor",
    lambda state: "driver_agent" if "fatigue" in state["messages"][-1].content.lower() or "drowsy" in state["messages"][-1].content.lower() else
                 "passenger_agent" if any(word in state["messages"][-1].content.lower() for word in ["passenger", "hat", "person"]) else
                 "operations_agent" if any(word in state["messages"][-1].content.lower() for word in ["route", "traffic", "load", "operation"]) else
                 "query_agent"
)

# Define terminal edges
workflow.add_edge("driver_agent", END)
workflow.add_edge("passenger_agent", END)
workflow.add_edge("operations_agent", END)
workflow.add_edge("query_agent", END)

# Compile the graph
app = workflow.compile()

# ====================== RUN FUNCTIONS ======================

def run_real_time_monitoring():
    """Placeholder for real-time monitoring loop"""
    print("🚍 REAL-TIME MONITORING STARTED")
    print("Camera feed would appear here in full version.")
    print("Press 'q' to enter QUERY MODE\n")
    input("Press Enter to continue to Query Mode...")

def run_query_mode():
    """Interactive query mode using the full multi-agent system"""
    print("\n" + "="*80)
    print("MULTI-AGENT QUERY MODE (5 Agents + 4 Tools)")
    print("Ask questions about fatigue, passengers, route, or logged events.")
    print("="*80)

    system_prompt = """You are a professional bus safety assistant.
    Use the available tools to get real information.
    Always base your answers on tool results or logged events.
    Be concise and clear."""

    while True:
        user_input = input("\n🧑‍💻 You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nThank you! Project completed successfully. 🏆")
            break

        # Run the graph
        inputs = {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]}
        
        for output in app.stream(inputs):
            for key, value in output.items():
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        print(f"🤖 Assistant: {last_msg.content}")

# Export for main.py
__all__ = ["app", "run_real_time_monitoring", "run_query_mode"]