from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import cv2
import time

# Load API Key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file. Please add it.")

print(f"✅ Gemini API Key Loaded: {API_KEY[:15]}...")

# Import agents
from agents.supervisor import supervisor_agent
from agents.driver_agent import driver_agent
from agents.passenger_agent import passenger_agent
from agents.operations_agent import operations_agent
from agents.query_agent import query_agent

# Import Voice Agent
from utils.voice import voice_agent

class AgentState(dict):
    messages: list

# ==================== LangGraph Workflow ====================
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("driver_agent", driver_agent)
workflow.add_node("passenger_agent", passenger_agent)
workflow.add_node("operations_agent", operations_agent)
workflow.add_node("query_agent", query_agent)

workflow.set_entry_point("supervisor")

# Improved Routing Logic
workflow.add_conditional_edges(
    "supervisor",
    lambda state: "driver_agent" if any(word in state["messages"][-1].content.lower() for word in ["fatigue", "drowsy", "tired", "eyes"]) else
                  "passenger_agent" if any(word in state["messages"][-1].content.lower() for word in ["passenger", "hat", "people", "person"]) else
                  "query_agent" if any(word in state["messages"][-1].content.lower() for word in ["when", "log", "history", "event", "record"]) else
                  "operations_agent"
)

workflow.add_edge("driver_agent", END)
workflow.add_edge("passenger_agent", END)
workflow.add_edge("operations_agent", END)
workflow.add_edge("query_agent", END)

app = workflow.compile()

# ====================== REAL-TIME MONITORING ======================
def run_real_time_monitoring():
    print("🚍 REAL-TIME MONITORING + VOICE AGENT STARTED")
    print("Press 'v' → Manual Voice Announcement")
    print("Press 'q' → Enter Query Mode\n")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        input("Press Enter to continue to Query Mode...")
        return

    last_alert = 0
    cooldown = 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Press 'q' = Query Mode | 'v' = Voice", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Bus Driver Assistant - Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):
            voice_agent.speak("Please remain seated while the bus is moving. Thank you.")

        # Auto drowsy alert (placeholder - improve later with real detection)
        if time.time() - last_alert > cooldown:
            # voice_agent.speak_drowsy_alert()   # Uncomment when ready
            last_alert = time.time()

    cap.release()
    cv2.destroyAllWindows()

# ====================== QUERY MODE ======================
def run_query_mode():
    print("\n" + "="*80)
    print("MULTI-AGENT QUERY MODE")
    print("="*80)

    system_prompt = "You are a professional bus safety assistant."

    while True:
        user_input = input("\n🧑‍💻 You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Project completed successfully! 🏆")
            break

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
        
        for output in app.stream({"messages": messages}):
            for key, value in output.items():
                if "messages" in value:
                    last = value["messages"][-1]
                    if hasattr(last, "content") and last.content:
                        print(f"🤖 Assistant: {last.content}")

__all__ = ["app", "run_real_time_monitoring", "run_query_mode"]