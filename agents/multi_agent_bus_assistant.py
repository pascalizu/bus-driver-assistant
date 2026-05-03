import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import numpy as np
import chromadb
from datetime import datetime
import os

from google import genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# ---------------------------
# GEMINI SETUP (FIXED)
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),  # FIXED SECURITY
    temperature=0.3
)


# ---------------------------
# CHROMA DB
# ---------------------------
chroma_client = chromadb.PersistentClient(path="events")
collection = chroma_client.get_or_create_collection(name="bus_events")


# ---------------------------
# MODELS
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

yolo_model = YOLO("yolov8n.pt")


# ---------------------------
# LOGGING
# ---------------------------
def log_event(description: str):
    """Logs detected events into persistent memory (ChromaDB)."""
    timestamp = datetime.now().strftime("%H:%M:%S")

    collection.add(
        documents=[f"{timestamp} | {description}"],
        metadatas=[{"timestamp": timestamp}],
        ids=[str(time.time())]
    )

    print(f"📝 Logged: {description}")


# ---------------------------
# FATIGUE DETECTION
# ---------------------------
def detect_fatigue(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return False, "No face detected"

    for face in results.multi_face_landmarks:
        left_eye = face.landmark[159].y - face.landmark[145].y
        right_eye = face.landmark[386].y - face.landmark[374].y

        eye_open = (left_eye + right_eye) / 2

        if eye_open < 0.015:
            return True, "Driver drowsy"

    return False, "Driver alert"


# ---------------------------
# YOLO DETECTION
# ---------------------------
def detect_passengers_and_red_hat(frame):
    results = yolo_model(frame, classes=0, verbose=False)

    count = 0
    red_hat = False

    for r in results:
        if r.boxes:
            count = len(r.boxes)

    return count, red_hat, f"{count} passengers detected"


# ---------------------------
# REAL-TIME MONITORING
# ---------------------------
def run_monitor():
    print("🚍 SYSTEM STARTED")

    cap = cv2.VideoCapture(0)

    last_alert = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        drowsy, status = detect_fatigue(frame)
        passengers, red_hat, desc = detect_passengers_and_red_hat(frame)

        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if drowsy else (0, 255, 0), 2)

        cv2.putText(frame, desc, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if drowsy and time.time() - last_alert > 20:
            log_event("Driver drowsy detected")

            response = llm.invoke([
                HumanMessage(content="Warn a bus driver about fatigue in one short sentence.")
            ])

            print("🚨 AI:", response.content)
            last_alert = time.time()

        cv2.imshow("Bus Assistant", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------
# TOOL (FIXED - DOCSTRING ADDED)
# ---------------------------
@tool
def query_event_memory(query: str) -> str:
    """
    Searches stored bus events and returns relevant historical logs.
    """
    results = collection.query(query_texts=[query], n_results=5)

    if not results["documents"][0]:
        return "No events found."

    return "\n".join(results["documents"][0])


# ---------------------------
# LANGGRAPH SETUP
# ---------------------------
tools = [query_event_memory]
tool_node = ToolNode(tools)
model_with_tools = llm.bind_tools(tools)


class AgentState(dict):
    pass


def supervisor(state):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda s: "tools" if s["messages"][-1].tool_calls else END
)

workflow.add_edge("tools", "supervisor")

app = workflow.compile()


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    run_monitor()

    print("\n💬 QUERY MODE STARTED")

    while True:
        q = input("You: ")

        if q.lower() in ["exit", "quit"]:
            break

        result = app.invoke({"messages": [HumanMessage(content=q)]})

        print("AI:", result["messages"][-1].content)

def run_agents():
    run_monitor()   # or your actual main function