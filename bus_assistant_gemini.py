import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import numpy as np
import chromadb
from datetime import datetime
import pyttsx3
from duckduckgo_search import DDGS

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key="AIzaSyANao1JE_nSPIWwbwanSsmK2WPjx3pM2oA",
    temperature=0.3
)

# Chroma
chroma_client = chromadb.PersistentClient(path="events")
collection = chroma_client.get_or_create_collection(name="bus_events")

# Vision
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
yolo_model = YOLO('yolov8n.pt')

# Voice
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# ====================== TOOLS ======================
@tool
def event_search_tool(query: str) -> str:
    """Search logged events. Always base your answer ONLY on these logs."""
    results = collection.query(query_texts=[query], n_results=15)
    if not results['documents'][0]:
        return "No matching events found in the log."
    return "LOGGED EVENTS:\n" + "\n".join(results['documents'][0])

@tool
def computer_vision_tool(task: str) -> str:
    """Get current real-time vision analysis."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Camera not accessible."

    # Fatigue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    drowsy = False
    if results.multi_face_landmarks:
        for lm in results.multi_face_landmarks:
            left = lm.landmark[159].y - lm.landmark[145].y
            right = lm.landmark[386].y - lm.landmark[374].y
            if (left + right) / 2 < 0.015:
                drowsy = True

    # Passengers & Red Hat
    results = yolo_model(frame, classes=0, verbose=False)
    passengers = 0
    red_hat = False
    h, w = frame.shape[:2]
    zone = int(w * 0.3)
    
    for r in results:
        for box in r.boxes:
            x1 = int(box.xyxy[0][0])
            if x1 > zone:
                passengers += 1
                head = frame[int(box.xyxy[0][1]):int(box.xyxy[0][3]), x1:int(box.xyxy[0][2])]
                if head.size > 0:
                    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array([0,70,50]), np.array([10,255,255])) + \
                           cv2.inRange(hsv, np.array([170,70,50]), np.array([180,255,255]))
                    if cv2.countNonZero(mask) / head.size > 0.18:
                        red_hat = True

    status = f"Current status: Driver is {'drowsy' if drowsy else 'alert'}. {passengers} passengers detected."
    if red_hat:
        status += " Red hat detected."
    return status

tools = [event_search_tool, computer_vision_tool]
llm_with_tools = llm.bind_tools(tools)

# ====================== REAL-TIME MONITORING ======================
def run_real_time_monitoring():
    print("🚍 REAL-TIME MONITORING STARTED")
    print("Press 'q' to enter QUERY MODE\n")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera error")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Press 'q' for Query Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Bus Driver Assistant - Real-Time", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ====================== QUERY MODE ======================
def run_query_mode():
    print("\n" + "="*70)
    print("MULTI-AGENT QUERY MODE")
    print("="*70)

    system_prompt = """You are a bus safety assistant.
    Always base your answers ONLY on the logged events or current vision tool.
    Do not hallucinate times, events, or detections."""

    while True:
        user_input = input("\n🧑‍💻 You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Project completed!")
            break

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] == "event_search_tool":
                    result = event_search_tool.invoke(tc["args"])
                elif tc["name"] == "computer_vision_tool":
                    result = computer_vision_tool.invoke(tc["args"])
                else:
                    result = "Tool executed."
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

            final = llm.invoke(messages)
            print(f"🤖 Assistant: {final.content}")
        else:
            print(f"🤖 Assistant: {response.content}")

if __name__ == "__main__":
    run_real_time_monitoring()
    run_query_mode()