import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import numpy as np
import chromadb
from datetime import datetime
import os

# Gemini Setup (Active - since you have your API key)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",  # Fast, multimodal, intelligent
    google_api_key="AIzaSyDzLntEwcZiXFPAob4LjvAe4HK63Jw_zW8",  # You already replaced this with your key
    temperature=0.3,
    # Optional: Convert response to AIMessage for LangGraph compatibility
    convert_system_message_to_human=True
)

# --- Chroma Setup for Event Logging ---
chroma_client = chromadb.PersistentClient(path="events")
collection = chroma_client.get_or_create_collection(name="bus_events")

# --- Vision Models ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
yolo_model = YOLO('yolov8n.pt')  # Nano for speed

# --- Real-Time Detection & Logging ---
def log_event(description: str):
    """Log detection event to Chroma with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_desc = f"Time: {timestamp} | {description}"
    collection.add(
        documents=[full_desc],
        metadatas=[{"timestamp": timestamp, "raw_desc": description}],
        ids=[str(time.time())]
    )
    print(f"📝 Logged: {full_desc}")

def detect_fatigue(frame):
    """Real driver fatigue detection using MediaPipe eye openness."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return False, "No face detected"
    
    for face_landmarks in results.multi_face_landmarks:
        left_eye = (face_landmarks.landmark[159].y - face_landmarks.landmark[145].y)
        right_eye = (face_landmarks.landmark[386].y - face_landmarks.landmark[374].y)
        eye_openness = (left_eye + right_eye) / 2
        
        if eye_openness < 0.015:
            return True, "Driver drowsy - eyes closed"
    
    return False, "Driver alert - eyes open"

def detect_passengers_and_red_hat(frame):
    """Real passenger counting + red hat detection using YOLO + HSV."""
    results = yolo_model(frame, classes=0, verbose=False)  # Person class only
    
    passenger_count = 0
    red_hat_detected = False
    
    h, w = frame.shape[:2]
    passenger_zone_x = int(w * 0.3)  # Exclude driver (left 30%)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if x1 > passenger_zone_x:
                passenger_count += 1
                
                head_y1 = y1
                head_y2 = int(y1 + (y2 - y1) * 0.25)
                head_region = frame[head_y1:head_y2, x1:x2]
                
                if head_region.size > 0:
                    hsv_head = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
                    lower_red1 = np.array([0, 70, 50])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([170, 70, 50])
                    upper_red2 = np.array([180, 255, 255])
                    
                    mask1 = cv2.inRange(hsv_head, lower_red1, upper_red1)
                    mask2 = cv2.inRange(hsv_head, lower_red2, upper_red2)
                    red_ratio = (cv2.countNonZero(mask1) + cv2.countNonZero(mask2)) / head_region.size
                    
                    if red_ratio > 0.2:
                        red_hat_detected = True
                        cv2.putText(frame, "RED HAT!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    desc = f"{passenger_count} passengers on board"
    if red_hat_detected:
        desc += ". Red hat detected!"
    
    return passenger_count, red_hat_detected, desc

# --- Real-Time Video Monitoring Loop ---
def run_real_time_monitoring():
    print("🚍 REAL-TIME BUS MONITORING STARTED (Gemini 3 Flash)")
    print("Close eyes / wear red hat / add people (right side) → detections logged")
    print("Press 'q' to stop and enter QUERY MODE\n")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Webcam not accessible")
        return
    
    last_fatigue_alert = 0
    alert_cooldown = 20
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        is_drowsy, fatigue_status = detect_fatigue(frame)
        cv2.putText(frame, fatigue_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (0, 0, 255) if is_drowsy else (0, 255, 0), 2)
        
        if is_drowsy and (time.time() - last_fatigue_alert > alert_cooldown):
            log_event("Driver drowsy")
            last_fatigue_alert = time.time()
        
        count, red_hat, passenger_status = detect_passengers_and_red_hat(frame)
        cv2.putText(frame, passenger_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if red_hat or count > 0:
            log_event(passenger_status)
        
        cv2.imshow('Bus Driver Assistant - Real-Time (Gemini Powered)', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring stopped → QUERY MODE\n")

# --- Tools ---
@tool
def query_event_memory(query: str) -> str:
    results = collection.query(query_texts=[query], n_results=10)
    if not results['documents'][0]:
        return "No events found."
    events = "\n".join(results['documents'][0])
    return f"Found events:\n{events}"

tools = [query_event_memory]
tool_node = ToolNode(tools)
model_with_tools = llm.bind_tools(tools)

# --- Query Agent Graph ---
class AgentState(dict):
    messages: list

def supervisor(state):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", lambda s: "tools" if s["messages"][-1].tool_calls else END)
workflow.add_edge("tools", "supervisor")
app = workflow.compile()

# --- Run Full System ---
if __name__ == "__main__":
    run_real_time_monitoring()
    
    print("="*70)
    print("GEMINI-POWERED QUERY MODE")
    print("Ask about real logged events!")
    print("="*70)
    
    while True:
        user_input = input("\n🧑‍💻 You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nProject complete! Ready Tensor gold 🏆")
            break
        
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for output in app.stream(inputs):
            for key, value in output.items():
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        print(f"🤖 Gemini: {last_msg.content}")