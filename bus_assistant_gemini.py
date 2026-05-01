import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import numpy as np
import chromadb
from datetime import datetime
import pyttsx3  # For voice announcements (offline TTS)

# ====================== GEMINI SETUP ======================
# NOTE: This API key is included for demonstration purposes only.
# In a production environment, use environment variables or secrets manager.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key="AIzaSyDzLntEwcZiXFPAob4LjvAe4HK63Jw_zW8",   # Demo key - replace with your own in production
    temperature=0.3
)

# ====================== CHROMA SETUP ======================
chroma_client = chromadb.PersistentClient(path="events")
collection = chroma_client.get_or_create_collection(name="bus_events")

# ====================== VISION MODELS ======================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
yolo_model = YOLO('yolov8n.pt')

# ====================== VOICE AGENT SETUP ======================
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 1.0)
    VOICE_AVAILABLE = True
except Exception as e:
    print(f"Voice agent init failed: {e}")
    VOICE_AVAILABLE = False

passenger_instructions = [
    "Please remain seated while the bus is in motion.",
    "No eating or drinking on the bus.",
    "Do not open the windows.",
    "No hawking or spitting on the bus.",
    "Please move to the rear to make room for others.",
    "Hold on to the handrails when standing.",
    "Keep aisles clear of bags.",
    "Thank you for riding safely!"
]

instruction_index = 0
DROWSY_ALERT_TEXT = "Driver, you appear drowsy. For safety, please pull over and take a short break if needed."

def speak(text: str):
    if VOICE_AVAILABLE:
        print(f"🔊 Announcing: {text}")
        tts_engine.say(text)
        tts_engine.runAndWait()
    else:
        print(f"🔊 (Voice disabled): {text}")
    
    log_event(f"Voice announcement: {text}")

# ====================== LOGGING ======================
def log_event(description: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_desc = f"Time: {timestamp} | {description}"
    collection.add(documents=[full_desc], ids=[str(time.time())])
    print(f"📝 Logged: {full_desc}")

# ====================== VISION FUNCTIONS ======================
def detect_fatigue(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return False, "No face detected"
    for landmarks in results.multi_face_landmarks:
        left = landmarks.landmark[159].y - landmarks.landmark[145].y
        right = landmarks.landmark[386].y - landmarks.landmark[374].y
        eye_openness = (left + right) / 2
        if eye_openness < 0.015:
            return True, "DROWSY - Eyes closed"
    return False, "Alert - Eyes open"

def detect_passengers_and_red_hat(frame):
    results = yolo_model(frame, classes=0, verbose=False)
    count = 0
    red_hat = False
    h, w = frame.shape[:2]
    passenger_zone = int(w * 0.3)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 > passenger_zone:
                count += 1
                head_region = frame[y1:int(y1 + (y2 - y1)*0.25), x1:x2]
                if head_region.size > 0:
                    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
                    red_mask = cv2.inRange(hsv, np.array([0,70,50]), np.array([10,255,255])) + \
                               cv2.inRange(hsv, np.array([170,70,50]), np.array([180,255,255]))
                    if cv2.countNonZero(red_mask) / head_region.size > 0.2:
                        red_hat = True
                        cv2.putText(frame, "RED HAT!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    desc = f"{count} passengers" + (" | Red hat detected" if red_hat else "")
    return count, red_hat, desc

# ====================== REAL-TIME MONITORING ======================
def run_real_time_monitoring():
    print("🚍 REAL-TIME MONITORING + VOICE AGENT STARTED")
    print("Automatic drowsy alert + manual 'v' for instructions")
    print("Press 'q' to stop and enter QUERY MODE\n")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Webcam not accessible")
        return
    
    global instruction_index
    last_drowsy_alert = 0
    drowsy_cooldown = 30
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        drowsy, fatigue_text = detect_fatigue(frame)
        cv2.putText(frame, fatigue_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (0,0,255) if drowsy else (0,255,0), 2)
        
        if drowsy and (time.time() - last_drowsy_alert > drowsy_cooldown):
            log_event("Driver drowsy")
            speak(DROWSY_ALERT_TEXT)
            last_drowsy_alert = time.time()
        
        count, red_hat, passenger_text = detect_passengers_and_red_hat(frame)
        cv2.putText(frame, passenger_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        log_event(passenger_text)
        
        cv2.imshow('Bus Driver Assistant - Real-Time + Voice', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):
            instruction = passenger_instructions[instruction_index]
            speak(instruction)
            instruction_index = (instruction_index + 1) % len(passenger_instructions)

    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring stopped → QUERY MODE\n")

# ====================== TOOL ======================
@tool
def query_event_memory(query: str) -> str:
    """Search logged bus events semantically using natural language."""
    results = collection.query(query_texts=[query], n_results=10)
    if not results['documents'][0]:
        return "No events found in the database."
    return "Logged events:\n" + "\n".join(results['documents'][0])

llm_with_tools = llm.bind_tools([query_event_memory])

# ====================== QUERY MODE ======================
def run_query_mode():
    print("\n" + "="*70)
    print("GEMINI QUERY MODE - Ask about logged events")
    print("="*70)
    
    system_prompt = "You are a bus safety assistant. Use the query_event_memory tool when needed and respond concisely with real data only."
    
    while True:
        user_input = input("\n🧑‍💻 You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nProject complete! Ready Tensor masterpiece 🏆")
            break
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "query_event_memory":
                    result = query_event_memory.invoke(tool_call["args"])
                    messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            final = llm.invoke(messages)
            print(f"🤖 Gemini: {final.content}")
        else:
            print(f"🤖 Gemini: {response.content}")

if __name__ == "__main__":
    run_real_time_monitoring()
    run_query_mode()