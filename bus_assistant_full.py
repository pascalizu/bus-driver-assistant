import cv2
import mediapipe as mp
from langchain_ollama import ChatOllama
from ultralytics import YOLO
import time
import numpy as np
import chromadb
from datetime import datetime
import os

# --- Chroma Setup (Event Logging) ---
if not os.path.exists("events"):
    os.makedirs("events")
chroma_client = chromadb.PersistentClient(path="events")
collection = chroma_client.get_or_create_collection(name="bus_events")

event_counter = 0  # Define and initialize here

# --- Models ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

yolo_model = YOLO('yolov8n.pt')

llm = ChatOllama(model="gemma2:2b")

print("Full Bus Driver Assistant STARTED (with Event Logging)")
print("Events will be saved to 'events' folder")
print("Features: Fatigue + Passengers + Red Hat + Logging")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Camera failed")
    exit()

print("Camera ready! Monitoring + Logging started...\n")

last_alert_time = 0
alert_cooldown = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    current_time_str = datetime.now().strftime("%H:%M:%S")

    # Fatigue detection
    face_results = face_mesh.process(rgb_frame)
    drowsy = False
    if face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            left_eye = landmarks.landmark[159].y - landmarks.landmark[145].y
            right_eye = landmarks.landmark[386].y - landmarks.landmark[374].y
            eye_openness = (left_eye + right_eye) / 2
            if eye_openness < 0.015:
                drowsy = True

    # Passenger & Red Hat detection
    yolo_results = yolo_model(frame, classes=[0], verbose=False, conf=0.5)
    passenger_count = 0
    red_hat_detected = False

    if yolo_results[0].boxes is not None:
        passenger_count = len(yolo_results[0].boxes)
        for box in yolo_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            head_bottom = y1 + int((y2 - y1) * 0.3)
            head_region = hsv_frame[y1:head_bottom, x1:x2]

            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(head_region, lower_red1, upper_red1)
            mask2 = cv2.inRange(head_region, lower_red2, upper_red2)
            red_mask = mask1 + mask2

            if cv2.countNonZero(red_mask) > 800:
                red_hat_detected = True
                cv2.putText(frame, "RED HAT!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- Log Event to Chroma ---
    event_text = f"Time: {current_time_str} | Passengers: {passenger_count}"
    if drowsy:
        event_text += " | Driver drowsy"
    if red_hat_detected:
        event_text += " | Red hat detected"

    collection.add(
        documents=[event_text],
        ids=[f"event_{event_counter}"],
        metadatas=[{
            "timestamp": current_time_str,
            "passengers": passenger_count,
            "drowsy": drowsy,
            "red_hat": red_hat_detected
        }]
    )
    event_counter += 1  # Increment here

    # --- On-Screen Display ---
    cv2.putText(frame, f"Passengers: {passenger_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Events logged: {event_counter}", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if red_hat_detected:
        cv2.putText(frame, "RED HAT DETECTED!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    if drowsy:
        cv2.putText(frame, "DROWSY! WAKE UP!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        current_time = time.time()
        if current_time - last_alert_time > alert_cooldown:
            print("\n🚨 AI FATIGUE ALERT 🚨")
            response = llm.invoke("In 1-2 short sentences, firmly but calmly warn a bus driver that fatigue has been detected. No questions.")
            print(f"🤖 AI: {response.content}\n")
            last_alert_time = current_time
    else:
        cv2.putText(frame, "Alert & Focused", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Bus Driver Assistant - With Logging', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nLogging complete! {event_counter} events saved in 'events' folder.")
print("You are now ready for the multi-agent system!")