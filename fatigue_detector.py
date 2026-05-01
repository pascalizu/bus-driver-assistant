import cv2
import mediapipe as mp
from langchain_ollama import ChatOllama
import time

# Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Use a small fast model
llm = ChatOllama(model="gemma2:2b")

print("Bus Driver Fatigue Assistant STARTED")
print("Close your eyes for 2-3 seconds → AI will give ONE warning")
print("Open eyes → alert clears")
print("Press 'q' in the video window to quit\n")

print("Trying to open camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera 0 failed, trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera 1 failed too — check if webcam is in use by another app.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Camera opened successfully!")
last_alert_time = 0
alert_cooldown = 30  # seconds between alerts

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera not accessible")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status_text = "ALERT & FOCUSED"
    status_color = (0, 255, 0)
    drowsy = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Simple eye openness measure
            left_eye = (face_landmarks.landmark[159].y - face_landmarks.landmark[145].y)
            right_eye = (face_landmarks.landmark[386].y - face_landmarks.landmark[374].y)
            eye_openness = (left_eye + right_eye) / 2

            if eye_openness < 0.015:  # Eyes closed
                drowsy = True

    if drowsy:
        status_text = "DROWSY! WAKE UP!"
        status_color = (0, 0, 255)

        current_time = time.time()
        if current_time - last_alert_time > alert_cooldown:
            print("\n🚨 AI FATIGUE ALERT 🚨")
            response = llm.invoke("In a short, friendly but firm voice, warn a bus driver showing signs of fatigue.")
            print(f"🤖 AI: {response.content}\n")
            last_alert_time = current_time

    else:
        status_text = "ALERT & FOCUSED"
        status_color = (0, 255, 0)

    cv2.putText(frame, status_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 4)
    cv2.imshow('Bus Driver Fatigue Assistant', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nAssistant stopped. Drive safe! 🚍")