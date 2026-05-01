import cv2
import mediapipe as mp
from langchain_ollama import ChatOllama
from ultralytics import YOLO
import time

# Models
face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
yolo_model = YOLO('yolov8n.pt')  # Tiny YOLO - auto-downloads first time
llm = ChatOllama(model="gemma2:2b")  # Your fast, good model

print("Full Bus Driver Assistant STARTED")
print("Features: Fatigue detection + Passenger counting")
print("Close eyes → AI warning | Look for people → count on screen")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_alert_time = 0
alert_cooldown = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Fatigue detection
    face_results = face_mesh.process(rgb_frame)
    drowsy = False
    if face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            left = (landmarks.landmark[159].y - landmarks.landmark[145].y)
            right = (landmarks.landmark[386].y - landmarks.landmark[374].y)
            eye_openness = (left + right) / 2
            if eye_openness < 0.015:
                drowsy = True

    # Passenger counting with YOLO
    yolo_results = yolo_model(frame, classes=[0], verbose=False)  # class 0 = person
    passenger_count = len(yolo_results[0].boxes)
    
    # Draw count on screen
    cv2.putText(frame, f"Passengers: {passenger_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    # Fatigue warning
    if drowsy:
        cv2.putText(frame, "DROWSY! WAKE UP!", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        current_time = time.time()
        if current_time - last_alert_time > alert_cooldown:
            print("\n🚨 AI FATIGUE ALERT 🚨")
            response = llm.invoke("In 1-2 short sentences only, calmly but firmly warn a bus driver showing signs of fatigue. No questions or extra text.")
            print(f"🤖 AI: {response.content}\n")
            last_alert_time = current_time
    else:
        cv2.putText(frame, "ALERT & FOCUSED", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('Bus Driver Assistant - Fatigue + Passengers', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Assistant stopped. Great job today!")