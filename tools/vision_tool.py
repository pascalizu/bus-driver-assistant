from langchain_core.tools import tool
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize models (loaded once)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
yolo_model = YOLO('yolov8n.pt')

@tool
def computer_vision_tool(task: str = "analyze") -> str:
    """Analyze current camera frame for driver fatigue, passenger count, and red hat detection."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return "Could not access camera."

    # Fatigue Detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    drowsy = False
    if results.multi_face_landmarks:
        for lm in results.multi_face_landmarks:
            left = lm.landmark[159].y - lm.landmark[145].y
            right = lm.landmark[386].y - lm.landmark[374].y
            if (left + right) / 2 < 0.015:
                drowsy = True

    # Passenger & Red Hat Detection
    results = yolo_model(frame, classes=0, verbose=False)
    passengers = 0
    red_hat = False
    h, w = frame.shape[:2]
    passenger_zone = int(w * 0.3)
    
    for r in results:
        for box in r.boxes:
            x1 = int(box.xyxy[0][0])
            if x1 > passenger_zone:
                passengers += 1
                # Check head region for red color
                y1, y2 = int(box.xyxy[0][1]), int(box.xyxy[0][3])
                head = frame[y1:int(y1 + (y2 - y1)*0.25), x1:int(box.xyxy[0][2])]
                if head.size > 0:
                    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array([0,70,50]), np.array([10,255,255])) + \
                           cv2.inRange(hsv, np.array([170,70,50]), np.array([180,255,255]))
                    if cv2.countNonZero(mask) / head.size > 0.18:
                        red_hat = True

    status = f"Driver is {'drowsy' if drowsy else 'alert'}. {passengers} passengers detected."
    if red_hat:
        status += " Red hat detected."
    return status