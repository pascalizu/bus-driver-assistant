import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
yolo_model = YOLO('yolov8n.pt')

def detect_fatigue(frame):
    """Detect driver fatigue using eye openness."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return False, "No face detected"
    
    for landmarks in results.multi_face_landmarks:
        left_eye = (landmarks.landmark[159].y - landmarks.landmark[145].y)
        right_eye = (landmarks.landmark[386].y - landmarks.landmark[374].y)
        eye_openness = (left_eye + right_eye) / 2
        
        if eye_openness < 0.015:
            return True, "DROWSY - Eyes closed"
    
    return False, "Alert - Eyes open"


def detect_passengers_and_red_hat(frame):
    """Detect passengers and red hat using YOLO + color filter."""
    results = yolo_model(frame, classes=0, verbose=False)
    
    passenger_count = 0
    red_hat_detected = False
    h, w = frame.shape[:2]
    passenger_zone_x = int(w * 0.3)  # Exclude driver area (left side)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if x1 > passenger_zone_x:
                passenger_count += 1
                
                # Check head region for red hat
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