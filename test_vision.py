import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Open webcam (0 = default camera). You can change 0 to a video file later.
cap = cv2.VideoCapture(0)

print("Face detection started! Look at the camera.")
print("Press 'q' on your keyboard to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't access camera. Check if it's connected.")
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        print(f"Found {len(results.detections)} face(s)!")
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Draw green rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, "FACE DETECTED", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the video window
    cv2.imshow('Bus Driver Assistant - Face Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Stopped. Goodbye!")