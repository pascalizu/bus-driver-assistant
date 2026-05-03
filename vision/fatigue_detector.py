import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

last_alert_time = 0
alert_cooldown = 30


def detect_fatigue(frame):
    """
    Returns:
        drowsy (bool)
    """

    global last_alert_time

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    drowsy = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            left_eye = (
                face_landmarks.landmark[159].y -
                face_landmarks.landmark[145].y
            )
            right_eye = (
                face_landmarks.landmark[386].y -
                face_landmarks.landmark[374].y
            )

            eye_openness = (left_eye + right_eye) / 2

            if eye_openness < 0.015:
                drowsy = True

    return drowsy