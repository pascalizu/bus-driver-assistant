# Bus Driver Assistant: Multi-Agent Vision AI System for Road Safety

**Real-Time Driver Fatigue Monitoring | Passenger Safety | Voice Alerts | Intelligent Querying**

![Bus Driver Assistant Banner](https://github.com/pascalizu/bus-driver-assistant/raw/main/screenshots/banner.png)

---

## 📋 Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)
- [Installation & Usage](#installation--usage)
- [License](#license)

---

### Abstract

This project presents **Bus Driver Assistant**, a modular multi-agent AI system developed for real-time safety monitoring in public transportation. The system leverages computer vision and intelligent agents to detect driver fatigue, monitor passengers, and deliver timely voice alerts.

The system utilizes **MediaPipe Face Mesh** for Eye Aspect Ratio (EAR)-based fatigue detection and **YOLOv8** for real-time passenger counting and object detection. A **LangGraph-based 5-agent architecture** (Supervisor, Driver, Passenger, Operations, and Query Agents) coordinates decision-making. Offline voice alerts are generated using `pyttsx3`, while **ChromaDB** enables semantic memory and natural language querying over logged events.

**Keywords**: Multi-Agent System, Computer Vision, Driver Fatigue Detection, LangGraph, Real-time Monitoring, Road Safety, Gemini AI

---

### Introduction

Road safety remains a critical global challenge, with driver fatigue being one of the leading causes of accidents in public transportation. The **Bus Driver Assistant** was developed to address this issue by creating an intelligent, affordable, and proactive AI-powered safety system.

This system actively monitors driver alertness, tracks passengers, delivers voice alerts, and supports natural language interaction through a multi-agent framework powered by LangGraph and Gemini 3 Flash.

---

### Related Work

Driver fatigue detection has evolved from physiological sensors to vision-based methods. Notable works include Ji et al. (2004) on eye tracking and Bergasa et al. (2006) on real-time vigilance monitoring. Recent advances use CNNs and facial landmarks. YOLO models have been applied for passenger detection in public transport.

Multi-agent frameworks like LangGraph are relatively new in safety applications. This project combines these technologies into a unified real-time system.

---

### Methodology

The system is built using **LangGraph** for multi-agent orchestration. Computer vision is handled by MediaPipe (EAR calculation) and YOLOv8. Voice output uses pyttsx3, and memory is managed by ChromaDB.

**Key Code Snippet - Vision Tool:**

```python
@tool
def computer_vision_tool(task: str = "analyze") -> str:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cap.release()
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if results.multi_face_landmarks:
        for lm in results.multi_face_landmarks:
            left = lm.landmark[159].y - lm.landmark[145].y
            right = lm.landmark[386].y - lm.landmark[374].y
            ear = (left + right) / 2
            if ear < 0.018:
                return "Driver is DROWSY"
    return "Driver is alert"