# Bus Driver Assistant: Multi-Agent Vision AI System

**Real-Time Driver Fatigue Monitoring | Passenger Safety | Voice Alerts | Intelligent Querying**

![Real-Time Camera Feed](screenshots/camera_feed.png)

---

## 📋 Table of Contents
- [Overview](#overview)
- [System in Action](#system-in-action)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation & Usage](#installation--usage)
- [Significance](#significance)

---

## 🎯 Overview

The **Bus Driver Assistant** is a modular multi-agent AI system designed to enhance road safety in public transportation by monitoring driver fatigue, passenger activity, and providing intelligent voice and query capabilities.

**Ready Tensor Module 2 Project**

---

## 📸 System in Action

### Real-Time Monitoring Interface
![Real-Time Camera Feed](screenshots/camera_feed.png)

### Terminal + Voice + Query Mode
![Terminal Output](screenshots/terminal_output.png)

---

## ✨ Key Features

- Real-time fatigue detection using Eye Aspect Ratio (MediaPipe)
- Passenger counting and red hat detection (YOLOv8)
- Automatic & manual voice alerts (`pyttsx3`)
- Multi-agent system with LangGraph
- Semantic memory using ChromaDB
- Natural language querying

---

## 🏗 System Architecture

```mermaid
flowchart TD
    A[Webcam] --> B[Vision Tool]
    B --> C[Supervisor Agent]
    C --> D[Driver Agent]
    C --> E[Passenger Agent]
    C --> F[Query Agent]
    D & E --> G[Chroma DB]
    F --> G
    D & E --> H[Voice Agent]