# Bus Driver Assistant: Multi-Agent Vision AI System for Road Safety

**Real-Time Driver Fatigue Monitoring | Passenger Safety | Voice Alerts | Intelligent Querying**

![Bus Driver Assistant Banner](https://github.com/pascalizu/bus-driver-assistant/raw/main/screenshots/banner.png)

---

## 📋 Table of Contents
- [Overview](#overview)
- [System in Action](#system-in-action)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Performance Evaluation](#performance-evaluation)
- [Installation & Usage](#installation--usage)
- [Significance & Impact](#significance--impact)

---

## 🎯 Overview

The **Bus Driver Assistant** is a modular multi-agent AI system designed to enhance safety in public transportation. It monitors driver fatigue in real-time, tracks passengers, provides voice alerts, and supports natural language queries over logged events.

**Ready Tensor Module 2 Project** — Demonstrating Multi-Agent Systems, Computer Vision, Tool Integration, and Memory Management.

---

## 📸 System in Action

### Real-Time Monitoring Interface
![Real-Time Camera Feed](https://github.com/pascalizu/bus-driver-assistant/raw/main/screenshots/camera_feed.png)

### Terminal + Voice + Query Mode
![Terminal Output](https://github.com/pascalizu/bus-driver-assistant/raw/main/screenshots/terminal_output.png)

---

## ✨ Key Features

- Real-time driver fatigue detection using Eye Aspect Ratio (MediaPipe)
- Passenger counting and red hat detection with YOLOv8
- Automatic & manual voice alerts using pyttsx3
- Semantic event logging with ChromaDB
- 5-Agent architecture orchestrated with LangGraph
- Natural language querying

---

## 🏗 System Architecture

```mermaid
flowchart TD
    A[Webcam Feed] --> B[Vision Tool]
    B --> C[Supervisor Agent]
    C --> D[Driver Agent]
    C --> E[Passenger Agent]
    C --> F[Query Agent]
    D & E --> G[Chroma DB]
    F --> G
    D & E --> H[Voice Agent]