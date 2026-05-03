# Bus Driver Assistant: Multi-Agent Vision AI with Voice Alerts

A real-time intelligent safety system for buses that monitors driver fatigue, tracks passengers, detects notable objects, provides automatic and manual voice announcements, and supports natural language queries over logged events.

## 🎯 Project Overview

The **Bus Driver Assistant** is a multimodal AI system designed to enhance safety and operational efficiency in public and private buses. It combines computer vision, voice interaction, event logging, and intelligent querying to assist drivers and improve passenger safety.

### Key Features
- **Real-time Driver Fatigue Detection** using MediaPipe eye landmark analysis
- **Passenger Counting & Red Hat Detection** using YOLOv8 + HSV color filtering
- **Automatic Voice Alert** for drowsy driver (with cooldown)
- **Manual Voice Announcements** for passengers (press 'v' during monitoring)
- **Event Logging** with Chroma vector database
- **Gemini 3 Flash Powered Query Mode** — ask natural language questions about logged events
- **Offline Voice Output** using pyttsx3

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/pascalizu/bus-driver-assistant.git
cd bus-driver-assistant