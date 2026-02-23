

# VisionHat_AI
VisionHat is a wearable, real-time computer vision assistant built on Raspberry Pi. The prototype allows visually impaired people to better understand their surroundings by hearing real-time descriptions of nearby objects through audio feedback.
The system runs fully offline and automatically starts on system boot.

## Overview
VisionHat captures live camera frames, performs real-time inference, and converts detected objects into spoken descriptions. All processing happens locally on the Raspberry Pi with no cloud dependency. Our prototype allows visually impaired people to 

Designed for wearable deployment (hat-mounted camera + portable power). This project was developed in under 48 hours as part of HackED 2026, a hackathon sprint focused on rapid prototyping and real-world impact.

## Core Features
- Real-time object detection (YOLO-based)
  
- Edge AI deployment on Raspberry Pi

- Bluetooth A2DP audio output

- Auto-start on boot via systemd

- Fully offline inference (no internet required)

## System Architecture
- Camera → YOLO Model → Detection Parsing → Text-to-Speech → Bluetooth Audio Output

- Camera captures live frames

- Computer Vision YOLO model performs object detection

- Detected objects are converted to speech-ready text

- Audio is streamed to Bluetooth headphones/speaker

## Real-World Impact
VisionHat can assist visually impaired users by providing real-time awareness of nearby objects such as people, obstacles, vehicles, and everyday items. By running fully offline, it ensures privacy, low latency, and reliability without requiring internet connectivity. The wearable form factor enables hands-free, continuous environmental feedback in daily life.
