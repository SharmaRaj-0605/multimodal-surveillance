# multimodal-surveillance
🎥 Real-time AI surveillance — YOLOv8 + depth estimation + scene captioning,  fused into one live pipeline with voice alerts.
 An intelligent real-time surveillance system that **sees**, **understands**, and **speaks** — combining BLIP image captioning, people counting, depth estimation, and voice alerts into one unified pipeline.

---

## 📌 Overview

Traditional surveillance systems only record footage. This system actively interprets it:

- 📸 **Scene Understanding** — BLIP generates a natural language description of what's happening in frame
- 🔢 **People Counting** — Detects and counts individuals in real time
- 📏 **Depth Estimation** — Classifies how close detected people are (near / mid / far)
- 🔊 **Voice Alerts** — Speaks out loud when someone is detected nearby

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🖼️ BLIP Captioning | Auto-generates scene captions from live video frames |
| 👥 People Counting | Real-time person detection using object detection model |
| 📏 Depth Estimation | Estimates proximity using monocular depth model |
| 🔔 Proximity Alerts | Triggers voice alert: *"Near person detected!"* |
| 🔊 Text-to-Speech | Speaks captions and alerts aloud |
| 🎥 Live Feed Support | Works with webcam or pre-recorded video |

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Captioning:** BLIP (Bootstrapped Language-Image Pretraining) — HuggingFace Transformers
- **Detection:** OpenCV / YOLOv8
- **Depth Estimation:** MiDaS 
- **Text-to-Speech:** pyttsx3 
  

---



## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/SharmaRaj-0605/multimodal-surveillance-system.git
cd multimodal-surveillance-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run on webcam
```bash
python pipeline.py --source 0
```

### 4. Run on a video file
```bash
python pipeline.py --source path/to/video.mp4
```

---

## 🔄 System Pipeline

```
Video Frame
    │
    ├──► BLIP Captioning ──────► "Two people walking near the entrance"
    │                                          │
    ├──► People Counter ──────► Count: 2       │
    │                                          ▼
    └──► Depth Estimator ─────► "NEAR" ──► 🔊 Voice Alert
```

---

## 📷 Sample Output

> *(Add demo GIF or screenshot of the live feed with overlays here)*

---

## 📄 References

- [BLIP Paper](https://arxiv.org/abs/2201.12086)
- [MiDaS Depth Estimation](https://github.com/isl-org/MiDaS)
- [YOLOv8](https://github.com/ultralytics/ultralytics)

---

## 👤 Author

**Raj Sharma**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/raj-sharma-908465403)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/SharmaRaj-0605)
