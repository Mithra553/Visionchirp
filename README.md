# 🎯 VisionChirp - Smart Assistant for the Visually Impaired

VisionChirp is an AI-powered assistant that combines **Object Detection** and **Text Recognition (OCR)** with real-time speech output to assist visually impaired users.

---

## 👁️ Features

- ✅ Real-time **Object Detection** using YOLOv8
- ✅ Accurate **Text Recognition** using EasyOCR
- ✅ Clear **Speech Output** with pyttsx3 (offline)
- ✅ Intelligent updates: Speaks only when new info is detected
- ✅ Multithreading for smooth performance

---

## 🛠️ Technologies Used

- 🔍 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection  
- 🔠 [EasyOCR](https://github.com/JaidedAI/EasyOCR) for reading text from images  
- 🗣️ [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for offline speech output  
- 🎥 OpenCV for real-time camera feed  
- 🧵 Python threading and queues for efficiency  

---

## 🚀 How It Works

1. The camera captures a frame.
2. YOLOv8 detects objects like person, bottle, laptop, etc.
3. EasyOCR reads any visible text in the frame.
4. If the detected objects or text change, the app **speaks it aloud**.
5. Runs completely **offline** — no internet required!

---

## ▶️ Run It Locally

### 1. Install Requirements

```bash
pip install ultralytics opencv-python easyocr pyttsx3
