import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import easyocr
import threading
import queue
import time

# === Load models ===
print("🔁 Loading models...")
yolo_model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'], gpu=False)
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# === Speech Queue and Thread ===
speech_queue = queue.Queue()

def speak_loop():
    while True:
        text = speech_queue.get()
        if text:
            engine.say(text)
            engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speak_loop, daemon=True)
speech_thread.start()

# === State Variables ===
last_objects = []
last_texts = []
last_spoken_time = 0
speak_cooldown = 6  # seconds

# === Start camera ===
cap = cv2.VideoCapture(0)

print("🎥 Starting VisionChirp...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === YOLOv8 Object Detection ===
    results = yolo_model.predict(frame, imgsz=640, conf=0.5)
    boxes = results[0].boxes
    detected_labels = []

    for box in boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.model.names[cls_id]
        detected_labels.append(label)
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
        cv2.putText(frame, label, tuple(xyxy[:2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # === OCR Detection ===
    ocr_texts = []
    results_ocr = reader.readtext(frame)
    for (bbox, text, prob) in results_ocr:
        if prob > 0.5:
            ocr_texts.append(text)
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # === Speak only if new objects or texts or timeout ===
    current_time = time.time()
    new_objects_detected = detected_labels != last_objects
    new_texts_detected = ocr_texts != last_texts

    if new_objects_detected or new_texts_detected or (current_time - last_spoken_time > speak_cooldown):
        speech = ""
        if detected_labels:
            speech += "I see: " + ", ".join(sorted(detected_labels)) + ". "
        if ocr_texts:
            speech += "Text reads: " + ", ".join(sorted(ocr_texts)) + ". "
        if speech.strip():
            print("🔊 Speaking:", speech)
            speech_queue.put(speech)
            last_spoken_time = current_time
            last_objects = detected_labels.copy()
            last_texts = ocr_texts.copy()

    # === Display window ===
    cv2.imshow("VisionChirp", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()