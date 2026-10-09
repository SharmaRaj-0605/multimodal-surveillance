import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import torch
import pyttsx3
import numpy as np


from ultralytics import YOLO

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

from PIL import Image
model = YOLO("yolov8n.pt")
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model = BlipForConditionalGeneration.from_pretrained(

    
    "Salesforce/blip-image-captioning-base"
)
device = "cuda" if torch.cuda.is_available() else "cpu"

midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"
)

midas.to(device)
midas.eval()

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

transform = midas_transforms.small_transform
engine = pyttsx3.init()
engine.setProperty("rate", 170)

def speak(text):
    engine.say(text)
    engine.runAndWait()


blip_model.to(device)
def run_surveillance():
     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
     if not cap.isOpened():
            print("CRITICAL: The camera backend failed to open. Try restarting your notebook kernel.")
     else:
            print("Camera backend successfully initialized!")
        
     frame_count = 0
        
     caption_text = "No caption yet"
        
     depth_map = None
        
        
     while True:
        
            ret, frame = cap.read()
        
            if not ret:
                break
        
            frame_count += 1
        
            # =========================
            # YOLOv8 DETECTION
            # =========================
            results = model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.35
            )
        
            annotated_frame = results[0].plot()
        
            # =========================
            # PERSON COUNT
            # =========================
            person_count = 0
        
            if results[0].boxes is not None:
        
                boxes = results[0].boxes
        
                for cls_id in boxes.cls:
        
                    if int(cls_id) == 0:
                        person_count += 1
        
            # =========================
            # CROWD ALERT
            # =========================
            if person_count >= 5:
        
                cv2.putText(
                    annotated_frame,
                    "ALERT: Crowd Detected!",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )
        
                if frame_count % 100 == 0:
                    speak("Crowd detected")
        
            # =========================
            # BLIP CAPTIONING
            # =========================
            if frame_count % 120 == 0:
        
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )
        
                image = Image.fromarray(rgb)
        
                inputs = processor(
            images=image,
            return_tensors="pt"
            ).to(device)
        
                output = blip_model.generate(**inputs)
        
                caption_text = processor.decode(
                    output[0],
                    skip_special_tokens=True
                )
        
            # SHOW CAPTION
            cv2.putText(
                annotated_frame,
                f"Scene: {caption_text}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
            # =========================
            # DEPTH ESTIMATION
            # =========================
            if frame_count % 30 == 0:
        
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )
        
                input_batch = transform(rgb).to(device)
        
                with torch.no_grad():
        
                    prediction = midas(input_batch)
        
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False
                    ).squeeze()
        
                depth = prediction.cpu().numpy()
        
                depth = cv2.normalize(
                    depth,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                )
        
                depth = depth.astype("uint8")
        
                depth_map = cv2.applyColorMap(
                    depth,
                    cv2.COLORMAP_MAGMA
                )
        
            # SHOW DEPTH MAP
            if depth_map is not None:
        
                small_depth = cv2.resize(
                    depth_map,
                    (320, 180)
                )
        
                cv2.imshow(
                    "Depth Map",
                    small_depth
                )
        
            # =========================
            # PEOPLE COUNT
            # =========================
            cv2.putText(
                annotated_frame,
                f"People Count: {person_count}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        
            # =========================
            # SHOW OUTPUT
            # =========================
            cv2.imshow(
                "Multimodal Surveillance Intelligence System",
                annotated_frame
            )
        
            key = cv2.waitKey(1)
        
            if key == ord("q"):
                break
        
     cap.release()
        
     cv2.destroyAllWindows()    
        
