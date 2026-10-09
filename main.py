import os
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

model = YOLO("yolov8s.pt")

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


processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model = BlipForConditionalGeneration.from_pretrained(

    
    "Salesforce/blip-image-captioning-base"
)
blip_model = blip_model.to(device)


engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()


import time
def run_surveillance():
 prev_time = time.time()
 fps = 0
 frame_count = 0
 cap = cv2.VideoCapture(0)
 depth_map = None
 depth = None
 caption_text = "No caption yet"

 while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
     # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # YOLO + TRACKING
    
    results = model.track(
        frame,
        persist=True,
        verbose=False,
        conf=0.35
    )

    annotated_frame = results[0].plot()

    person_count = 0
    near_people = 0
    cv2.rotate(frame, 1)
 
    # DEPTH ESTIMATION
   
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

   
    # PERSON ANALYSIS + DISTANCE
   
    if results[0].boxes is not None:

        boxes = results[0].boxes

        for box, cls_id in zip(
            boxes.xyxy,
            boxes.cls
        ):

            if int(cls_id) != 0:
                continue

            person_count += 1

            x1, y1, x2, y2 = map(
                int,
                box
            )

            if depth is not None:

                roi = depth[
                    y1:y2,
                    x1:x2
                ]

                if roi.size > 0:

                    avg_depth = np.mean(roi)

                    if avg_depth > 200:
                        label = "NEAR"
                        near_people += 1

                    elif avg_depth > 100:
                        label = "MEDIUM"

                    else:
                        label = "FAR"

                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

    
    # CROWD ALERT
    
    if person_count >= 5:

        cv2.putText(
            annotated_frame,
            "ALERT: CROWD DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        if frame_count % 100 == 0:
            speak("Crowd detected")

   
    # PROXIMITY ALERT
   
    if near_people > 0:

        cv2.putText(
            annotated_frame,
            f"NEARBY PEOPLE: {near_people}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if frame_count % 60 == 0:
            speak("Person close to camera")

   
    # BLIP CAPTION
    
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

        output = blip_model.generate(
            **inputs
        )

        caption_text = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        caption_text += (
            f". Detected {person_count} people."
        )

        if near_people > 0:
            caption_text += (
                f" {near_people} person near camera."
            )

    
    # DISPLAY TEXT
    
    cv2.putText(
        annotated_frame,
        f"Scene: {caption_text}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"People Count: {person_count}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    # DEPTH WINDOW
    
    if depth_map is not None:
        small_depth = cv2.resize(
           depth_map,
            (320, 180)
        )
        cv2.imshow("Depth Map", small_depth)


 
    cv2.putText(
          annotated_frame,
          f"FPS: {fps:.2f}",
           (20, 210),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

   
   
    # MAIN WINDOW
   
    cv2.imshow(
        "Multimodal Surveillance Intelligence System",
        annotated_frame
        )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

 cap.release()
 cv2.destroyAllWindows()
        

       

    
