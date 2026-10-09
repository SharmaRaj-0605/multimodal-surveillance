import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pyttsx3
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
from ultralytics import YOLO


model = YOLO("yolov8s.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

engine = pyttsx3.init()
engine.setProperty("rate", 150)


def speak(text):
    engine.say(text)
    engine.runAndWait()


def run_surveillance(max_seconds=None):
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        cap.release()
        return

    run_start = time.perf_counter()
    frame_count = 0
    total_latency_ms = 0.0
    depth_map = None
    depth = None
    caption_text = "No caption yet"

    try:
        while True:
            if (
                max_seconds is not None
                and time.perf_counter() - run_start >= max_seconds
            ):
                break

            ret, frame = cap.read()
            if not ret:
                print("Webcam frame could not be read; stopping surveillance.")
                break

            frame_start = time.perf_counter()
            frame_count += 1

            # Person detection and tracking
            results = model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.35,
            )
            annotated_frame = results[0].plot()

            person_count = 0
            near_people = 0

            # Update the depth map every 30 frames
            if frame_count % 30 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_batch = transform(rgb).to(device)

                with torch.no_grad():
                    prediction = midas(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                depth = prediction.cpu().numpy()
                depth = cv2.normalize(
                    depth, None, 0, 255, cv2.NORM_MINMAX
                ).astype("uint8")
                depth_map = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)

            # Count people and estimate relative proximity from the depth map
            if results[0].boxes is not None:
                for box, cls_id in zip(
                    results[0].boxes.xyxy, results[0].boxes.cls
                ):
                    if int(cls_id) != 0:
                        continue

                    person_count += 1
                    x1, y1, x2, y2 = map(int, box)

                    if depth is not None:
                        roi = depth[y1:y2, x1:x2]
                        if roi.size > 0:
                            avg_depth = float(np.mean(roi))

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
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                            )

            # Crowd alert
            if person_count >= 5:
                cv2.putText(
                    annotated_frame,
                    "ALERT: CROWD DETECTED",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                if frame_count % 100 == 0:
                    speak("Crowd detected")

            # Proximity alert
            if near_people > 0:
                cv2.putText(
                    annotated_frame,
                    f"NEARBY PEOPLE: {near_people}",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                if frame_count % 60 == 0:
                    speak("Person close to camera")

            # Generate a scene caption every 120 frames
            if frame_count % 120 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                inputs = processor(images=image, return_tensors="pt").to(device)

                with torch.no_grad():
                    output = blip_model.generate(**inputs)

                caption_text = processor.decode(
                    output[0], skip_special_tokens=True
                )
                caption_text += f". Detected {person_count} people."
                if near_people > 0:
                    caption_text += f" {near_people} person near camera."

            # Frame annotations
            cv2.putText(
                annotated_frame,
                f"Scene: {caption_text}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"People Count: {person_count}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # Synchronize pending GPU work before stopping the latency timer.
            if device == "cuda":
                torch.cuda.synchronize()

            frame_latency_ms = (time.perf_counter() - frame_start) * 1000
            total_latency_ms += frame_latency_ms
            elapsed_seconds = max(time.perf_counter() - run_start, 1e-9)
            avg_fps = frame_count / elapsed_seconds
            avg_latency_ms = total_latency_ms / frame_count

            cv2.putText(
                annotated_frame,
                f"Average FPS: {avg_fps:.2f}",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Avg Processing Latency: {avg_latency_ms:.1f} ms",
                (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            # Display the depth map when one is available
            if depth_map is not None:
                small_depth = cv2.resize(depth_map, (320, 180))
                cv2.imshow("Depth Map", small_depth)

            cv2.imshow("Multimodal Surveillance Intelligence System", annotated_frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    finally:
        elapsed_total = time.perf_counter() - run_start
        cap.release()
        cv2.destroyAllWindows()

        if frame_count > 0:
            print("\n--- Surveillance Benchmark ---")
            print(f"Frames processed: {frame_count}")
            print(f"Test duration: {elapsed_total:.2f} seconds")
            print(
                "Average loop throughput: "
                f"{frame_count / max(elapsed_total, 1e-9):.2f} FPS"
            )
            print(
                "Average processing latency: "
                f"{total_latency_ms / frame_count:.2f} ms"
            )
            print(f"Device: {device}")


       

    
