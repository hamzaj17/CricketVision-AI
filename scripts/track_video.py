from ultralytics import YOLO
import cv2
import os
import time

MODEL_PATH = "models/cricketvision_v1.pt"
INPUT_FOLDER = "video_data/inputs"
OUTPUT_FOLDER = "video_data/outputs/tracking"

TRACKER = "bytetrack.yaml"

CONFIDENCE = 0.30


os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

video_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
]

print("=" * 60)
print("CricketVision AI - Object Tracking")
print("=" * 60)

for video in video_files:

    input_path = os.path.join(INPUT_FOLDER, video)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"tracked_{video}"
    )

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = 0

    start = time.time()

    print(f"\nProcessing: {video}")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            conf=CONFIDENCE,
            verbose=False
        )

        annotated = results[0].plot()

        writer.write(annotated)

        frame_count += 1

        if frame_count % 50 == 0:
            print(f"Frames Processed : {frame_count}")

    cap.release()
    writer.release()

    elapsed = time.time() - start

    print(f"Finished : {video}")
    print(f"Frames   : {frame_count}")
    print(f"Time     : {elapsed:.2f} sec")

print("\nTracking Completed Successfully!")