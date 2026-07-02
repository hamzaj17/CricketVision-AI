from ultralytics import YOLO
import cv2
import os
import time

MODEL_PATH = "models/cricketvision_v1.pt"

INPUT_FOLDER = "videos/input"

OUTPUT_FOLDER = "videos/output"

CONFIDENCE = 0.35

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

video_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
]

print("=" * 60)
print("CricketVision AI - Object Tracking")
print("=" * 60)

for video_name in video_files:

    input_path = os.path.join(INPUT_FOLDER, video_name)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "tracked_" + video_name
    )

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = 0

    start = time.time()

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            verbose=False
        )

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        frame_count += 1

        if frame_count % 100 == 0:
            print(f"{video_name} : {frame_count} frames")

    cap.release()
    out.release()

    elapsed = time.time() - start

    print(f"\nFinished : {video_name}")
    print(f"Frames   : {frame_count}")
    print(f"Time     : {elapsed:.2f} sec\n")

print("=" * 60)
print("Tracking Completed!")
print("=" * 60)