from ultralytics import YOLO
import cv2
import os
import time

# ======================================================
# Configuration
# ======================================================

MODEL_PATH = "runs/detect/train-10/weights/best.pt"

INPUT_FOLDER = "video_data"

OUTPUT_FOLDER = "outputs/videos"

CONFIDENCE_THRESHOLD = 0.35

# ======================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

video_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
]

print("=" * 60)
print("CricketVision AI - Video Detection")
print("=" * 60)
print(f"Videos Found : {len(video_files)}")
print()

for video_name in video_files:

    input_path = os.path.join(INPUT_FOLDER, video_name)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"detected_{video_name}"
    )

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0

    start = time.time()

    print(f"Processing: {video_name}")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        frame_count += 1

        if frame_count % 50 == 0:
            print(f"Frames Processed: {frame_count}")

    cap.release()
    out.release()

    elapsed = time.time() - start

    print(f"Finished : {video_name}")
    print(f"Frames   : {frame_count}")
    print(f"Time     : {elapsed:.2f} sec")
    print()

print("=" * 60)
print("All Videos Processed Successfully!")
print(f"Results Saved To: {OUTPUT_FOLDER}")
print("=" * 60)