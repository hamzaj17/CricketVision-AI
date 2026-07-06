from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "models/cricketvision_final.pt"

VIDEO_FOLDER = "test_videos"

OUTPUT_FOLDER = "outputs"

CONFIDENCE = 0.20
IMAGE_SIZE = 960

model = YOLO(MODEL_PATH)

videos = list(Path(VIDEO_FOLDER).glob("*"))

print(f"\nFound {len(videos)} videos\n")

for video in videos:

    print(f"Processing {video.name}")

    model.predict(
        source=str(video),
        save=True,
        save_txt=False,
        save_conf=False,
        project=OUTPUT_FOLDER,
        name=video.stem,
        exist_ok=True,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        stream=False,
        verbose=False
    )

print("\nDone!")