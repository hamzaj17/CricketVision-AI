from ultralytics import YOLO
import os
import shutil
from pathlib import Path

MODEL_PATH = "models/cricketvision_v1.pt"

SOURCE_FOLDER = "datasets/rejected_frames"
OUTPUT_DATASET = "datasets/auto_label_dataset"

CONFIDENCE = 0.15


IMAGES_FOLDER = os.path.join(OUTPUT_DATASET, "images")
LABELS_FOLDER = os.path.join(OUTPUT_DATASET, "labels")

os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)

print("=" * 60)
print("Copying Images...")
print("=" * 60)

image_extensions = (".jpg", ".jpeg", ".png")

image_files = []

for file in os.listdir(SOURCE_FOLDER):
    if file.lower().endswith(image_extensions):

        src = os.path.join(SOURCE_FOLDER, file)
        dst = os.path.join(IMAGES_FOLDER, file)

        shutil.copy2(src, dst)
        image_files.append(file)

print(f"Copied {len(image_files)} images.")

print("\nRunning YOLO Auto-Labeling...\n")

model = YOLO(MODEL_PATH)

model.predict(
    source=IMAGES_FOLDER,
    save=False,
    save_txt=True,
    save_conf=False,
    conf=CONFIDENCE,
    project=OUTPUT_DATASET,
    name="temp_predictions",
    exist_ok=True,
    verbose=True
)


generated_labels = os.path.join(
    OUTPUT_DATASET,
    "temp_predictions",
    "labels"
)

if os.path.exists(generated_labels):

    for file in os.listdir(generated_labels):

        shutil.move(
            os.path.join(generated_labels, file),
            os.path.join(LABELS_FOLDER, file)
        )


temp_folder = os.path.join(
    OUTPUT_DATASET,
    "temp_predictions"
)

if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)


num_images = len(os.listdir(IMAGES_FOLDER))
num_labels = len(os.listdir(LABELS_FOLDER))

print("\n" + "=" * 60)
print("Auto Label Dataset Created")
print("=" * 60)

print(f"Images           : {num_images}")
print(f"Label Files      : {num_labels}")
print(f"No Detections    : {num_images - num_labels}")

print("\nDataset Location:")
print(OUTPUT_DATASET)

print("=" * 60)