import os
import random
import shutil
from pathlib import Path


ROBOFLOW_DATASET = "datasets/cricket_dataset_v1"

REMAINING_DATASET = "datasets/remaining_dataset"

# final dataset
OUTPUT_DATASET = "datasets/cricket_dataset_final"

# Split ratios
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42


def create_folder(path):
    os.makedirs(path, exist_ok=True)


def copy_pair(image_path, label_path, dst_img, dst_lbl):

    shutil.copy2(image_path, dst_img)
    shutil.copy2(label_path, dst_lbl)


for split in ["train", "valid", "test"]:

    create_folder(f"{OUTPUT_DATASET}/{split}/images")
    create_folder(f"{OUTPUT_DATASET}/{split}/labels")


dataset = []


def collect_from_folder(image_folder, label_folder):

    image_folder = Path(image_folder)
    label_folder = Path(label_folder)

    if not image_folder.exists():
        return

    for image in image_folder.iterdir():

        if image.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label = label_folder / (image.stem + ".txt")

        if label.exists():

            dataset.append((image, label))

        else:

            print(f"Missing label -> {image.name}")


for split in ["train", "valid", "test"]:

    collect_from_folder(
        f"{ROBOFLOW_DATASET}/{split}/images",
        f"{ROBOFLOW_DATASET}/{split}/labels"
    )

collect_from_folder(
    f"{REMAINING_DATASET}/images",
    f"{REMAINING_DATASET}/labels"
)

print("=" * 60)
print(f"Total image-label pairs found : {len(dataset)}")
print("=" * 60)


random.seed(RANDOM_SEED)
random.shuffle(dataset)


total = len(dataset)

train_count = int(total * TRAIN_RATIO)
valid_count = int(total * VALID_RATIO)

test_count = total - train_count - valid_count

train_data = dataset[:train_count]
valid_data = dataset[train_count:train_count + valid_count]
test_data = dataset[train_count + valid_count:]



def export_split(split_name, split_data):

    img_out = Path(f"{OUTPUT_DATASET}/{split_name}/images")
    lbl_out = Path(f"{OUTPUT_DATASET}/{split_name}/labels")

    for image, label in split_data:

        copy_pair(
            image,
            label,
            img_out / image.name,
            lbl_out / label.name
        )


export_split("train", train_data)
export_split("valid", valid_data)
export_split("test", test_data)


yaml_text = """train: ../train/images
val: ../valid/images
test: ../test/images

nc: 4

names:
  0: ball
  1: bat
  2: player
  3: stumps
"""

with open(f"{OUTPUT_DATASET}/data.yaml", "w") as f:
    f.write(yaml_text)

print("\nDataset Created Successfully!")
print("=" * 60)

print(f"Train : {len(train_data)}")
print(f"Valid : {len(valid_data)}")
print(f"Test  : {len(test_data)}")

print("=" * 60)

print(f"Output Folder : {OUTPUT_DATASET}")

print("=" * 60)