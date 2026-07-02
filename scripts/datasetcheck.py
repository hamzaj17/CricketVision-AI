import os
import cv2

folders = [
    "cricket.v1i.yolov8/train/images",
    "cricket.v1i.yolov8/valid/images",
    "cricket.v1i.yolov8/test/images",
]

bad = []

for folder in folders:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            bad.append(path)

print("Bad images:", bad)