import cv2
import os
from collections import Counter

folder = "datasets/extracted_frames"

sizes = Counter()

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    img = cv2.imread(path)
    if img is None:
        continue
    h, w = img.shape[:2]
    sizes[(w, h)] += 1

print("Image resolutions:")
for (w, h), count in sizes.items():
    print(f"{w} x {h} : {count}")