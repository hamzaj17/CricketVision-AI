import cv2
import os
import shutil

INPUT_FOLDER = "datasets/extracted_frames"
REJECTED_FOLDER = "datasets/rejected_frames"

os.makedirs(REJECTED_FOLDER, exist_ok=True)

images = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

total = len(images)

if total == 0:
    print("No images found.")
    exit()

print("=" * 60)
print("CricketVision Dataset Reviewer")
print("=" * 60)
print("Controls:")
print("K = Keep")
print("D = Reject (move to rejected folder)")
print("Q = Quit")
print("=" * 60)

kept = 0
rejected = 0

for i, image_name in enumerate(images, start=1):

    path = os.path.join(INPUT_FOLDER, image_name)

    image = cv2.imread(path)

    if image is None:
        continue

    display = image.copy()

    cv2.putText(
        display,
        f"{i}/{total}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        image_name,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Dataset Reviewer", display)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('k'):

        kept += 1

    elif key == ord('d'):

        shutil.move(
            path,
            os.path.join(REJECTED_FOLDER, image_name)
        )

        rejected += 1

    elif key == ord('q'):

        break

cv2.destroyAllWindows()

print("\nReview Finished")
print(f"Kept      : {kept}")
print(f"Rejected  : {rejected}")