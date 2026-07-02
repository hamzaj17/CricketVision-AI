import cv2
import os
import shutil

# ==========================================================
# Configuration
# ==========================================================

SOURCE_FOLDER = "datasets/extracted_frames"
DESTINATION_FOLDER = "datasets/initial_100"

os.makedirs(DESTINATION_FOLDER, exist_ok=True)

# ==========================================================
# Load Images
# ==========================================================

images = sorted([
    f for f in os.listdir(SOURCE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if len(images) == 0:
    print("No images found.")
    exit()

selected = set(
    os.listdir(DESTINATION_FOLDER)
)

index = 0

print("=" * 65)
print("CricketVision AI - Initial Dataset Selector")
print("=" * 65)
print("Controls")
print("S = Select")
print("N = Next / Skip")
print("B = Previous")
print("Q = Quit")
print("=" * 65)

# ==========================================================
# Main Loop
# ==========================================================

while 0 <= index < len(images):

    image_name = images[index]

    image_path = os.path.join(SOURCE_FOLDER, image_name)

    image = cv2.imread(image_path)

    if image is None:
        index += 1
        continue

    display = image.copy()

    # Resize only for display if image is too large
    max_width = 1200
    if display.shape[1] > max_width:
        scale = max_width / display.shape[1]
        display = cv2.resize(
            display,
            (
                int(display.shape[1] * scale),
                int(display.shape[0] * scale)
            )
        )

    cv2.putText(
        display,
        f"Image: {index + 1}/{len(images)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Selected: {len(selected)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        image_name,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow("Initial Dataset Selector", display)

    key = cv2.waitKey(0) & 0xFF

    # ------------------------------------------------------
    # Select
    # ------------------------------------------------------

    if key == ord("s"):

        if image_name not in selected:

            shutil.copy2(
                image_path,
                os.path.join(DESTINATION_FOLDER, image_name)
            )

            selected.add(image_name)

            print(f"[SELECTED] {image_name}")

        index += 1

    # ------------------------------------------------------
    # Skip
    # ------------------------------------------------------

    elif key == ord("n"):

        index += 1

    # ------------------------------------------------------
    # Back
    # ------------------------------------------------------

    elif key == ord("b"):

        index = max(0, index - 1)

    # ------------------------------------------------------
    # Quit
    # ------------------------------------------------------

    elif key == ord("q"):

        break

cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("Selection Finished")
print("=" * 60)
print(f"Images Selected : {len(selected)}")
print(f"Saved To        : {DESTINATION_FOLDER}")
print("=" * 60)