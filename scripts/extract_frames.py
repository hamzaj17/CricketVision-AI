import cv2
import os
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
VIDEO_FOLDER = "video_data"
OUTPUT_FOLDER = "datasets/extracted_frames"
FRAME_INTERVAL = 10  # Save every 10th frame

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

total_frames_saved = 0
total_videos = 0

# Loop through all videos
for video_file in os.listdir(VIDEO_FOLDER):

    if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    total_videos += 1

    video_path = os.path.join(VIDEO_FOLDER, video_file)

    # Folder name = video name
    video_name = Path(video_file).stem

    output_dir = os.path.join(OUTPUT_FOLDER, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    print(f"\nProcessing: {video_file}")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:

            filename = f"frame_{frame_count:06d}.jpg"

            save_path = os.path.join(output_dir, filename)

            cv2.imwrite(save_path, frame)

            saved_count += 1
            total_frames_saved += 1

        frame_count += 1

    cap.release()

    print(f"Total Frames: {frame_count}")
    print(f"Saved Frames: {saved_count}")

print("\n==============================")
print("Frame Extraction Complete!")
print("==============================")
print(f"Videos Processed : {total_videos}")
print(f"Frames Extracted : {total_frames_saved}")