import cv2
import os

VIDEO_FOLDER = "video_data"

print("=" * 70)
print(f"{'Video':30} {'FPS':>8} {'Duration(s)':>15} {'Frames':>10}")
print("=" * 70)

for video in os.listdir(VIDEO_FOLDER):

    if not video.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    path = os.path.join(VIDEO_FOLDER, video)

    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps if fps else 0

    print(f"{video:30} {fps:8.2f} {duration:15.2f} {total_frames:10}")

    cap.release()