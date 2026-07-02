from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="C:\\Projects\\CricketVision_AI\\datasets\\cricket_dataset_v1\\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0
    )

if __name__ == "__main__":
    main()