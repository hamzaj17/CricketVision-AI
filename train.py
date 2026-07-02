from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="cricket.v1i.yolov8/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=0
    )

if __name__ == "__main__":
    main()