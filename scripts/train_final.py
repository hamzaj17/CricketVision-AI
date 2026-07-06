from ultralytics import YOLO


def main():

    DATASET = "datasets/cricket_dataset_final/data.yaml"

    model = YOLO("yolov8n.pt")

    model.train(
        data=DATASET,
        epochs=100,
        imgsz=640,
        batch=16,
        workers=0,
        device=0,
        project="runs/detect",
        name="CricketVision_Final",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        patience=20,
        cos_lr=True,
        amp=True,
        cache=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.10,
        scale=0.50,
        shear=2,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.10,
        copy_paste=0.0,
        close_mosaic=10,
        plots=True,
        verbose=True
    )


if __name__ == "__main__":
    main()