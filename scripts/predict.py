from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train-7/weights/best.pt")

    results = model.predict(
        source="C:\\Users\\Hamza\\Downloads\\imag.jpg",     
        conf=0.25,
        save=True
    )

    print("Prediction completed!")
    print("Number of predictions:", len(results))

if __name__ == "__main__":
    main()