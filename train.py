from ultralytics import YOLO

model = YOLO('yolov8m.pt')  # Medium model for faster training

def main():
    model.train(
        data='Dataset/splitData/data.yaml',
        epochs=150,
        batch=16,
        imgsz=416,
        lr0=0.0005,
        augment=True,
        half=True,
        patience=10,
        device=0
    )

if __name__ == '__main__':
    main()
