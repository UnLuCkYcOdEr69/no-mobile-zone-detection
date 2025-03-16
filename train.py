from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Train with corrected dataset path
model.train(
    data="C:/Users/ad807/OneDrive/Desktop/No mobile zone/dataset/data.yaml", 
    epochs=50,
    batch=8,
    imgsz=640,
    device="cpu"
)

