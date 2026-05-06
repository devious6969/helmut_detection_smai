from ultralytics import YOLO

# ---------------------------------------------------
# LOAD PRETRAINED YOLOv8 MODEL
# ---------------------------------------------------

model = YOLO("yolov8n.pt")

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

model.train(
    data="dataset/data.yaml",
    epochs=20,
    imgsz=640,
    batch=4,
    device="cpu"
)

print("Training Completed")