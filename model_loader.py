from ultralytics import YOLO

def load_model():
    model = YOLO("best.pt")
    print("✅ YOLO model loaded with classes:", model.names)
    return model
