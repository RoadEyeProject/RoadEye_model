from ultralytics import YOLO

def load_model():
    model = YOLO("best_27.8.pt")
    print("✅ YOLO model loaded with classes:", model.names)
    return model
