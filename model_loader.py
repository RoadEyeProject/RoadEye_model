from ultralytics import YOLO
from pathlib import Path

def load_model():
    ROOT = Path(__file__).resolve().parents[1]
    model = YOLO(ROOT / "model" / "best_27.8.pt")
    print("✅ YOLO model loaded with classes:", model.names)
    return model
