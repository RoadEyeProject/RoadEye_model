import redis
import json
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import cv2
import warnings
from dotenv import load_dotenv
import os

load_dotenv()

SAVE_DETECTION_IMAGES = os.getenv("SAVE_DETECTION_IMAGES", "false").lower() == "true"

warnings.simplefilter("ignore", category=FutureWarning)

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

IMAGE_QUEUE = "image_queue"
EVENT_QUEUE = "event_queue"

#DEVELOPMENT FLAG - Set to False for production cooldown behavior
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "true").lower() == "true"

# Dictionary to track last report time for each event type
last_report_times = {}

print(f"Development Mode: {'ON' if DEVELOPMENT_MODE else 'OFF'}")

# UPDATED: Event type mapping for your 4-class model
# Based on your current model: {0: 'police_car', 1: 'Arrow Board', 2: 'cones', 3: 'accident'}
EVENT_TYPE_MAPPING = {
    "police_car": "police",        # Model class 0 -> police
    "Arrow Board": "roadworks",    # Model class 1 -> roadworks  
    "cones": "roadworks",          # Model class 2 -> roadworks (same as Arrow Board)
    "accident": "accident"         # Model class 3 -> accident
}

# UPDATED: Class display names for better logging
CLASS_DISPLAY_NAMES = {
    "police_car": "Police Car",
    "Arrow Board": "Arrow Board", 
    "cones": "Traffic Cones",
    "accident": "Accident"
}

# Connect to Redis
redis_client = redis.Redis(
  host=REDIS_HOST,
  port=REDIS_PORT,
  decode_responses=False,
)

# Load YOLOv8 model with new 4-class weights
model = YOLO('best.pt')

# Verify loaded labels
print("Model classes:", model.names)
print("Class mapping:")
for model_class, event_type in EVENT_TYPE_MAPPING.items():
    display_name = CLASS_DISPLAY_NAMES.get(model_class, model_class)
    print(f"  - {display_name} → {event_type}")


# Function to decode Base64 image
def decode_base64_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

# UPDATED: Function to detect all 4 classes
def detect_events(image_pil, confidence_threshold=0.3):
    """
    Detect police cars, Arrow Boards, cones, and accidents in the image
    Returns list of detected events
    
    NOTE: Both 'cones' and 'Arrow Board' are mapped to 'roadworks' event type
    """
    results = model(image_pil, conf=confidence_threshold)

    # Debugging outputs
    boxes = results[0].boxes
    if len(boxes) == 0:
        print("No detections found")
        return []

    detected_events = []
    
    for i, box in enumerate(boxes):
        conf = float(box.conf)
        cls_idx = int(box.cls)
        cls_name = model.names[cls_idx]
        display_name = CLASS_DISPLAY_NAMES.get(cls_name, cls_name)

        if conf >= confidence_threshold:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Map class name to event type
            event_type = EVENT_TYPE_MAPPING.get(cls_name, cls_name.lower())
            
            detected_events.append({
                "class": cls_name,
                "display_name": display_name,
                "event_type": event_type,
                "confidence": conf,
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                }
            })
            
            print(f"{display_name} → {event_type} (confidence: {conf:.2f})")

    return detected_events

# UPDATED: Main image processing loop
def process_images():
    print("Listening for images in Redis...")
    print(f"Monitoring for: {list(CLASS_DISPLAY_NAMES.values())}")
    print(f"Event types: {list(set(EVENT_TYPE_MAPPING.values()))}")
    
    
    while True:
        try:
            image_data = redis_client.blpop(IMAGE_QUEUE, timeout=5)
            if not image_data:
                continue

            json_data = json.loads(image_data[1].decode("utf-8"))
            base64_image = json_data.get("image")
            location = json_data.get("location", {})
            timestamp = json_data.get("timestamp", "")

            image_pil = decode_base64_image(base64_image)
            if image_pil is None:
                continue

            # Detect all event types
            detected_events = detect_events(image_pil)
            
            # Group events by event type to handle cooldown logic
            events_by_type = {}
            for event_info in detected_events:
                event_type = event_info["event_type"]
                if event_type not in events_by_type:
                    events_by_type[event_type] = []
                events_by_type[event_type].append(event_info)
            
            # Filter events based on cooldown status (per event type)
            events_to_report = []
            events_blocked = []
            
            for event_type in events_by_type.items():
            
                # Only process and save if we have events to report
                if events_to_report:
                    # Send only non-blocked events to Redis
                    for event_info in events_to_report:
                        event = {
                            "eventType": event_info["event_type"], 
                            "location": location,
                            "timestamp": timestamp,
                            "confidence": round(event_info["confidence"], 2),
                            "detectedClass": event_info["class"],
                            "displayName": event_info["display_name"],
                            "bbox": event_info["bbox"]
                        }
                        redis_client.rpush(EVENT_QUEUE, json.dumps(event))
                        print(f"Event sent to Redis: {event['eventType']} ({event['displayName']}, confidence: {event['confidence']:.2f})")

        except Exception as e:
            print(f"Error processing image: {e}")


if __name__ == "__main__":
    try:
        process_images()
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cv2.destroyAllWindows()