import torch
import redis
import json
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
IMAGE_QUEUE = "image_queue"
EVENT_QUEUE = "event_queue"

# Connect to Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

# Load YOLOv8 model with custom weights
model = YOLO('police_model.pt')

# Verify loaded labels
print("Model classes:", model.names)

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

# Improved function to check if "police car" is detected
def detect_police_car(image_pil):
    results = model(image_pil, conf=0.7)  # confidence threshold at 70%
    
    # Print detection details for debugging
    for r in results:
        boxes = r.boxes
        if len(boxes) == 0:
            print("No detections found")
            return False
            
        # Print confidence scores and classes
        for i, box in enumerate(boxes):
            conf = float(box.conf)
            cls_idx = int(box.cls)
            cls_name = model.names[cls_idx]
            print(f"Detection #{i+1}: {cls_name} (confidence: {conf:.2f})")
            
        # Check if any detection is a police car with high confidence
        police_detections = [(i, float(box.conf)) for i, box in enumerate(boxes) 
                            if model.names[int(box.cls)] == "police car" and float(box.conf) >= 0.7]
        
        if police_detections:
            print(f"Police car detected with confidence: {max([conf for _, conf in police_detections]):.2f}")
            return True
            
    return False

# Main image processing loop
def process_images():
    print("Listening for images in Redis...")
    while True:
        try:
            image_data = redis_client.blpop(IMAGE_QUEUE, timeout=5)
            if not image_data:
                continue
                
            json_data = json.loads(image_data[1].decode("utf-8"))
            base64_image = json_data.get("image")
            location = json_data.get("location", {})
            
            image_pil = decode_base64_image(base64_image)
            if image_pil is None:
                continue
                
            # Call detect_police_car with only the image parameter
            if detect_police_car(image_pil):
                event = {"eventType": "police", "location": location}
                redis_client.rpush(EVENT_QUEUE, json.dumps(event))
                print(f"Event sent to Redis: {event}")
                
        except Exception as e:
            print(f"Error processing image: {e}")

if __name__ == "__main__":
    process_images()