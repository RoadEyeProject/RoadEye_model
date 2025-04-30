import torch
import redis
import json
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
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

# Improved function to check if "police car" is detected, and display image live
def detect_police_car(image_pil):
    results = model(image_pil, conf=0.6)  # confidence threshold at 70%

    # Get annotated image as numpy array
    annotated_frame = results[0].plot()
    
    # Convert annotated image from RGB to BGR (for OpenCV)
    #annotated_image_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Display the image in a window called "Live Detection"
    #cv2.imshow("Live Detection", annotated_frame)
    #cv2.waitKey(1)  # Display each frame for a brief moment

    # Debugging outputs
    boxes = results[0].boxes
    if len(boxes) == 0:
        print("No detections found")
        return False

    detected = False
    for i, box in enumerate(boxes):
        conf = float(box.conf)
        cls_idx = int(box.cls)
        cls_name = model.names[cls_idx]
        print(f"Detection #{i+1}: {cls_name} (confidence: {conf:.2f})")

        if cls_name == "police car" and conf >= 0.6:
            print(f"Police car detected with high confidence: {conf:.2f}")
            detected = True

    return detected

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

            if detect_police_car(image_pil):
                event = {"eventType": "police", "location": location}
                redis_client.rpush(EVENT_QUEUE, json.dumps(event))
                print(f"Event sent to Redis: {event}")

        except Exception as e:
            print(f"Error processing image: {e}")

if __name__ == "__main__":
    try:
        process_images()
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cv2.destroyAllWindows()
