import torch
import redis
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
IMAGE_QUEUE = "image_queue"  # Incoming images
EVENT_QUEUE = "event_queue"  # Outgoing detected objects

# Connect to Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

# Load YOLOv5 model
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
model.to(device)

# Function to decode Base64 image correctly
def decode_base64_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB") 

        return image  # return PIL image directly

    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

# Function to check if "keyboard" is detected
def detect_objects(image_pil):
    results = model(image_pil)
    detected_objects = results.pandas().xyxy[0]["name"].tolist()

    print(f"Detected objects: {detected_objects}")

    if "keyboard" in detected_objects:
        print("Keyboard detected!")
        return True
    return False

# Function to process images from Redis
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

            if detect_objects(image_pil):
                event = {"eventType": "keyboard", "location": location}
                redis_client.rpush(EVENT_QUEUE, json.dumps(event))
                print(f"Event sent to Redis: {event}")

        except Exception as e:
            print(f"Error processing image: {e}")

if __name__ == "__main__":
    process_images()
