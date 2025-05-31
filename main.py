import json
from datetime import datetime
from dotenv import load_dotenv
from model_loader import load_model
from image_utils import decode_base64_image
from redis_utils import pop_image, push_event, is_on_cooldown, set_cooldown
from detection import detect_events

load_dotenv()
model = load_model()

def process_images():
    print("📡 Listening for incoming images...")

    while True:
        try:
            raw = pop_image()
            if not raw:
                continue

            message = json.loads(raw)
            user_id = message["id"]
            image_b64 = message["image"]
            location = message.get("location", {})
            timestamp = message.get("timestamp", datetime.utcnow().isoformat())

            image = decode_base64_image(image_b64)
            if image is None:
                continue

            detections = detect_events(model, image)
            for det in detections:
                event_type = det["class"]

                if is_on_cooldown(user_id, event_type):
                    print(f"⏱️ Cooldown active for {user_id} - {event_type}, skipping...")
                    continue

                event = {
                    "userId": user_id,
                    "eventType": event_type,
                    "location": location,
                    "timestamp": timestamp,
                    "confidence": round(det["confidence"], 2),
                    "displayName": det["displayName"],
                    "bbox": det["bbox"]
                }

                push_event(json.dumps(event))
                set_cooldown(user_id, event_type)
                print(f"✅ Event pushed: {event_type} by {user_id}")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    process_images()
