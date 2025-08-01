import json
from datetime import datetime
from dotenv import load_dotenv
from model_loader import load_model
from image_utils import decode_base64_image
from redis_utils import pop_image, push_event, is_on_cooldown, set_cooldown, publish_event
from detection import detect_events
from db import increment_user_event

load_dotenv()
model = load_model()

EVENT_KEYS = {
    "police_car": "Police Car",
    "accident": "Accident",
    "Arrow Board": "Road Construction",
    "cones": "Road Construction"
}

def process_images():
    print("📡 Listening for incoming images...")

    while True:
        try:
            raw = pop_image()
            if not raw:
                continue

            message = json.loads(raw)
            user_id = message["userId"]
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
                
                updated_name_event_type = EVENT_KEYS.get(event_type)
                event = {
                    "userId": user_id,
                    "eventType": updated_name_event_type,
                    "location": location,
                    "timestamp": timestamp,
                    "confidence": round(det["confidence"], 2),
                    "displayName": det["displayName"],
                    "bbox": det["bbox"]
                }

                push_event(json.dumps(event))
                publish_event(json.dumps({
                    "userId": user_id,
                    "event": updated_name_event_type,
                    "timestamp": timestamp,
                    "cooldown": 3 * 60
                }))
                set_cooldown(user_id, event_type)
                increment_user_event(user_id, det["displayName"])
                print(f"✅ Event pushed to redis: {updated_name_event_type} by {user_id}")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    process_images()
