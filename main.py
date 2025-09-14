# model/main.py
import os
import json
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# ---- Resolve project root and load envs (works when run from ROADEYE/) ----
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)  # optional local overrides

# Make Ultralytics write settings to a writable place on macOS/local
os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", str(Path.home() / ".yolo_config"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.environ["ULTRALYTICS_SETTINGS_DIR"])

# ---- Imports that depend on env being loaded ----
from model_loader import load_model
from image_utils import decode_base64_image
from redis_utils import pop_image, push_event, is_on_cooldown, set_cooldown, publish_event
from detection import detect_events
from db import increment_user_event
from show_detections import to_bgr_ndarray, normalize_bbox
from realtime_view import AsyncViewer


EVENT_KEYS = {
    "police_car": "Police Car",
    "accident": "Accident",
    "Arrow Board": "Road Construction",
    "cones": "Road Construction",
}

IDLE_SLEEP_SEC = 0.05  # prevent busy spinning if queue is empty


def _env_summary():
    # Helpful one-time log so you know which envs are actually set
    redis_url = os.getenv("REDIS_URL") or f"redis://{os.getenv('REDIS_HOST','localhost')}:{os.getenv('REDIS_PORT','6379')}/{os.getenv('REDIS_DB','0')}"
    print("🔧 Env loaded from:", ROOT / ".env", "and", ROOT / ".env.local")
    print("🔗 REDIS_URL:", redis_url)
    print("📂 ULTRALYTICS_SETTINGS_DIR:", os.environ["ULTRALYTICS_SETTINGS_DIR"])


def process_images(model):
    print("📡 Listening for incoming images...")
    while True:
        try:
            raw = pop_image()
            if not raw:
                time.sleep(IDLE_SLEEP_SEC)
                continue

            # raw may be bytes from Redis; ensure str
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")

            message = json.loads(raw)

            user_id = message.get("userId")
            image_b64 = message.get("image")
            if not user_id or not image_b64:
                # Malformed message; skip
                # (Optional: log message once every N)
                continue

            location = message.get("location", {})
            timestamp = message.get("timestamp") or datetime.utcnow().isoformat()

            image = decode_base64_image(image_b64)
            if image is None:
                continue
            
            
            img_bgr = to_bgr_ndarray(image)
            ih, iw = img_bgr.shape[:2]

            detections = detect_events(
                model,
                image,
                conf_thresh=0.7,
                device=DEVICE,
                half=USE_HALF,
                imgsz=640
                )
            viewer.show(image, detections)
            for det in detections:
                event_type = det.get("class")
                if not event_type:
                    continue

                # Cooldown by (user_id, event_class)
                if is_on_cooldown(user_id, event_type):
                    print(f"⏱️ Cooldown active for {user_id} - {event_type}, skipping...")
                    continue

                # Map to display name for the websocket/UI
                updated_name_event_type = EVENT_KEYS.get(event_type, event_type)
                x, y, w, h = normalize_bbox(det["bbox"], iw, ih)
                event = {
                    "userId": user_id,
                    "eventType": updated_name_event_type,
                    "location": location,
                    "timestamp": timestamp,
                    "confidence": round(float(det.get("confidence", 0.0)), 2),
                    "displayName": det.get("displayName", event_type),
                    "bbox": [x, y, w, h],
                }

                # Push to your Redis queues/channels
                push_event(json.dumps(event))
                publish_event(
                    json.dumps(
                        {
                            "userId": user_id,
                            "event": updated_name_event_type,
                            "timestamp": timestamp,
                            "cooldown": 3 * 60,  # seconds
                        }
                    )
                )
                set_cooldown(user_id, event_type)
                increment_user_event(user_id, event["displayName"])
                print(f"✅ Event pushed: {updated_name_event_type} by {user_id}")

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
            viewer.stop()
            break
        except Exception as e:
            # Keep the loop alive; log and continue
            print(f"❌ Error: {e}")
            time.sleep(0.25)


if __name__ == "__main__":
    _env_summary()
    model, DEVICE, USE_HALF = load_model()
    viewer = AsyncViewer(window_name="RoadEye Detections", target_fps=20.0, draw_scale=0.9)
    viewer.start()
    process_images(model)
