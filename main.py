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
from datetime import datetime
import time

load_dotenv()

SAVE_DETECTION_IMAGES = os.getenv("SAVE_DETECTION_IMAGES", "false").lower() == "true"

warnings.simplefilter("ignore", category=FutureWarning)

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

IMAGE_QUEUE = "image_queue"
EVENT_QUEUE = "event_queue"

# Directory to save detected images
DETECTIONS_DIR = "/app/detected_images"
# Create the directory if it doesn't exist
os.makedirs(DETECTIONS_DIR, exist_ok=True)

#DEVELOPMENT FLAG - Set to False for production cooldown behavior
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "true").lower() == "true"

#COOLDOWN SETTINGS
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "1"))  # Default 1 minute
COOLDOWN_SECONDS = COOLDOWN_MINUTES * 60

# Dictionary to track last report time for each event type
last_report_times = {}

print(f"Development Mode: {'ON' if DEVELOPMENT_MODE else 'OFF'}")
print(f"Cooldown Period: {COOLDOWN_MINUTES} minute(s) between same object reports")

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
print(f"Detected images will be saved to: {DETECTIONS_DIR}")

def is_event_in_cooldown(event_type):
    """
    Check if an event type is still in cooldown period
    Returns True if in cooldown, False if can report
    
    NOTE: Both 'cones' and 'Arrow Board' map to 'roadworks', 
    so they share the same cooldown timer
    """
    if DEVELOPMENT_MODE:
        return False  # No cooldown in development mode
    
    if event_type not in last_report_times:
        return False  # Never reported before
    
    time_since_last = time.time() - last_report_times[event_type]
    is_cooling_down = time_since_last < COOLDOWN_SECONDS
    
    if is_cooling_down:
        remaining_time = COOLDOWN_SECONDS - time_since_last
        print(f"{event_type} in cooldown - {remaining_time:.0f}s remaining")
    
    return is_cooling_down

def update_last_report_time(event_type):
    """
    Update the last report time for an event type
    
    NOTE: This affects both cones and Arrow Board since they both 
    map to 'roadworks' event type
    """
    last_report_times[event_type] = time.time()
    if not DEVELOPMENT_MODE:
        print(f"Cooldown started for {event_type} - next report in {COOLDOWN_MINUTES} minute(s)")

def get_cooldown_status():
    """
    Get current cooldown status for all event types
    """
    current_time = time.time()
    status = {}
    
    # Check cooldown status for each unique event type
    for event_type in ["police", "roadworks", "accident"]:
        if event_type in last_report_times:
            time_since_last = current_time - last_report_times[event_type]
            if time_since_last < COOLDOWN_SECONDS:
                remaining = COOLDOWN_SECONDS - time_since_last
                status[event_type] = f"Cooldown: {remaining:.0f}s remaining"
            else:
                status[event_type] = "Ready to report"
        else:
            status[event_type] = "Never reported"
    
    return status

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

def save_detection_image(image_pil, detected_events, location=None):
    """
    Save only the annotated image with detections (no original or JSON)
    """
    try:
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits of microseconds
        
        # Create filename based on detected classes (use display names)
        detected_classes = [CLASS_DISPLAY_NAMES.get(event["class"], event["class"]) for event in detected_events]
        classes_str = "_".join(detected_classes).replace(" ", "")
        
        # Run detection to get annotated image
        results = model(image_pil, conf=0.3)  # Use same threshold as detect_events
        annotated_frame = results[0].plot()
        
        # Convert from RGB to BGR for OpenCV saving
        annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        # Save only the annotated image
        annotated_filename = f"{timestamp}_{classes_str}_detected.jpg"
        annotated_path = os.path.join(DETECTIONS_DIR, annotated_filename)
        cv2.imwrite(annotated_path, annotated_bgr)
        
        print(f"Saved detection: {annotated_filename}")
        
        return {"annotated_path": annotated_path}
        
    except Exception as e:
        print(f"Error saving detection image: {e}")
        return None

def cleanup_old_images(max_images=100):
    """
    Keep only the latest max_images detections to prevent disk space issues
    Only runs if image saving is enabled
    """
    if not SAVE_DETECTION_IMAGES:
        return
        
    try:
        # Get all image files sorted by modification time
        image_files = []
        for filename in os.listdir(DETECTIONS_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(DETECTIONS_DIR, filename)
                image_files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (oldest first)
        image_files.sort(key=lambda x: x[1])
        
        # Remove oldest files if we have too many
        if len(image_files) > max_images:
            files_to_remove = image_files[:-max_images]
            for filepath, _ in files_to_remove:
                try:
                    os.remove(filepath)
                    # Also remove corresponding info file
                    info_file = filepath.replace('.jpg', '_info.json').replace('.jpeg', '_info.json')
                    if os.path.exists(info_file):
                        os.remove(info_file)
                except Exception as e:
                    print(f"Warning: Could not remove old file {filepath}: {e}")
            
            print(f"Cleaned up {len(files_to_remove)} old detection images")
            
    except Exception as e:
        print(f"Warning: Could not cleanup old images: {e}")

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
            
            # Create unique ID for this detection
            detection_id = f"{cls_name}_{int(datetime.now().timestamp()*1000)}_{i}"
            
            detected_events.append({
                "id": detection_id,
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

# UPDATED: Legacy function for backward compatibility
def detect_police_car(image_pil):
    """
    Legacy function - now detects any of the 4 event types
    Returns True if police car, Arrow Board, cones, or accident is detected
    """
    events = detect_events(image_pil)
    return len(events) > 0

# UPDATED: Main image processing loop
def process_images():
    print("Listening for images in Redis...")
    print(f"Monitoring for: {list(CLASS_DISPLAY_NAMES.values())}")
    print(f"Event types: {list(set(EVENT_TYPE_MAPPING.values()))}")
    
    # Cleanup old images on startup
    cleanup_old_images()
    
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
            
            for event_type, events_of_type in events_by_type.items():
                if is_event_in_cooldown(event_type):
                    events_blocked.extend(events_of_type)
                    class_names = [e["display_name"] for e in events_of_type]
                    print(f"{event_type} detection(s) blocked by cooldown: {', '.join(class_names)}")
                else:
                    events_to_report.extend(events_of_type)
                    # Update cooldown timer for this event type (affects both cones and Arrow Board if roadworks)
                    update_last_report_time(event_type)
                    class_names = [e["display_name"] for e in events_of_type]
                    print(f"{event_type} detection(s) will be reported: {', '.join(class_names)}")
            
            # Only process and save if we have events to report
            if events_to_report:
                # Save detection images (only if enabled)
                saved_files = None
                if SAVE_DETECTION_IMAGES:
                    saved_files = save_detection_image(image_pil, detected_events, location)
                
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
                    
                    # Add image paths if saving was successful
                    if saved_files and SAVE_DETECTION_IMAGES:
                        event["saved_images"] = {
                            "annotated": os.path.basename(saved_files["annotated_path"])
                        }
                    
                    redis_client.rpush(EVENT_QUEUE, json.dumps(event))
                    print(f"Event sent to Redis: {event['eventType']} ({event['displayName']}, confidence: {event['confidence']:.2f})")
                
                # Show summary
                print(f"Summary: {len(events_to_report)} reported, {len(events_blocked)} blocked by cooldown")
                
                # Show current cooldown status every 10 detections (only in development)
                if DEVELOPMENT_MODE and len(os.listdir(DETECTIONS_DIR) if SAVE_DETECTION_IMAGES else []) % 10 == 0:
                    status = get_cooldown_status()
                    print("Current cooldown status:")
                    for event_type, status_msg in status.items():
                        print(f"   {event_type}: {status_msg}")
                
                # Periodic cleanup (only if saving images)
                if SAVE_DETECTION_IMAGES and len(os.listdir(DETECTIONS_DIR)) > 200:
                    cleanup_old_images()
            
            elif detected_events:
                blocked_summary = {}
                for event in events_blocked:
                    event_type = event["event_type"]
                    if event_type not in blocked_summary:
                        blocked_summary[event_type] = []
                    blocked_summary[event_type].append(event["display_name"])
                
                summary_parts = []
                for event_type, class_names in blocked_summary.items():
                    summary_parts.append(f"{event_type}: {', '.join(class_names)}")
                
                print(f"All {len(detected_events)} detections blocked by cooldown ({'; '.join(summary_parts)}) - no reports sent")

        except Exception as e:
            print(f"Error processing image: {e}")

if __name__ == "__main__":
    try:
        process_images()
    except KeyboardInterrupt:
        print("Stopped by user.")
        print("Final cooldown status:")
        status = get_cooldown_status()
        for event_type, status_msg in status.items():
            print(f"   {event_type}: {status_msg}")
    finally:
        cv2.destroyAllWindows()