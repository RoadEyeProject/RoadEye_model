from pymongo import MongoClient
from bson import ObjectId
import os

# Connect to MongoDB
MONGO_URI_MODEL = os.getenv("MONGO_URI_MODEL")
mongo_client = MongoClient(MONGO_URI_MODEL)
db = mongo_client.get_database()  # Extracts DB from URI
users = db.users  # Assuming your collection is named "users"

# Mapping YOLO model classes to canonical MongoDB field names
EVENT_KEYS = {
    "police_car": "Police Car",
    "accident": "Accident",
    "Arrow Board": "Road Construction",
    "cones": "Road Construction"
}

def increment_user_event(user_id: str, display_name: str):
    """
    Increment the count for the given event type in the user's MongoDB record.
    """
    event_key = EVENT_KEYS.get(display_name)
    if not event_key:
        print(f"⚠️ Unknown event '{display_name}' – skipping DB update")
        return

    try:
        object_id = ObjectId(user_id)  # Convert string to ObjectId
    except Exception as e:
        print(f"❌ Invalid user_id '{user_id}': {e}")
        return

    result = users.update_one(
        {"_id": object_id},
        {"$inc": {f"reportCounts.{event_key}": 1}}
    )

    if result.matched_count == 0:
        print(f"❌ No user found with ID {user_id}")
    elif result.modified_count == 0:
        print(f"⚠️ User found, but no update made for event '{event_key}'")
    else:
        print(f"📊 Updated MongoDB report count for user {user_id}: {event_key}")
