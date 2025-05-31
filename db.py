from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.get_database()  # Will use DB from URI
users = db.users  # Assuming your collection is called "users"

EVENT_KEYS = {
    "Police Car": "police",
    "Accident": "accident",
    "Arrow Board": "road_construction",
    "Traffic Cones": "road_construction"
}

def increment_user_event(user_id: str, display_name: str):
    """
    Increment the count for the given event type in the user's MongoDB record.
    """
    event_key = EVENT_KEYS.get(display_name)
    if not event_key:
        print(f"⚠️ Unknown event '{display_name}' – skipping DB update")
        return

    result = users.update_one(
        {"_id": user_id},
        {"$inc": {f"reportCounts.{event_key}": 1}},
        upsert=True
    )
    print(f"📊 Updated MongoDB report count for user {user_id}: {event_key}")
