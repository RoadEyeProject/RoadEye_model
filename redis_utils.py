import redis
import os

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", 1))

client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def is_on_cooldown(user_id, event_type):
    key = f"cooldown:{user_id}:{event_type}"
    return client.exists(key)

def set_cooldown(user_id, event_type):
    key = f"cooldown:{user_id}:{event_type}"
    client.setex(key, COOLDOWN_MINUTES * 60, "1")

def push_event(event):
    client.rpush("event_queue", event)

def pop_image():
    data = client.blpop("image_queue", timeout=5)
    return data[1] if data else None

def publish_event(event):
    client.publish("detected_events", event)
