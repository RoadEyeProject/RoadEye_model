import base64
from PIL import Image
from io import BytesIO

def decode_base64_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        print(f"❌ Error decoding image: {e}")
        return None
