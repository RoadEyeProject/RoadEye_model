import numpy as np
import cv2
from PIL import Image
import math


def normalize_bbox(bbox, img_w, img_h):
    """
    Accepts bbox in one of the following forms and returns integer (x, y, w, h):
      - [x, y, w, h]
      - [x1, y1, x2, y2]
      - {"x":..., "y":..., "w":..., "h":...}
      - {"x1":..., "y1":..., "x2":..., "y2":...}
      - {"xmin":..., "ymin":..., "xmax":..., "ymax":...}
    Values may be absolute pixels or normalized floats in [0,1].
    """
    # Extract raw numbers
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x", "y", "w", "h")):
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        elif all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            x, y, w, h = x1, y1, (x2 - x1), (y2 - y1)
        elif all(k in bbox for k in ("xmin", "ymin", "xmax", "ymax")):
            x1, y1, x2, y2 = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
            x, y, w, h = x1, y1, (x2 - x1), (y2 - y1)
        else:
            raise ValueError(f"Unsupported bbox dict keys: {list(bbox.keys())}")
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        a, b, c, d = bbox
        # Heuristic: if c>d and (a<b) it might be x1,y1,x2,y2
        if c > 0 and d > 0 and (c > 1.0 or d > 1.0) and (c > a and d > b):
            # looks like x1,y1,x2,y2 in pixels
            x, y, w, h = a, b, (c - a), (d - b)
        else:
            # assume x,y,w,h
            x, y, w, h = a, b, c, d
    else:
        raise TypeError(f"Unsupported bbox type: {type(bbox)}")

    # If values look normalized (<=1), scale to pixels
    def maybe_scale(val, dim):
        return val * dim if 0.0 <= float(val) <= 1.0 else float(val)

    x = maybe_scale(x, img_w)
    y = maybe_scale(y, img_h)
    w = maybe_scale(w, img_w)
    h = maybe_scale(h, img_h)

    # Clamp to image bounds and cast to int
    x = max(0, min(int(round(x)), img_w - 1))
    y = max(0, min(int(round(y)), img_h - 1))
    w = max(1, min(int(math.ceil(w)), img_w - x))
    h = max(1, min(int(math.ceil(h)), img_h - y))
    return x, y, w, h


def to_bgr_ndarray(img):
    """
    Normalize various image types (PIL.Image, np.ndarray RGB/GRAY/BGRA, bytes) to BGR np.ndarray.
    """
    if img is None:
        raise ValueError("to_bgr_ndarray: image is None")

    # PIL.Image -> RGB -> BGR
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        arr = np.array(img)                   # RGB
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # Bytes (e.g., JPEG bytes) -> decode with OpenCV
    if isinstance(img, (bytes, bytearray)):
        buf = np.frombuffer(img, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # returns BGR
        if arr is None:
            raise ValueError("to_bgr_ndarray: failed to decode bytes")
        return arr

    # Already ndarray
    if isinstance(img, np.ndarray):
        if img.ndim == 2:                     # GRAY -> BGR
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3:
            if img.shape[2] == 3:
                # Heuristic: assume already BGR (common for OpenCV); if you *know* it's RGB, convert here.
                return img
            if img.shape[2] == 4:             # BGRA -> BGR
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    raise TypeError(f"to_bgr_ndarray: unsupported image type {type(img)}")

def show_detections(image, detections, window_name="Detections"):
    """
    Draw bounding boxes and labels, then display with OpenCV.
    detections: list of dicts with keys ["bbox"= [x,y,w,h], "displayName", "confidence"]
    """
    vis = to_bgr_ndarray(image).copy()
    ih, iw = vis.shape[:2]

    for det in detections:
        x, y, w, h = normalize_bbox(det["bbox"], iw, ih)
        label = det.get("displayName", det.get("class", "obj"))
        conf  = float(det.get("confidence", 0.0))

        cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(vis, f"{label} {conf:.2f}",
                    (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, vis)
    cv2.waitKey(1)
