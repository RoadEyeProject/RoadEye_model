# realtime_view.py
import cv2
import math
import time
import numpy as np
from PIL import Image
from multiprocessing import Process, Queue
from multiprocessing.queues import Queue as QueueType
from typing import List, Dict, Tuple, Any

# --- helpers copied from your file (trimmed) ---
def to_bgr_ndarray(img):
    if img is None:
        raise ValueError("to_bgr_ndarray: image is None")
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if isinstance(img, (bytes, bytearray)):
        buf = np.frombuffer(img, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("to_bgr_ndarray: failed to decode bytes")
        return arr
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3:
            if img.shape[2] == 3:
                return img
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    raise TypeError(f"to_bgr_ndarray: unsupported image type {type(img)}")

def normalize_bbox(bbox, img_w, img_h):
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
        if c > 0 and d > 0 and (c > 1.0 or d > 1.0) and (c > a and d > b):
            x, y, w, h = a, b, (c - a), (d - b)
        else:
            x, y, w, h = a, b, c, d
    else:
        raise TypeError(f"Unsupported bbox type: {type(bbox)}")

    def scale(v, dim): return v * dim if 0.0 <= float(v) <= 1.0 else float(v)
    x = max(0, min(int(round(scale(x, img_w))), img_w - 1))
    y = max(0, min(int(round(scale(y, img_h))), img_h - 1))
    w = max(1, min(int(math.ceil(scale(w, img_w))), img_w - x))
    h = max(1, min(int(math.ceil(scale(h, img_h))), img_h - y))
    return x, y, w, h

# --- viewer process ---
def _viewer_loop(q: QueueType, window_name: str, target_fps: float, draw_scale: float):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    min_dt = 1.0 / target_fps if target_fps > 0 else 0.0
    last = 0.0

    try:
        while True:
            # drain queue, keep only latest (drop stale frames)
            item = None
            while True:
                try:
                    item = q.get_nowait()
                except Exception:
                    break
            if item is None:
                # nothing new; give UI a tiny breather
                cv2.waitKey(1)
                continue

            if item == "__quit__":
                break

            img, detections = item
            vis = to_bgr_ndarray(img)
            ih, iw = vis.shape[:2]

            # draw boxes (thin lines to keep it light)
            if detections:
                vis = vis.copy()
                for det in detections:
                    x, y, w, h = normalize_bbox(det["bbox"], iw, ih)
                    label = det.get("displayName", det.get("class", "obj"))
                    conf  = float(det.get("confidence", 0.0))
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(vis, f"{label} {conf:.2f}",
                                (x, max(y-10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # optional downscale for faster UI
            if draw_scale != 1.0:
                vis = cv2.resize(
                    vis, (int(iw * draw_scale), int(ih * draw_scale)),
                    interpolation=cv2.INTER_AREA
                )

            # simple FPS limit for rendering
            now = time.time()
            if now - last < min_dt:
                time.sleep(max(0.0, min_dt - (now - last)))
            last = time.time()

            cv2.imshow(window_name, vis)
            cv2.waitKey(1)
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass

class AsyncViewer:
    def __init__(self, window_name: str = "Detections", target_fps: float = 20.0, draw_scale: float = 1.0):
        self.window_name = window_name
        self.q: QueueType = Queue(maxsize=1)  # always keep latest
        self.p: Process | None = Process(target=_viewer_loop, args=(self.q, window_name, target_fps, draw_scale))

    def start(self):
        if self.p is not None and not self.p.is_alive():
            self.p.start()

    def stop(self):
        try:
            self.q.put("__quit__")
        except Exception:
            pass
        if self.p is not None:
            self.p.join(timeout=1.0)

    def show(self, image: Any, detections: List[Dict]):
        # non-blocking put: drop if viewer is still busy
        try:
            if self.q.full():
                _ = self.q.get_nowait()
        except Exception:
            pass
        try:
            self.q.put_nowait((image, detections))
        except Exception:
            pass
