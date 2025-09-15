# ROADEYE/model/video_infer_gui.py
import os
import sys
import cv2
import time
import math
import threading
from pathlib import Path
from tkinter import Tk, Button, Label, filedialog, StringVar, DISABLED, NORMAL
from dotenv import load_dotenv
import numpy as np
from PIL import Image

# ---- Resolve project root and load envs (works when run from ROADEYE/) ----
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)  # optional local overrides

# Make Ultralytics write settings to a writable place on macOS/local
os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", str(Path.home() / ".yolo_config"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.environ["ULTRALYTICS_SETTINGS_DIR"])

# ---- Project imports (your existing modules) ----
sys.path.append(str(ROOT / "model"))
from model_loader import load_model
from detection import detect_events


def draw_boxes_bgr(frame_bgr: np.ndarray, detections, color=(0, 255, 0)):
    """
    Draws YOLO detections on a BGR frame in-place.
    detections: list of { class, displayName, confidence, bbox:{x1,y1,x2,y2} }
    """
    h, w = frame_bgr.shape[:2]
    for det in detections:
        bbox = det.get("bbox", {})
        x1 = int(max(0, min(w - 1, bbox.get("x1", 0))))
        y1 = int(max(0, min(h - 1, bbox.get("y1", 0))))
        x2 = int(max(0, min(w - 1, bbox.get("x2", w - 1))))
        y2 = int(max(0, min(h - 1, bbox.get("y2", h - 1))))
        label = det.get("displayName", det.get("class", "obj"))
        conf = float(det.get("confidence", 0.0))

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        txt = f"{label} {conf:.2f}"
        # neat text bg
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ytxt = max(0, y1 - th - 6)
        cv2.rectangle(frame_bgr, (x1, ytxt), (x1 + tw + 6, ytxt + th + 6), (0, 0, 0), -1)
        cv2.putText(frame_bgr, txt, (x1 + 3, ytxt + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR ndarray -> PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def process_video(
    in_path: Path,
    out_path: Path,
    model,
    device: str,
    half: bool,
    status_cb=lambda s: None,
    progress_cb=lambda cur, total: None,
    conf_thresh: float = 0.7,
    imgsz: int = 640,
    max_out_width: int = 1280,  # downscale for speed if needed, keep aspect
):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {in_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional downscale to keep output manageable
    if src_w > max_out_width:
        scale = max_out_width / src_w
        out_w = max_out_width
        out_h = int(round(src_h * scale))
    else:
        out_w, out_h = src_w, src_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely compatible
    out = cv2.VideoWriter(str(out_path), fourcc, src_fps, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    status_cb(f"Processing… ({src_w}x{src_h} @ {src_fps:.1f}fps) → {out_w}x{out_h}")
    t0 = time.time()
    frames_done = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Resize for output if needed (we detect on the resized frame for speed)
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != (out_w, out_h):
                frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # Run detection on PIL image (your detect_events supports PIL/np)
            pil_img = bgr_to_pil(frame_bgr)
            detections = detect_events(
                model,
                pil_img,
                conf_thresh=conf_thresh,
                device=device,
                half=half,
                imgsz=imgsz
            )

            # Draw on the (possibly resized) frame
            draw_boxes_bgr(frame_bgr, detections)

            out.write(frame_bgr)
            frames_done += 1

            if total_frames:
                progress_cb(frames_done, total_frames)
            else:
                progress_cb(frames_done, frames_done)  # unknown total

    finally:
        cap.release()
        out.release()

    dt = time.time() - t0
    fps_eff = frames_done / dt if dt > 0 else 0.0
    status_cb(f"Done: {frames_done} frames in {dt:.1f}s ({fps_eff:.1f} FPS). Saved → {out_path.name}")


# ----------------- Tiny Tkinter GUI -----------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("RoadEye • Video Inference")

        self.model = None
        self.device = "cpu"
        self.half = False

        self.status = StringVar(value="Load model to begin.")
        self.progress = StringVar(value="")

        self.btn_load = Button(root, text="1) Load Model", command=self.load_model, width=24)
        self.btn_pick = Button(root, text="2) Pick Video & Run", command=self.pick_and_run, state=DISABLED, width=24)
        self.lbl_status = Label(root, textvariable=self.status, anchor="w", justify="left")
        self.lbl_progress = Label(root, textvariable=self.progress, anchor="w", justify="left")

        self.btn_load.pack(padx=12, pady=(12, 6))
        self.btn_pick.pack(padx=12, pady=6)
        self.lbl_status.pack(padx=12, pady=(10, 2))
        self.lbl_progress.pack(padx=12, pady=(0, 12))

    def load_model(self):
        self.status.set("Loading model…")
        self.root.update_idletasks()
        try:
            model, device, half = load_model()  # <-- your existing function
            self.model = model
            self.device = device
            self.half = half
            self.status.set(f"✅ Model ready. Device: {device}  •  Half: {half}")
            self.btn_pick.configure(state=NORMAL)
        except Exception as e:
            self.status.set(f"❌ Failed to load model: {e}")

    def pick_and_run(self):
        if self.model is None:
            self.status.set("Load model first.")
            return
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.m4v"), ("All files", "*.*")]
        )
        if not path:
            return

        in_path = Path(path)
        out_path = in_path.with_name(in_path.stem + "_det.mp4")

        self.btn_pick.configure(state=DISABLED)
        self.status.set("Preparing…")
        self.progress.set("")

        def run():
            try:
                process_video(
                    in_path,
                    out_path,
                    self.model,
                    self.device,
                    self.half,
                    status_cb=self.status.set,
                    progress_cb=self.update_progress
                )
                self.progress.set(f"Output saved: {out_path}")
            except Exception as e:
                self.status.set(f"❌ Error: {e}")
            finally:
                self.btn_pick.configure(state=NORMAL)

        threading.Thread(target=run, daemon=True).start()

    def update_progress(self, cur, total):
        if total:
            pct = 100.0 * cur / total
            self.progress.set(f"Progress: {cur}/{total} ({pct:.1f}%)")
        else:
            self.progress.set(f"Frames: {cur}")

def main():
    root = Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
