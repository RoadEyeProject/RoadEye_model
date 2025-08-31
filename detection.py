# detection.py
CLASS_DISPLAY_NAMES = {
    "Police Car": "Police Car",
    "Arrow Board": "Arrow Board",
    "Traffic Cones": "Traffic Cones",
    "Accident": "Accident",
}

def detect_events(model, image, conf_thresh=0.7, device="cpu", half=False, imgsz=640):
    """
    model: ultralytics.YOLO
    image: PIL.Image or ndarray
    device: "cuda" | "mps" | "cpu"
    half: bool (True for CUDA FP16, False otherwise)
    """
    # Call Ultralytics predict with explicit device/precision
    # NOTE: half=True is effective only on CUDA
    results = model(
        image,
        conf=conf_thresh,
        device=device,
        half=half,
        imgsz=imgsz
    )

    detections = []
    r0 = results[0]
    for box in r0.boxes:
        conf = float(box.conf)
        cls_idx = int(box.cls)
        cls_name = model.names[cls_idx]

        if conf >= conf_thresh:
            # xyxy is a (1,4) tensor → take [0] and cast to float
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "class": cls_name,
                "displayName": CLASS_DISPLAY_NAMES.get(cls_name, cls_name),
                "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })
    return detections
