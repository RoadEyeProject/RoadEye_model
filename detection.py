"""
NOTE: In the future, there is a potential that not all classes
will be named this nice, so we will rename them here.
"""
CLASS_DISPLAY_NAMES = {
    "Police Car": "Police Car",
    "Arrow Board": "Arrow Board", 
    "Traffic Cones": "Traffic Cones",
    "Accident": "Accident"
}

def detect_events(model, image, conf_thresh=0.3):
    results = model(image, conf=conf_thresh)
    detections = []

    for box in results[0].boxes:
        conf = float(box.conf)
        cls_idx = int(box.cls)
        cls_name = model.names[cls_idx]
        if conf >= conf_thresh:
            detections.append({
                "class": cls_name,
                "displayName": CLASS_DISPLAY_NAMES.get(cls_name, cls_name),
                "confidence": conf,
                "bbox": {
                    "x1": float(box.xyxy[0][0]),
                    "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]),
                    "y2": float(box.xyxy[0][3])
                }
            })
    return detections
