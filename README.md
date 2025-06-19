# RoadEye - Real-Time Object Detection for Road Safety

RoadEye is an AI-based object detection system built with YOLOv8 to identify critical road elements such as police vehicles, traffic cones, road signs, and accident zones – designed to automate Waze-style road alerts using smart camera input.

---

## ⚙️ How It Works

1. **Data Collection**  
   The model is trained on a curated dataset of Israeli road scenes. Images include both **positive examples** (with police cars, cones, signs) and **negative examples** (empty roads, misleading lights like ambulance or reflections).

2. **Labeling & Preprocessing**  
   All positive images were labeled in YOLO format. The dataset was split into `train/valid/test`, and negative images were added to reduce false detections (negative mining).

3. **Model Training with YOLOv8**  
   YOLOv8m was trained using Ultralytics with:
   - Image size of 640×640
   - ~100 epochs
   - Augmentations: Mosaic, Flip, HSV, Lighting Noise, and more
   - Early stopping and class balancing techniques

4. **Evaluation & Export**  
   Best weights (`best.pt`) were exported to both `.pt` and `.onnx` formats. The model was evaluated using mAP metrics, and inference was validated on unseen road images.

5. **Real-Time Inference Pipeline**  
   The trained model is optimized for **real-time predictions** via webcam or video feed. Output includes bounding boxes, confidence scores, and class IDs – ideal for integration into an alert system (e.g., mobile app or edge device).

---

## Key Features

- **Trained YOLOv8 Model** - Custom-trained on real Israeli road data.
- **Classes Detected**:
  - `0` - Police Car  
  - `1` - Traffic Cone (Road Works) 
  - `2` - Directional Sign (Road Works) 
  - `3` - Accident  
- **Negative Mining** - False Positive reduction (e.g. blue lights).
- **Augmentation** - Advanced pipeline: Mosaic, Flip, HSV, Lighting, etc.
- **Export Support** - Includes `best.pt`, `onnx`, and image samples.
- **Real-Time Inference** - Designed for integration with live stream.

---

## Project Structure

```bash
RoadEye_Final_Model/
├── yolov8n.pt                # Pre-trained model (optional)
├── runs/train/               # Training results & logs
├── runs/detect/              # Inference examples
├── data.yaml                 # Class names + paths
├── Models/
│   ├── best.pt               # Trained model weights
│   ├── model.onnx            # Exported ONNX version
├── Data/
│   ├── RoadEye.zip           # Dataset (positive & negative)
│   └── ...                   # Unzipped training images & labels
