# RoadEye - Real-Time Object Detection for Road Safety

RoadEye is an AI-based object detection system built with YOLOv8 to identify critical road elements such as police vehicles, traffic cones, road signs, and accident zones – designed to automate Waze-style road alerts using smart camera input.

---

## Key Features

- **Trained YOLOv8 Model** - Custom-trained on real Israeli road data.
- **Classes Detected**:
  - `0` - Police Car  
  - `1` - Traffic Cone  
  - `2` - Directional Sign  
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
