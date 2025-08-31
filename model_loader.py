# model_loader.py
from pathlib import Path
import os
import torch
from ultralytics import YOLO

# -------- Device selection --------
def pick_device():
    """
    Returns (device_str, use_half)
      - device_str in {"cuda", "mps", "cpu"}
      - use_half = True only on CUDA and when supported
    Respects env MODEL_DEVICE in {"auto","cuda","mps","cpu"} (default: auto)
    """
    want = (os.getenv("MODEL_DEVICE") or "auto").lower()

    def cuda_ok():
        try:
            if not torch.cuda.is_available():
                return False
            # Optional: require reasonable compute capability for FP16 speedups
            major, minor = torch.cuda.get_device_capability(0)
            return (major, minor) >= (7, 0)  # Volta/Turing/Ampere+
        except Exception:
            return False

    def mps_ok():
        try:
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except Exception:
            return False

    if want == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif want == "mps":
        device = "mps" if mps_ok() else "cpu"
    elif want == "cpu":
        device = "cpu"
    else:  # auto
        device = "cuda" if cuda_ok() else ("mps" if mps_ok() else "cpu")

    use_half = (device == "cuda") and cuda_ok()

    # Small perf knobs
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # speed up fixed-size inputs
    # (Keep FP32 on MPS/CPU; half on MPS is generally not supported)

    return device, use_half


def load_model():
    ROOT = Path(__file__).resolve().parents[1]
    weights = ROOT / "model" / "best_27.8.pt"

    device, use_half = pick_device()
    model = YOLO(weights)  # weights load on CPU; .predict will move as needed

    # Print summary once
    print(f"✅ YOLO loaded: {weights.name}")
    print(f"🖥️  Device: {device} | Precision: {'FP16' if use_half else 'FP32'}")
    print("📚 Classes:", model.names)

    return model, device, use_half
