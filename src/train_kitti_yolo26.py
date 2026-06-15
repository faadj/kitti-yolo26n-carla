# KITTI Object Detection — YOLO26n Training Pipeline
# Run on Google Colab (free T4 GPU recommended)
# Model: YOLO26n (Ultralytics, January 2026)
# Dataset: KITTI 2D Object Detection — 7,481 images, 8 classes

# ── CELL 1: Setup ─────────────────────────────────────────────────────────────
# Install the Ultralytics library which contains YOLO26 and all training tools
!uv pip install ultralytics

import ultralytics
# Verify installation and confirm GPU is available before proceeding
ultralytics.checks()

# ── CELL 2: Train ─────────────────────────────────────────────────────────────
from ultralytics import YOLO

# Load YOLO26n with pretrained COCO weights as the starting point
model = YOLO("yolo26n.pt")

# Fine-tune on KITTI: auto-downloads dataset (390 MB), trains for 10 epochs at 640x640
# KITTI classes: car, van, truck, pedestrian, person_sitting, cyclist, tram, misc
# Training set: 5,985 images | Validation set: 1,496 images
# Loss components: box_loss (localisation), cls_loss (classification), dfl_loss (distribution)
results = model.train(data="kitti.yaml", epochs=10, imgsz=640)

# ── CELL 3: Evaluate ──────────────────────────────────────────────────────────
from ultralytics import YOLO

# Load best checkpoint for evaluation
model_best = YOLO(f"{model.trainer.save_dir}/weights/best.pt")

# Run validation on 1,496 KITTI validation images
# mAP50: detection accuracy at IoU 0.50
# mAP50-95: stricter average across IoU 0.50 to 0.95
metrics = model_best.val(data="kitti.yaml")

# ── CELL 4: Predict and Visualise ─────────────────────────────────────────────
from ultralytics import YOLO
import glob, os

# Load the best trained model for inference
model_best = YOLO(f"{model.trainer.save_dir}/weights/best.pt")

# Run detection on a sample KITTI image and save with bounding boxes drawn
prediction_results = model_best.predict(
    "https://ultralytics.com/assets/kitti-inference-im0.png",
    save=True
)

# Display the saved result image inline in the notebook
from IPython.display import Image as IPImage
saved_dir = prediction_results[0].save_dir
saved_images = glob.glob(os.path.join(saved_dir, "*.png")) + glob.glob(os.path.join(saved_dir, "*.jpg"))
IPImage(saved_images[0])

# ── CELL 5: Test on 5 random validation images ────────────────────────────────
import os, glob, random
from ultralytics import YOLO
from IPython.display import Image as IPImage, display

model_best = YOLO(f"{model.trainer.save_dir}/weights/best.pt")
val_images = glob.glob("/content/datasets/kitti/images/val/*.png")
random.seed(42)
sample_images = random.sample(val_images, 5)

prediction_results = model_best.predict(
    source=sample_images,
    save=True,
    project="/content/kitti_sample_predictions",
    name="five_random",
    exist_ok=True
)

for i, result in enumerate(prediction_results):
    saved_path = os.path.join(
        "/content/kitti_sample_predictions/five_random",
        os.path.basename(result.path)
    )
    print(f"\nImage {i+1}: {os.path.basename(result.path)}")
    print(f"Detections: {result.verbose()}")
    display(IPImage(saved_path))

# ── CELL 6: Export to ONNX ────────────────────────────────────────────────────
from ultralytics import YOLO

model_best = YOLO(f"{model.trainer.save_dir}/weights/best.pt")

# Export to ONNX with opset 17 for Python 3.7 compatibility (CARLA 0.9.11 environment)
# ONNX (Open Neural Network Exchange): universal format readable by any inference engine
model_best.export(format="onnx", opset=17)

# ── CELL 7: Download model weights ────────────────────────────────────────────
from google.colab import files

# Download best.pt — for Python 3.10+ environments
files.download(f"{model.trainer.save_dir}/weights/best.pt")

# Download best.onnx — for Python 3.7 environments (CARLA 0.9.11 deployment)
files.download(f"{model.trainer.save_dir}/weights/best.onnx")
