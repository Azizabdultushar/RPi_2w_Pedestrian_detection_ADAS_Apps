# RPi_2w_Pedestrian_detection_ADAS_Apps
# Pedestrian Detection on Raspberry Pi Zero 2 W
Using YOLOv8n + Docker + WiderPerson Dataset + TFLite

---

# Project Goal

Train a lightweight pedestrian detection model on Ubuntu using Docker,
then deploy it to Raspberry Pi Zero 2 W with a CSI camera.

Pipeline:

Dataset
    ↓
YOLOv8 Training
    ↓
TFLite INT8 Export
    ↓
Raspberry Pi Deployment
    ↓
Realtime Pedestrian Detection

---

# PHASE 1 — Install Docker on Ubuntu

## Update Ubuntu

```bash
sudo apt update
sudo apt upgrade -y
```

## Install Docker

```bash
sudo apt install docker.io -y
```

## Enable Docker

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

## Verify Docker

```bash
docker --version
```

---

# PHASE 2 — Create Project

## Create Project Folder

```bash
mkdir -p ~/pedestrian_ai
cd ~/pedestrian_ai
```

## Create Required Directories

```bash
mkdir dataset
mkdir models
mkdir outputs
```

---

# PHASE 3 — Create Docker Environment

## Create Dockerfile

```bash
nano Dockerfile
```

Paste:

```dockerfile
FROM python:3.10-slim

WORKDIR /app/pedestrian_ai

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    wget \
    unzip

RUN pip install --no-cache-dir \
    ultralytics \
    tensorflow \
    onnx \
    onnxruntime \
    tflite-runtime \
    pillow

CMD ["/bin/bash"]
```

Save:
CTRL+O → ENTER → CTRL+X

---

# Build Docker Image

```bash
docker build -t pedestrian-yolo .
```

---

# Run Docker Container

```bash
docker run -it \
  --name pedestrian-container \
  -v $(pwd):/app/pedestrian_ai \
  pedestrian-yolo
```

---

# PHASE 4 — Download WiderPerson Dataset

## Download Dataset Using Browser

Official dataset:
http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/

Download:
- Images
- Annotations

---

# Move Dataset Into Project

Example:

```bash
mv ~/Downloads/WiderPerson* ~/pedestrian_ai/dataset/
```

---

# Extract Dataset

```bash
cd ~/pedestrian_ai/dataset

unzip WiderPerson.zip
```

---

# Verify Dataset Structure

```bash
ls
```

Expected:

```text
Annotations
Images
train.txt
val.txt
```

---

# PHASE 5 — Create YOLO Folder Structure

Inside Docker:

```bash
mkdir -p dataset/images/train
mkdir -p dataset/images/val

mkdir -p dataset/labels/train
mkdir -p dataset/labels/val
```

---

# PHASE 6 — Convert WiderPerson to YOLO Format

## Create Conversion Script

```bash
nano convert_widerperson.py
```

Paste:

```python
import os
import shutil
from PIL import Image

BASE = "/app/pedestrian_ai/dataset"

ANNOTATIONS = os.path.join(BASE, "Annotations")
IMAGES = os.path.join(BASE, "Images")

TRAIN_TXT = os.path.join(BASE, "train.txt")
VAL_TXT = os.path.join(BASE, "val.txt")

YOLO_IMAGES_TRAIN = os.path.join(BASE, "images/train")
YOLO_IMAGES_VAL = os.path.join(BASE, "images/val")

YOLO_LABELS_TRAIN = os.path.join(BASE, "labels/train")
YOLO_LABELS_VAL = os.path.join(BASE, "labels/val")


def convert_split(txt_file, out_img_dir, out_label_dir):

    with open(txt_file, "r") as f:
        image_ids = [x.strip() for x in f.readlines()]

    for image_id in image_ids:

        img_src = os.path.join(IMAGES, image_id + ".jpg")
        ann_src = os.path.join(ANNOTATIONS, image_id + ".jpg.txt")

        if not os.path.exists(img_src):
            continue

        img = Image.open(img_src)
        img_w, img_h = img.size

        shutil.copy(img_src, out_img_dir)

        label_path = os.path.join(out_label_dir, image_id + ".txt")

        with open(ann_src, "r") as ann_f:
            lines = ann_f.readlines()

        with open(label_path, "w") as out_f:

            for line in lines[1:]:

                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                cls = int(parts[0])

                if cls not in [1, 2]:
                    continue

                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])

                w = x2 - x1
                h = y2 - y1

                x_center = x1 + w / 2
                y_center = y1 + h / 2

                x_center /= img_w
                y_center /= img_h
                w /= img_w
                h /= img_h

                out_f.write(
                    f"0 {x_center} {y_center} {w} {h}\n"
                )


convert_split(
    TRAIN_TXT,
    YOLO_IMAGES_TRAIN,
    YOLO_LABELS_TRAIN
)

convert_split(
    VAL_TXT,
    YOLO_IMAGES_VAL,
    YOLO_LABELS_VAL
)

print("DONE")
```

Save:
CTRL+O → ENTER → CTRL+X

---

# Run Conversion

```bash
python convert_widerperson.py
```

Expected output:

```text
DONE
```

---

# Verify Converted Dataset

```bash
ls dataset/images/train | head
```

```bash
ls dataset/labels/train | head
```

---

# PHASE 7 — Create data.yaml

## Create YAML File

```bash
nano data.yaml
```

Paste:

```yaml
path: ./dataset

train: images/train
val: images/val

names:
  0: person
```

Save:
CTRL+O → ENTER → CTRL+X

---

# PHASE 8 — Trial Training

Run a quick 1-epoch training test.

```bash
yolo detect train \
  data=data.yaml \
  model=yolov8n.pt \
  imgsz=320 \
  epochs=1
```

Purpose:
- validate dataset
- validate labels
- generate preview images

---

# Verify Preview Images

```bash
ls runs/detect/train
```

Check:
- train_batch*.jpg

Ensure:
- boxes align correctly
- pedestrians detected properly

---

# PHASE 9 — Final Training

## Start Full Training

```bash
yolo detect train \
  data=data.yaml \
  model=yolov8n.pt \
  imgsz=320 \
  epochs=80 \
  batch=16
```

---

# Recommended Settings

| Parameter | Value |
|---|---|
| Model | yolov8n |
| Resolution | 320 |
| Classes | 1 |
| Epochs | 80 |
| Batch Size | 16 |

---

# Training Output

Best model saved at:

```text
runs/detect/train/weights/best.pt
```

---

# PHASE 10 — Test Model

## Run Prediction

```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=dataset/images/val
```

Results saved at:

```text
runs/detect/predict
```

---

# PHASE 11 — Export for Raspberry Pi

## Export INT8 TFLite

```bash
yolo export \
  model=runs/detect/train/weights/best.pt \
  format=tflite \
  int8=True \
  imgsz=320
```

Generated model:

```text
best_int8.tflite
```

---

# Why INT8?

For Raspberry Pi Zero 2 W:
- faster inference
- lower RAM usage
- smaller model size
- lower CPU load

---

# Expected Raspberry Pi Performance

| Hardware | FPS |
|---|---|
| Raspberry Pi Zero 2 W | 3–6 FPS |

---

# Next Phase

Deploy model on Raspberry Pi:

CSI Camera
    ↓
Picamera2
    ↓
TFLite Runtime
    ↓
Realtime Pedestrian Detection

---


# 🚶 Pedestrian AI — Real-Time Detection on Raspberry Pi Zero 2W

## Overview

A real-time pedestrian detection system running on a **Raspberry Pi Zero 2W** (headless, SSH-only), streaming live annotated video to any browser on the local network using a TFLite INT8 model and Flask.

\---

## Hardware

|Component|Details|
|-|-|
|Board|Raspberry Pi Zero 2W|
|OS|Raspberry Pi OS — Debian 13 (Trixie), aarch64|
|Kernel|Linux 6.12.75+rpt-rpi-v8|
|Camera|OV5647 (Pi Camera v1) via CSI|
|RAM|512 MB|

\---

## Software Stack

|Component|Version|
|-|-|
|Python|3.11.13 (built from source via `altinstall`)|
|tflite-runtime|2.14.0|
|numpy|< 2.0 (pinned for tflite compatibility)|
|Flask|latest|
|OpenCV|`opencv-python-headless`|
|Camera backend|`rpicam-vid` (libcamera CLI)|

\---

## Model

|Property|Value|
|-|-|
|File|`best\_int8.tflite`|
|Input shape|`\[1, 320, 320, 3]` — float32|
|Output shape|`\[1, 5, 2100]` — float32|
|Output format|YOLO-style: `\[x, y, w, h, confidence]` × 2100 candidates|
|Confidence threshold|0.4 (tunable)|

\---

## Architecture

```
OV5647 Camera
     │
     ▼
rpicam-vid (MJPEG, 640×480 @ 15fps)
     │ stdout pipe
     ▼
Python 3.11 — OpenCV JPEG decoder
     │
     ├──► TFLite Interpreter (best\_int8.tflite)
     │         └── Bounding box overlay (OpenCV)
     │
     ▼
Flask MJPEG stream → Browser (http://192.168.31.194:5000)
```

\---

## Key Challenges \& Solutions

### 1\. Python 3.13 only in Debian 13 Trixie

`python3.11` is not available via apt on Trixie. **Solution:** built Python 3.11.13 from source using `make altinstall`.

### 2\. tflite-runtime incompatible with NumPy 2.x

`tflite-runtime 2.14.0` was compiled against NumPy 1.x. **Solution:** pinned `numpy<2`.

### 3\. picamera2 / libcamera compiled for Python 3.13 only

`\_libcamera.cpython-313-aarch64-linux-gnu.so` cannot load in Python 3.11. **Solution:** bypassed picamera2 entirely — used `rpicam-vid` CLI piped via stdout, decoded with OpenCV.

### 4\. No swap manager (dphys-swapfile missing)

Debian 13 uses ZRAM instead. **Solution:** used `fallocate` + `mkswap` + `swapon` directly during the Python build, then removed the swapfile afterward.

\---

## Project Structure

```
\~/tflite-project/
├── venv/                  # Python 3.11 virtual environment
├── best\_int8.tflite       # Pedestrian detection model
├── stream.py              # Flask + rpicam-vid streaming app
├── test\_model.py          # Model verification script
└── setup\_tflite.sh        # Full environment setup script
```

\---

## Setup Script Summary (`setup\_tflite.sh`)

1. Create 1GB swapfile (for compilation headroom)
2. Install build dependencies via apt
3. Download \& compile Python 3.11.13 from source
4. Create venv and install `tflite-runtime`, `flask`, `opencv-python-headless`
5. Verify tflite loads correctly
6. Remove swapfile

\---

## Running the Stream

```bash
source \~/tflite-project/venv/bin/activate
cd \~/tflite-project
python stream.py
```

Open in any browser on the same network:

```
http://192.168.31.194:5000
```

To keep running after closing SSH:

```bash
screen -S stream
python stream.py
# Detach: Ctrl+A then D
# Reattach: screen -r stream
```

\---

## Inference Details

* Frames captured at **640×480 @ 15fps** via `rpicam-vid`
* Resized to **320×320** for model input
* Inference runs on **every 2nd frame** to reduce CPU load on the Zero 2W
* Output transposed from `\[1, 5, 2100]` → `\[2100, 5]`, filtered by confidence
* Bounding boxes drawn in **green** with confidence score label

\---

## Tuning

|Parameter|Location|Effect|
|-|-|-|
|`CONF\_THRESH`|`stream.py` line 10|Lower → more detections, higher → stricter|
|`CAM\_W / CAM\_H`|`stream.py` line 11|Resolution (lower = faster)|
|`--framerate`|`stream.py` rpicam cmd|Reduce if CPU overloads|
|`skip % 2`|`stream.py` generate\_frames|Increase to skip more frames|





