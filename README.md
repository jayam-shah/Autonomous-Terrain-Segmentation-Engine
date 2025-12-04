# Autonomous Terrain Segmentation Engine

> GPU-trained semantic segmentation pipeline for navigation and terrain understanding (aka **`semnav_terrain`**).

This project implements a compact **terrain segmentation engine** built and trained on a **GPU with PyTorch + CUDA** and exported to **ONNX** for fast, real-time inference. It predicts pixel-wise terrain classes and combines them with **pseudo-depth** and **object detection** to highlight safe free-space and nearby obstacles from a regular RGB webcam.

The core model is a **DeepLabV3 + MobileNetV3-Large** backbone tailored to a 7+ class outdoor navigation setup like below:

- `ground`
- `sidewalk`
- `stairs`
- `water`
- `person`
- `car`
- `sky`

---

## Features

- ✅ **GPU-trained semantic segmentation** using PyTorch (CUDA)
- ✅ **DeepLabV3 + MobileNetV3-Large** backbone
- ✅ **Albumentations** augmentations (resize, pad, flip, color jitter, small rotations)
- ✅ Dataset utilities to clean labels and clamp IDs to a fixed class set
- ✅ **ONNX export** for deployment
- ✅ **Real-time webcam demo**:
  - Semantic segmentation (terrain classes)
  - **Free-space highlighting** (e.g. `ground` + `sidewalk`)
  - **Pseudo-depth** from a MiDaS ONNX model
  - **Object detection** using YOLOv5n ONNX
  - Optional **text-to-speech callouts** for nearby obstacles

Everything in this repo is bootstrapped from the Jupyter notebook:

> `Autonomous Terrain Segmentation Engine.ipynb`

---

## Project Structure

The notebook creates a project folder like:

```text
semnav_terrain/
  configs/
    deeplabv3_mbv3.yaml
    fusion_lite.yaml
  data/
    images/           # RGB images (PNG)
    labels/           # segmentation masks (PNG, single-channel or RGB)
    depth/            # optional depth maps (same naming as images)
    splits/
      fold0_train.txt
      fold0_val.txt
      fold0_test.txt
    eda_samples/
  export/
    model.onnx        # exported segmentation model
    model.ts          # optional TorchScript export
  models/
    midas_small.onnx  # MiDaS depth model (downloaded)
    yolov5n.onnx      # YOLOv5n detection model (downloaded)
  reports/
    figs/             # training curves, qualitative examples, etc.
    summary.csv       # metrics per epoch
  runs/
    fold0/
      best.pth        # best PyTorch checkpoint
  scripts/
    make_splits.py    # (from notebook)
    viz_samples.py    # (from notebook)
  semnav/
    __init__.py
    models/
      deeplabv3.py    # DeepLabV3 + MobileNetV3-Large model
      fusion_lite.py  # RGB + depth fusion model (lightweight)
    losses/
      dice.py
      boundary.py
    metrics/
      miou.py
      boundary_iou.py
      recall2m.py
  train.py            # training entrypoint
  infer.py            # export / batch inference
  webcam_demo.py      # live demo with ONNX, depth, YOLO, TTS
  requirements.txt
  README.md
````

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

If you’re using the structure from the notebook directly, your working dir may be `semnav_terrain/`.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows (PowerShell / CMD)
```

### 3. Install Python dependencies

Make sure you install a **GPU build of PyTorch** that matches your CUDA version (from the official PyTorch website), then:

```bash
pip install -r requirements.txt
```

Key packages include:

* `torch` (CUDA build)
* `torchvision`
* `numpy`, `yaml`, `tqdm`, `rich`
* `albumentations`, `matplotlib`, `opencv-python`
* `onnx`, **`onnxruntime-gpu`**

---

## Dataset Format

The dataset is expected under `semnav_terrain/data/`:

```text
data/
  images/
    <key>.png
  labels/
    <key>.png
  depth/                  # optional
    <key>.png / <key>.npy
  splits/
    fold0_train.txt
    fold0_val.txt
    fold0_test.txt
```

* Each line in `fold0_train.txt` / `fold0_val.txt` / `fold0_test.txt` is a **basename** (e.g. `city_001_frame_000123`) without extension.
* For each key, the code loads:

  * `images/<key>.png`
  * `labels/<key>.png`
  * optionally `depth/<key>.png` (or similar)

### Label conventions

The preprocessing step:

* Resizes labels to match image H×W (nearest-neighbor)
* Converts 3-channel masks to single-channel
* Clamps label values to `{0..6, 255}` where:

  * `0..6` → 7 terrain classes, consistent with the config
  * `255` → ignore index

This logic lives in the preprocessing part of the notebook (`label fix` cell).

---

## Configuration

Main config file:

```yaml
# configs/deeplabv3_mbv3.yaml
model: deeplabv3_mbv3
num_classes: 7
input_size: [512,384]
optimizer: {name: AdamW, lr: 3e-4, weight_decay: 0.01}
loss: {ce_weighted: true, dice: 0.3, boundary: 0.2}
train: {epochs: 5, batch_size: 2, amp: false}   # CPU-friendly starter settings
val: {tta_flip: true}
```

For full GPU training I ran with a **CUDA device** and increased training settings (larger batch size, more epochs, and mixed precision via `amp: true`).

There is also a `fusion_lite.yaml` for an RGB+depth fusion variant:

```yaml
model: fusion_lite
rgb_backbone: deeplabv3_mbv3
depth_backbone: resnet18_unetlite
fusion_scales: [8,4]
num_classes: 7
```

---

## Training (GPU)

Although the reference notebook shows a CPU-friendly loop for teaching/validation, I trained the actual model on a **single NVIDIA GPU**.

Basic training command:

```bash
cd semnav_terrain

python train.py \
  --cfg configs/deeplabv3_mbv3.yaml \
  --fold 0
```

What happens inside:

* Loads the configuration from `configs/deeplabv3_mbv3.yaml`
* Builds the `DeeplabV3_MBV3` model
* Uses `DiceLoss` + boundary-aware cross-entropy combination
* Builds train/val dataloaders from the `data/splits/fold0_*.txt` files
* Tracks validation **mIoU** each epoch
* Saves the best checkpoint to `runs/fold0/best.pth`

> **Note:** For GPU training, set your device to `cuda` and enable AMP (mixed precision) in the training script / config. I trained this engine end-to-end on GPU, which significantly speeds up experiments compared to CPU-only runs.

---

## Export to ONNX

After training, export the best checkpoint to ONNX:

```bash
cd semnav_terrain

python infer.py \
  --weights runs/fold0/best.pth \
  --export onnx \
  --output export/model.onnx
```

This produces `export/model.onnx`, which is what the webcam demo uses for real-time inference via `onnxruntime-gpu`.

---

## Downloading Auxiliary Models (MiDaS + YOLOv5n)

The notebook includes cells that download the depth and detection models into `semnav_terrain/models/`:

* `midas_small.onnx` – MiDaS small depth model
* `yolov5n.onnx` – YOLOv5n object detector

If you’re not running the notebook, you can create the directory and place those models manually:

```bash
mkdir -p semnav_terrain/models
# Place your midas_small.onnx and yolov5n.onnx here
```

---

## Real-Time Webcam Demo

Once you have:

* `export/model.onnx` (terrain segmentation)
* `models/midas_small.onnx` (pseudo-depth)
* `models/yolov5n.onnx` (object detection)

you can run the demo:

```bash
cd semnav_terrain

python webcam_demo.py \
  --weights export/model.onnx \
  --backend onnx \
  --input-size 512 384 \
  --classes ground sidewalk stairs water person car sky \
  --free-space ground sidewalk \
  --pseudo-depth models/midas_small.onnx \
  --det-onnx    models/yolov5n.onnx \
  --det-size 640 \
  --det-interval 2 \
  --det-thresh 0.35 \
  --max-distance 5.0 \
  --alpha 0.35 \
  --free-alpha 0.10 \
  --show-fps
```

Optional speech and rich detection callouts (as configured in the notebook) use additional flags like:

* `--speak` / `--speak-classes`
* `--cooldown`, `--speak-min-gap`, `--speak-per-class-gap`
* `--speak-miss-frames`, `--reannounce-drop`, `--reannounce-abs`, `--reannounce-time`

The demo overlays:

* Colorized segmentation
* Highlighted free space (e.g. drivable sidewalk/ground)
* Bounding boxes with depth-aware distance estimates from MiDaS
* (Optionally) spoken warnings for close obstacles

---

## Results

The training loop tracks and reports:

* Per-class and mean **Intersection-over-Union (mIoU)**
* Optional boundary-aware metrics (boundary IoU, recall@2m, etc.)

You can summarize your own results here, for example:

* mIoU on validation fold
* Qualitative screenshots (stored in `reports/figs/`)

```text
# Example (fill with your numbers)
mIoU (val, fold0): XX.XX
Best epoch: YY
```

---

## Notebook

The full pipeline (project scaffolding, preprocessing, training, export, demo) is documented step-by-step in:

> `Autonomous Terrain Segmentation Engine.ipynb`

I used this notebook to build the repo structure, run experiments on GPU, and then export a clean project layout for GitHub.

---

## Hardware & Performance

* Trained and tested primarily on a **single NVIDIA GPU** using **PyTorch + CUDA**
* Inference (ONNX) runs with **onnxruntime-gpu** for real-time or near real-time performance in the webcam demo
* Also supports CPU fallback (slower), which is why the default config uses small, CPU-friendly settings

---

## Acknowledgements

This project builds on several excellent open-source components:

* [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html) for DeepLabV3 + MobileNetV3-Large
* [Albumentations](https://albumentations.ai/) for data augmentation
* [ONNX](https://onnx.ai/) and [onnxruntime-gpu](https://onnxruntime.ai/) for fast deployment
* [MiDaS](https://github.com/isl-org/MiDaS) for monocular depth estimation
* [YOLOv5](https://github.com/ultralytics/yolov5) for object detection

---

## License

This project is for fun and made by Jayam Shah

