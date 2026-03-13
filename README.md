# FireYOLO

Object detection training and deployment pipeline built on top of [TinyissimoYOLO](tinyissimoYOLO_README.md), using a customized [Ultralytics](https://github.com/ultralytics/ultralytics) codebase. Designed for training lightweight YOLO models that can run on resource-constrained platforms (MCUs, smart glasses, etc.).

---

## Setup

**Requirements:** Python >= 3.10, PyTorch >= 1.7, CUDA (recommended).

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Repository Structure

```
FireYOLO/
├── a_train_export.py            # Main script: train, export to ONNX, and validate
├── inference.py                 # Inference script (Ultralytics + ONNX Runtime flow)
├── nms.py                       # Custom NMS utilities
├── requirements.txt             # Python dependencies
├── sweeps/                      # W&B hyperparameter sweep configs
│   └── sweep_models_all_classes.yaml
├── ultralytics/                 # Local (modified) Ultralytics codebase
│   └── cfg/
│       ├── models/tinyissimo/   # Tinyissimo model definitions
│       └── datasets/            # Dataset YAMLs (COCO, VOC, etc.)
├── results_save/                # Saved training/export runs
└── wandb/                       # Weights & Biases run logs and artifacts
```

---

## Training

### Standard Training (`a_train_export.py`)

This is the main entry point. It trains a TinyissimoYOLO model, exports to ONNX, and runs validation — all in one script. Results are logged to [Weights & Biases](https://wandb.ai).

If the selected dataset is not already available locally, the script will download it automatically.

```bash
python a_train_export.py --model tinyissimo-v8-n --data coco
```

**Arguments:**

| Argument  | Default            | Description                                                                 |
|-----------|--------------------|-----------------------------------------------------------------------------|
| `--model` | `tinyissimo-v8-n`  | Model architecture. Maps to a YAML file in `ultralytics/cfg/models/tinyissimo/`. |
| `--data`  | `coco`             | Dataset config name (without `.yaml`). Maps to `ultralytics/cfg/datasets/`. |

**Available models:** `tinyissimo-v8-b`, `tinyissimo-v8-n`, `tinyissimo-v8-s`, `tinyissimo-v8-m`, `tinyissimo-v8-l`, `tinyissimo-v8-x`

These correspond to different width/depth scaling factors (from smallest `b` to largest `x`). Batch size is automatically adjusted based on model size to fit in ~24 GB VRAM.

**Training details:**
- Optimizer: SGD
- Image size: 128x128
- Epochs: 1000
- Output: `results/<model>_<data>/` (weights, metrics, ONNX export)

---

## Evaluation

Validation is run automatically at the end of training by `a_train_export.py`, both for the trained PyTorch model and for the exported ONNX model.

---

## Inference

Run inference on a single image using a trained model (supports both `.pt` and `.onnx`):

```bash
python inference.py
```

Edit `model_path_onnx` and `image_path` inside the script to point to your model and image. The script runs both Ultralytics and custom NMS pipelines and visualizes detections.

---

## Model Configuration

Model architectures live in `ultralytics/cfg/models/tinyissimo/`. Each YAML defines backbone and head layers, plus scaling factors:

```yaml
nc: 80        # Number of classes — change this to match your dataset
scale: 'n'    # Scale variant (b, n, s, m, l, x)
```

To use a custom number of classes, edit the `nc` field in the model YAML or in the dataset YAML.

---

## Dataset Configuration

Dataset YAML files in `ultralytics/cfg/datasets/` define paths to images/labels and class names. Available configs include COCO, VOC, and others. To add a custom dataset, create a new YAML following the same format.

---

## W&B Sweeps

To run a grid sweep across all models and datasets:

```bash
wandb sweep sweeps/sweep_models_all_classes.yaml
wandb agent <sweep_id>
```

This trains every combination of model size (`b` through `x`) on both VOC and COCO.

---

## Acknowledgments

Based on [TinyissimoYOLO](https://arxiv.org/abs/2311.01057) by ETH Zurich, built on top of [Ultralytics](https://github.com/ultralytics/ultralytics).