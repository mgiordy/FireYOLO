import torch
from ultralytics import YOLO
import wandb
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Train and export YOLO model.')
parser.add_argument('--data', type=str, default='coco.yaml', help='Dataset configuration file')
parser.add_argument('--model', type=str, default='tinyissimo-v8-n', help='Model configuration file or name')
args = parser.parse_args()

wandb.init(project='results')

device = torch.device("cuda")

model_name = f"./ultralytics/cfg/models/tinyissimo/{args.model}.yaml"
model = YOLO(model_name)

img_size = 256
input_size = (1, 1, img_size, img_size)

# Must fit in 24GB RAM
if args.model == 'tinyissimo-v8-b' or args.model == 'tinyissimo-v8-n' or args.model == 'tinyissimo-v8-s':
    batch_size = 512
elif args.model == 'tinyissimo-v8-m' or args.model == 'tinyissimo-v8-l':
    batch_size = 256
elif args.model == 'tinyissimo-v8-x':
    batch_size = 128

# Train - WARNING! Overwrites existing trained models
folder_name = f"{args.model}_{args.data}"
model.train(data=f"{args.data}.yaml", project="results", name=folder_name, optimizer='SGD', imgsz=img_size, epochs=1000, batch=batch_size, exist_ok=True)

# Export
model.export(format="onnx", project="results", name=folder_name, imgsz=[img_size, img_size], exist_ok=True)

# Validate
onnx_model = YOLO(f"./results/{folder_name}/weights/best.onnx")
val_metrics = onnx_model.val(data=f"{args.data}.yaml", project="results", name=folder_name, task='detect', imgsz=img_size, batch=batch_size, exist_ok=True)

val_dict = {f"val_{key}": value for key, value in val_metrics.results_dict.items()}

wandb.log(val_dict)

wandb.finish()