import torch
from ultralytics import YOLO
import wandb
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Train and export YOLO model.')
parser.add_argument('--data', type=str, default='coco.yaml', help='Dataset configuration file')
parser.add_argument('--model', type=str, default='tinyissimo-v8-n', help='Model configuration file or name')
args = parser.parse_args()

load = False
exp_id = 'exp1'

version = 'v8'

if version == 'v1':
    print('Please, check to modify ultralytics/nn/modules/head/Detect')
    print('for TinyissimoYOLOv1.3 small and big change')
    print('line 36 to: self.reg_max=16')
    exit(1)

device = torch.device("cuda")
if load:
    model_name = f'./results/{exp_id}/weights/last.pt'
    model = YOLO(model_name)
else:
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

# Train
model.train(data=f"{args.data}.yaml", project="results", name="exp", optimizer='SGD', imgsz=img_size, epochs=1000, batch=batch_size)

# Export
model.export(format="onnx", project="results", name="exp", imgsz=[img_size, img_size])
