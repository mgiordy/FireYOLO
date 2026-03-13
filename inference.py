from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import onnx
import onnxruntime as ort
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ultralytics.utils import ops
import nms

model_path_onnx = "results/tinyissimo-v8-b_VOC/weights/best.onnx"
image_path = "ultralytics/assets/bus_resized.jpg"


# Load a model
model = YOLO(model_path_onnx)

# Load the image
image = Image.open(image_path)

# Run inference on an image
results = model(image_path, imgsz=(256, 256))

print(f"Classes: {results[0].names}")

print("Results:")
for r in results:
    print(r.boxes)  



# Load the ONNX model
onnx_model = onnx.load(model_path_onnx)

try:
    onnx.checker.check_model(onnx_model)  # Validate model
except onnx.checker.ValidationError as e:
    print("Model is invalid: %s" % e)

print("Model is valid!")

ort_session = ort.InferenceSession(model_path_onnx)

# Resize the image to 256x256
image = Image.open(image_path)
resized_image = image.resize((256, 256))
input_tensor = np.array(resized_image) / 255.0

plt.figure(figsize=(5, 5))
plt.imshow(input_tensor)
plt.savefig('input.png')

# Standardise the image
# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_std = np.array([0.229, 0.224, 0.225])
# input_tensor = (input_tensor - imagenet_mean) / imagenet_std

# HWC to NCHW
input_tensor = np.transpose(input_tensor, (2, 0, 1))
input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

input_name = ort_session.get_inputs()[0].name
print(f"Input name  : {input_name}")
output = ort_session.run(None, {input_name: input_tensor})
output = torch.Tensor(output[0])
print(f"Output shape: {output.shape}")

# Ultralytics NMS implementation
preds = ops.non_max_suppression(torch.tensor(output), 0.25, 0.7)

print("\n\n\n")
print(preds)

# Custom NMS implementation
output = output.permute(0, 2, 1)
preds = nms.nms_yolov8(torch.tensor(output), 0.25, 0.7)


preds = preds[0]

print("\n\n\n")
print(preds)

# boxes = ops.scale_boxes(input_tensor.shape[2:], preds[:, :4], input_tensor.shape)

# Display the image with detections using Matplotlib
plt.figure(figsize=(10, 10))
fig, ax = plt.subplots(1, figsize=(10, 10))

image = Image.open(image_path)
resized_image = image.resize((256, 256))
input_tensor = np.array(resized_image)

# Extracting box details
x1, y1, x2, y2, scores, class_ids = preds.T

# Plot image
fig, ax = plt.subplots(1, figsize=(6, 6))
ax.imshow(input_tensor)

# Plot bounding boxes
for i in range(len(class_ids)):
    width = x2[i] - x1[i]
    height = y2[i] - y1[i]
    rect = patches.Rectangle((x1[i], y1[i]), width, height, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    
    # Add label
    label = f"Class {int(class_ids[i])}: {scores[i]:.2f}"
    ax.text(x1[i], y1[i] - 5, label, color='g', fontsize=10, backgroundcolor='black')

# Save output
plt.savefig('output.png')
plt.show()