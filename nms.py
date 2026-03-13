import torch


def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    """
    Converts bounding boxes from (x, y, w, h) to (x1, y1, x2, y2)
    where (x, y) is the center of the box.
    
    Args:
        xywh (torch.Tensor): Tensor of shape (N, 4) where each row is (x, y, w, h).
    
    Returns:
        torch.Tensor: Converted tensor of shape (N, 4) with (x1, y1, x2, y2).
    """
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=1)


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two boxes (tensors)."""
    # Compute intersection coordinates
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)  # Avoid division by zero


def nms_yolov8(predictions, conf_threshold=0.25, iou_threshold=0.45, max_detections=300):
    """
    Perform Non-Maximum Suppression (NMS) on YOLOv8 output.
    
    Args:
        predictions (torch.Tensor): YOLOv8 output (B, N, 4 + n)
            - N: number of detections
            - Last dim: (x, y, w, h, class confidences)
        conf_threshold (float): Confidence threshold for filtering.
        iou_threshold (float): IoU threshold for suppression.
        max_detections (int): Maximum number of detections per image.

    Returns:
        List[torch.Tensor]: List of tensors (one per batch) with selected detections.
    """
    batch_size = predictions.shape[0]
    results = []

    for i in range(batch_size):
        preds = predictions[i]

        # Extract class scores and highest confidence class
        class_temp = preds[..., 4:].max(dim=-1)

        # Append confidence and class index to the bounding boxes
        preds = torch.cat((preds[:, :4], class_temp.values.unsqueeze(-1), class_temp.indices.unsqueeze(-1)), dim=-1)

        # Convert (x, y, w, h) -> (x1, y1, x2, y2)
        preds[:, :4] = xywh_to_xyxy(preds[:, :4])

        # Apply confidence threshold
        mask = preds[:, 4] > conf_threshold
        preds = preds[mask]

        if preds.shape[0] == 0:
            results.append(torch.empty((0, 6)))  # No detections
            continue

        # Sort by confidence score
        preds = preds[torch.argsort(preds[:, 4], descending=True)]

        # Perform NMS
        selected_boxes = []
        while preds.shape[0]:
            best_box = preds[0].unsqueeze(0)  # Take the highest confidence box
            selected_boxes.append(best_box)

            if len(selected_boxes) >= max_detections or preds.shape[0] == 1:
                break

            remaining_boxes = preds[1:]

            # Compute IoUs
            ious = torch.tensor([compute_iou(best_box[0, :4], box[:4]) for box in remaining_boxes])

            # Keep boxes with IoU < threshold
            preds = remaining_boxes[ious < iou_threshold]

        # Stack selected boxes
        results.append(torch.cat(selected_boxes, dim=0))

    return results  # Returns a list of tensors for each batch
