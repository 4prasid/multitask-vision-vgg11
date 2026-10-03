"""
Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """
    IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
        """
        super().__init__()
        self.eps = eps
        
        # check the reduction value
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        self.reduction = reduction 


    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format.
        """
        
        # check the size pred_boxes and target_boxes
        if pred_boxes.shape != target_boxes.shape or pred_boxes.shape[-1] != 4 or pred_boxes.ndim != 2:
            raise ValueError("pred_boxes and target_boxes must have the same shape [B, 4]")

        # Convert to (x_min, y_min, x_max, y_max) format 
        # (x_min, y_min, x_max, y_max) = (x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2)

        # pred_boxes_x1y1x2y2
        pred_x1y1 = pred_boxes[..., 0:2] - pred_boxes[..., 2:4] / 2
        pred_x2y2 = pred_boxes[..., 0:2] + pred_boxes[..., 2:4] / 2
        pred_boxes_x1y1x2y2 = torch.cat([pred_x1y1, pred_x2y2], dim=1)

        # target_boxes_x1y1x2y2
        target_x1y1 = target_boxes[..., 0:2] - target_boxes[..., 2:4] / 2
        target_x2y2 = target_boxes[..., 0:2] + target_boxes[..., 2:4] / 2
        target_boxes_x1y1x2y2 = torch.cat([target_x1y1, target_x2y2], dim=1)

        # compute intersection coordinates
        inter_x1 = torch.max(pred_boxes_x1y1x2y2[:, 0], target_boxes_x1y1x2y2[:, 0])
        inter_y1 = torch.max(pred_boxes_x1y1x2y2[:, 1], target_boxes_x1y1x2y2[:, 1])
        inter_x2 = torch.min(pred_boxes_x1y1x2y2[:, 2], target_boxes_x1y1x2y2[:, 2])
        inter_y2 = torch.min(pred_boxes_x1y1x2y2[:, 3], target_boxes_x1y1x2y2[:, 3])

        # compute intersection width and height
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)

        # compute intersection area
        inter_area = inter_w * inter_h

        # compute area from corner format: area = (x_max - x_min) * (y_max - y_min)

        # pred_area 
        pred_w = torch.clamp(pred_boxes_x1y1x2y2[:, 2] - pred_boxes_x1y1x2y2[:, 0], min=0)
        pred_h = torch.clamp(pred_boxes_x1y1x2y2[:, 3] - pred_boxes_x1y1x2y2[:, 1], min=0)
        pred_area = pred_w * pred_h

        # target_area 
        target_w = torch.clamp(target_boxes_x1y1x2y2[:, 2] - target_boxes_x1y1x2y2[:, 0], min=0)
        target_h = torch.clamp(target_boxes_x1y1x2y2[:, 3] - target_boxes_x1y1x2y2[:, 1], min=0)
        target_area = target_w * target_h

        # union_area = pred_area + target_area - inter_area
        union_area = pred_area + target_area - inter_area
        union_area = torch.clamp(union_area, min=self.eps) # avoid division by zero

        # compute iou
        iou = inter_area / union_area 

        # compute iou loss
        iou_loss = 1 - iou

        # apply reduction
        if self.reduction == "mean":
            return iou_loss.mean()
        elif self.reduction == "sum":
            return iou_loss.sum()
        else: # self.reduction == "none"
            return iou_loss
        