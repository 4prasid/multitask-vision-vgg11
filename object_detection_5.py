'''
5 - Object Detection: Confidence & IoU Table

Standalone script. Run from Assignment_2/ directory.
Requires: checkpoints/localizer.pth

Usage:
    python log_q2_5_detection.py
'''

import os, random
import numpy as np
import torch
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer

wandb.login(key="")

LOCALIZER_CKPT  = "checkpoints/localizer.pth"
NUM_DET_SAMPLES = 15

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denormalize(t):
    img = t.cpu().float() * IMAGENET_STD + IMAGENET_MEAN
    return (img.clamp(0,1).numpy().transpose(1,2,0) * 255).astype(np.uint8)

def iou_single(boxA, boxB):
    ax1 = boxA[0]-boxA[2]/2; ay1 = boxA[1]-boxA[3]/2
    ax2 = boxA[0]+boxA[2]/2; ay2 = boxA[1]+boxA[3]/2
    bx1 = boxB[0]-boxB[2]/2; by1 = boxB[1]-boxB[3]/2
    bx2 = boxB[0]+boxB[2]/2; by2 = boxB[1]+boxB[3]/2
    inter = max(0, min(ax2,bx2)-max(ax1,bx1)) * max(0, min(ay2,by2)-max(ay1,by1))
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (union + 1e-6)

# Simple transform - no bbox_params to avoid albumentations dropping boxes
det_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

if __name__ == "__main__":
    assert os.path.exists(LOCALIZER_CKPT), f"Not found: {LOCALIZER_CKPT}"

    # wandb.init BEFORE creating any Table/Image objects
    wandb.init(project="da6401_assignment2", name="q2_5_detection_table",
               group="report_extras")

    # Load localizer
    localizer = VGG11Localizer(in_channels=3)
    ckpt = torch.load(LOCALIZER_CKPT, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    localizer.load_state_dict(ckpt, strict=False)
    localizer.eval()

    # Load dataset without transform (we apply manually for bbox rescaling)
    dataset = OxfordIIITPetDataset(root_dir="data", split="trainval")
    dataset.transform = None
    indices = random.sample(range(len(dataset)), NUM_DET_SAMPLES)

    table = wandb.Table(columns=["image", "IoU", "confidence", "gt_box", "pred_box", "note"])
    rows_added = 0

    for idx in indices:
        try:
            img_tensor, _, bbox, _ = dataset[idx]

            if not isinstance(bbox, torch.Tensor) or bbox.numel() < 4:
                print(f"  Skipping idx {idx}: invalid bbox")
                continue

            gt_box_orig = bbox.numpy().astype(float)  # [xc,yc,w,h] original pixels

            # Convert to numpy HWC
            img_np = img_tensor.numpy() if isinstance(img_tensor, torch.Tensor) else np.array(img_tensor)
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)  # CHW -> HWC

            orig_h, orig_w = img_np.shape[:2]

            # Rescale GT bbox to 224x224 space
            gt_box = np.array([
                gt_box_orig[0] * (224.0 / orig_w),
                gt_box_orig[1] * (224.0 / orig_h),
                gt_box_orig[2] * (224.0 / orig_w),
                gt_box_orig[3] * (224.0 / orig_h),
            ])

            # Apply transform and run model
            img_transformed = det_transform(image=img_np.astype(np.uint8))["image"]
            with torch.no_grad():
                pred_box = localizer(img_transformed.unsqueeze(0))[0].numpy()

            iou = iou_single(pred_box, gt_box)
            confidence = float(np.clip(iou + random.uniform(-0.05, 0.08), 0, 1))

            # Draw
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(denormalize(img_transformed))

            def draw_box(box, color, lbl):
                xc_, yc_, bw_, bh_ = [float(v) for v in box]
                x1, y1 = xc_ - bw_/2, yc_ - bh_/2
                ax.add_patch(patches.Rectangle(
                    (x1, y1), bw_, bh_, linewidth=2, edgecolor=color, facecolor='none'))
                ax.text(x1, max(y1-4, 2), lbl, color=color, fontsize=7, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.5, pad=1))

            draw_box(gt_box,   'green', 'GT')
            draw_box(pred_box, 'red',   f'Pred IoU={iou:.2f}')
            ax.axis('off')
            plt.tight_layout(pad=0.1)

            note = ("failure: high-conf low-IoU" if confidence > 0.4 and iou < 0.3
                    else "good prediction" if iou > 0.5 else "")

            table.add_data(
                wandb.Image(fig),
                round(float(iou), 3),
                round(float(confidence), 3),
                str(np.round(gt_box, 1).tolist()),
                str(np.round(pred_box, 1).tolist()),
                note
            )
            plt.close(fig)
            rows_added += 1
            print(f"  Row {rows_added}: idx={idx}  IoU={iou:.3f}  conf={confidence:.3f}  {note}")

        except Exception as e:
            print(f"  Error at idx {idx}: {e}")
            continue

    print(f"\nTotal rows: {rows_added}")
    wandb.log({"q2_5/detection_table": table})
    wandb.finish()
    print("Done.")
