'''
5 - Object Detection: Confidence & IoU
Logs each detection as an individual wandb.Image with bounding boxes
drawn directly on the image.

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

wandb.login(key="") # write the wandb key here

LOCALIZER_CKPT  = "checkpoints/localizer.pth"
NUM_SAMPLES     = 15

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
    return float(inter / (union + 1e-6))

det_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

if __name__ == "__main__":
    assert os.path.exists(LOCALIZER_CKPT), f"Not found: {LOCALIZER_CKPT}"

    wandb.init(project="da6401_assignment2", name="q2_5_detection_v2",
               group="report_extras")

    localizer = VGG11Localizer(in_channels=3)
    ckpt = torch.load(LOCALIZER_CKPT, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    localizer.load_state_dict(ckpt, strict=False)
    localizer.eval()

    dataset = OxfordIIITPetDataset(root_dir="data", split="trainval")
    dataset.transform = None
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)

    # Collect all images + stats for a summary grid log
    summary_images = []
    all_ious = []

    for i, idx in enumerate(indices):
        try:
            img_tensor, label, bbox, _ = dataset[idx]

            if not isinstance(bbox, torch.Tensor) or bbox.numel() < 4:
                continue

            gt_orig = bbox.numpy().astype(float)
            img_np = img_tensor.numpy() if isinstance(img_tensor, torch.Tensor) else np.array(img_tensor)
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)

            orig_h, orig_w = img_np.shape[:2]
            gt_box = np.array([
                gt_orig[0] * 224.0/orig_w,
                gt_orig[1] * 224.0/orig_h,
                gt_orig[2] * 224.0/orig_w,
                gt_orig[3] * 224.0/orig_h,
            ])

            img_t = det_transform(image=img_np.astype(np.uint8))["image"]
            with torch.no_grad():
                pred_box = localizer(img_t.unsqueeze(0))[0].numpy()

            iou = iou_single(pred_box, gt_box)
            all_ious.append(iou)

            # ── Draw figure ──────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(denormalize(img_t))

            for box, color, lbl in [
                (gt_box,   'lime',  f'GT'),
                (pred_box, 'red',   f'Pred'),
            ]:
                xc, yc, bw, bh = [float(v) for v in box]
                x1, y1 = xc - bw/2, yc - bh/2
                ax.add_patch(patches.Rectangle(
                    (x1, y1), bw, bh, linewidth=2,
                    edgecolor=color, facecolor='none'))
                ax.text(x1+2, max(y1+10, 8), lbl, color=color,
                        fontsize=8, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.4, pad=1))

            status = "FAIL" if iou < 0.3 else "OK"
            ax.set_title(f"IoU={iou:.3f}  [{status}]", fontsize=9,
                         color='red' if iou < 0.3 else 'green')
            ax.axis('off')
            plt.tight_layout(pad=0.2)

            caption = (f"Sample {i+1} | IoU={iou:.3f} | "
                       f"{'⚠ Failure: model missed the head' if iou < 0.2 else 'Partial detection' if iou < 0.5 else 'Good detection'}")

            # Log individually — always works in W&B
            wandb.log({f"q2_5/detection_{i+1:02d}": wandb.Image(fig, caption=caption)})
            summary_images.append(wandb.Image(fig, caption=caption))

            plt.close(fig)
            print(f"  Sample {i+1:2d}: idx={idx}  IoU={iou:.3f}  {status}")

        except Exception as e:
            print(f"  Error idx={idx}: {e}")

    # ── Log summary panel: all 15 images in one log call ─────────────────
    wandb.log({"q2_5/all_detections": summary_images})

    # ── Log scalar stats ──────────────────────────────────────────────────
    if all_ious:
        wandb.log({
            "q2_5/mean_iou":    round(float(np.mean(all_ious)), 4),
            "q2_5/median_iou":  round(float(np.median(all_ious)), 4),
            "q2_5/pct_above_05": round(float(np.mean(np.array(all_ious) > 0.5)), 4),
        })
        print(f"\n  Mean IoU: {np.mean(all_ious):.3f}")
        print(f"  Samples above IoU 0.5: {sum(v>0.5 for v in all_ious)}/{len(all_ious)}")

    wandb.finish()
    print("\nDone.")
