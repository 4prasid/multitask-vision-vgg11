'''
5. - Object Detection: Table + Images
Logs:
  A W&B Table with columns: Sample, IoU, Confidence, GT_class, Pred_class, Status
  Individual detection images (GT green, Pred red boxes)
  Summary media panel
'''

import os, random
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer
from models.classification import VGG11Classifier
from models.vgg11 import VGG11Encoder

wandb.login(key="") # update with wandb key

LOCALIZER_CKPT  = "checkpoints/localizer.pth"
CLASSIFIER_CKPT = "checkpoints/classifier.pth"
NUM_SAMPLES     = 15

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denormalize(t):
    img = t.cpu().float() * IMAGENET_STD + IMAGENET_MEAN
    return (img.clamp(0,1).numpy().transpose(1,2,0) * 255).astype(np.uint8)

def iou_single(boxA, boxB):
    ax1=boxA[0]-boxA[2]/2; ay1=boxA[1]-boxA[3]/2
    ax2=boxA[0]+boxA[2]/2; ay2=boxA[1]+boxA[3]/2
    bx1=boxB[0]-boxB[2]/2; by1=boxB[1]-boxB[3]/2
    bx2=boxB[0]+boxB[2]/2; by2=boxB[1]+boxB[3]/2
    inter = max(0,min(ax2,bx2)-max(ax1,bx1)) * max(0,min(ay2,by2)-max(ay1,by1))
    union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return float(inter/(union+1e-6))

det_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

if __name__ == "__main__":
    assert os.path.exists(LOCALIZER_CKPT),  f"Not found: {LOCALIZER_CKPT}"
    assert os.path.exists(CLASSIFIER_CKPT), f"Not found: {CLASSIFIER_CKPT}"

    # ── Load models ───────────────────────────────────────────────────────
    localizer = VGG11Localizer(in_channels=3)
    ckpt = torch.load(LOCALIZER_CKPT, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    localizer.load_state_dict(ckpt, strict=False)
    localizer.eval()

    encoder    = VGG11Encoder(in_channels=3)
    classifier = VGG11Classifier(num_classes=37, encoder=encoder)
    ckpt2 = torch.load(CLASSIFIER_CKPT, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt2: ckpt2 = ckpt2["state_dict"]
    classifier.load_state_dict(ckpt2, strict=False)
    classifier.eval()

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = OxfordIIITPetDataset(root_dir="data", split="trainval")
    dataset.transform = None
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)

    # ── Collect all results first, THEN init wandb and log ────────────────
    # This ensures nothing is created before wandb.init()
    results = []

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
                gt_orig[0]*224.0/orig_w, gt_orig[1]*224.0/orig_h,
                gt_orig[2]*224.0/orig_w, gt_orig[3]*224.0/orig_h,
            ])

            img_t = det_transform(image=img_np.astype(np.uint8))["image"]
            inp   = img_t.unsqueeze(0)

            with torch.no_grad():
                pred_box   = localizer(inp)[0].numpy()
                logits     = classifier(inp)
                probs      = F.softmax(logits, dim=1)[0]
                confidence = float(probs.max().item())
                pred_class = int(probs.argmax().item())

            iou = iou_single(pred_box, gt_box)
            is_failure = confidence > 0.5 and iou < 0.3
            status = "HIGH-CONF FAIL" if is_failure else ("FAIL" if iou < 0.3 else "OK")

            results.append({
                "sample":     i+1,
                "idx":        idx,
                "img_t":      img_t,
                "gt_box":     gt_box,
                "pred_box":   pred_box,
                "iou":        iou,
                "confidence": confidence,
                "gt_class":   int(label),
                "pred_class": pred_class,
                "status":     status,
            })
            print(f"  Sample {i+1:2d}: IoU={iou:.3f}  Conf={confidence:.3f}  {status}")

        except Exception as e:
            print(f"  Error idx={idx}: {e}")

    print(f"\nCollected {len(results)} samples. Initializing W&B...")

    # NOW init wandb — after all computation is done 
    wandb.init(project="da6401_assignment2", name="q2_5_detection_v4",
               group="report_extras")

    # Build and log table with images 
    table = wandb.Table(columns=[
        "image_with_boxes", "sample", "IoU", "confidence",
        "gt_class", "pred_class", "status"
    ])

    summary_images = []

    for r in results:
        # Draw figure
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(denormalize(r["img_t"]))

        for box, color, lbl in [
            (r["gt_box"],   'lime', 'GT'),
            (r["pred_box"], 'red',  'Pred'),
        ]:
            xc, yc, bw, bh = [float(v) for v in box]
            x1, y1 = xc-bw/2, yc-bh/2
            ax.add_patch(patches.Rectangle(
                (x1,y1), bw, bh, linewidth=2, edgecolor=color, facecolor='none'))
            ax.text(x1+2, max(y1+10,8), lbl, color=color, fontsize=8,
                    fontweight='bold', bbox=dict(facecolor='black', alpha=0.4, pad=1))

        ax.set_title(f"IoU={r['iou']:.3f}  Conf={r['confidence']:.3f}  [{r['status']}]",
                     fontsize=8, color='red' if r['iou'] < 0.3 else 'green')
        ax.axis('off')
        plt.tight_layout(pad=0.2)

        img_wandb = wandb.Image(fig)
        table.add_data(
            img_wandb,
            r["sample"],
            round(r["iou"], 3),
            round(r["confidence"], 3),
            r["gt_class"],
            r["pred_class"],
            r["status"]
        )
        summary_images.append(wandb.Image(
            fig, caption=f"Sample {r['sample']} | IoU={r['iou']:.3f} | Conf={r['confidence']:.3f} | {r['status']}"))
        plt.close(fig)

    #  Log everything in one call 
    wandb.log({
        "q2_5_detection_table":  table,
        "q2_5_all_detections":   summary_images,
        "q2_5_mean_iou":         round(float(np.mean([r["iou"]        for r in results])), 4),
        "q2_5_mean_confidence":  round(float(np.mean([r["confidence"] for r in results])), 4),
    })

    import time
    print("Waiting 30s for media to sync...")
    time.sleep(30)

    wandb.finish()
    print("\nDone.")
