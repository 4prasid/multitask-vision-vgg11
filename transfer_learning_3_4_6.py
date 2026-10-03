'''
3. Transfer Learning Showdown + 4/5/6 logging

Trains the U-Net segmentation model under three strategies:
  Strategy A: Frozen encoder      (strict feature extractor)
  Strategy B: Partial fine-tuning (freeze blocks 1-3, unfreeze blocks 4-5)
  Strategy C: Full fine-tuning    (all weights updated)

Also logs — alongside 3 — everything needed for:
  4: Feature maps from block1 and block5 of the classifier encoder
  5: Bounding box prediction table (GT green, Pred red, IoU, confidence)
  6: Segmentation sample images + Pixel Accuracy vs Dice Score curves

'''

import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.pets_dataset import OxfordIIITPetDataset
from models.vgg11 import VGG11Encoder
from models.segmentation import VGG11UNet
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from losses.iou_loss import IoULoss

wandb.login(key="")

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE        = 16
EPOCHS            = 20
LR                = 1e-4
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_CKPT   = "checkpoints/classifier.pth"
LOCALIZER_CKPT    = "checkpoints/localizer.pth"   # needed for Q2.5
NUM_VIZ_SAMPLES   = 5    # for Q2.6 segmentation grid
NUM_DET_SAMPLES   = 15   # for Q2.5 detection table

#  Transforms 
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco', label_fields=[]))

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco', label_fields=[]))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denormalize(t):
    """Convert normalized image tensor [3,H,W] → uint8 numpy [H,W,3]."""
    img = t.cpu().float() * IMAGENET_STD + IMAGENET_MEAN
    img = (img.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return img

# Metrics 

def dice_score(pred_mask, gt_mask, num_classes=3, eps=1e-6):
    """Mean Dice over all classes (excluding ignore index if any)."""
    dice = 0.0
    for c in range(num_classes):
        pred_c = (pred_mask == c).float()
        gt_c   = (gt_mask   == c).float()
        inter  = (pred_c * gt_c).sum()
        dice  += (2 * inter + eps) / (pred_c.sum() + gt_c.sum() + eps)
    return (dice / num_classes).item()

def pixel_accuracy(pred_mask, gt_mask):
    return (pred_mask == gt_mask).float().mean().item()

def iou_single(boxA, boxB):
    """IoU for two boxes in [xc, yc, w, h] pixel format."""
    ax1 = boxA[0] - boxA[2]/2;  ay1 = boxA[1] - boxA[3]/2
    ax2 = boxA[0] + boxA[2]/2;  ay2 = boxA[1] + boxA[3]/2
    bx1 = boxB[0] - boxB[2]/2;  by1 = boxB[1] - boxB[3]/2
    bx2 = boxB[0] + boxB[2]/2;  by2 = boxB[1] + boxB[3]/2
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    areaA = (ax2-ax1) * (ay2-ay1)
    areaB = (bx2-bx1) * (by2-by1)
    union = areaA + areaB - inter
    return inter / (union + 1e-6)

#  Dataset 

def make_loaders():
    dataset    = OxfordIIITPetDataset(root_dir="data", split="trainval")
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform   = val_transform
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# Freeze helpers 

def freeze_all(encoder):
    for p in encoder.parameters():
        p.requires_grad = False

def unfreeze_blocks(encoder, blocks):
    """Unfreeze named blocks, e.g. blocks=['block4_conv','block4_pool',...]"""
    for name, module in encoder.named_modules():
        for b in blocks:
            if name.startswith(b):
                for p in module.parameters():
                    p.requires_grad = True

def unfreeze_all(encoder):
    for p in encoder.parameters():
        p.requires_grad = True


# 4: Feature map visualization 

def log_feature_maps(classifier_ckpt_path: str):
    """
    Pass one dog image through the trained classifier and log
    feature maps from block1 and block5 to W&B (Q2.4).
    Called once before the main training loop.
    """
    print("\nLogging feature maps...")
    encoder = VGG11Encoder(in_channels=3)
    cls_model = VGG11Classifier(num_classes=37, encoder=encoder)
    ckpt = torch.load(classifier_ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    cls_model.load_state_dict(ckpt, strict=False)
    cls_model.eval()

    # grab one val image (first dog image we can find, else just first image)
    dataset = OxfordIIITPetDataset(root_dir="data", split="trainval")
    dataset.transform = val_transform
    img_tensor, label, _, _ = dataset[0]
    img_input = img_tensor.unsqueeze(0)  # [1,3,224,224]

    with torch.no_grad():
        _, features = cls_model.encoder(img_input, return_features=True)

    block1_maps = features["block1"][0]  # [64, 224, 224]
    block5_maps = features["block5"][0]  # [512,  14,  14]

    def make_grid_fig(maps, title, max_channels=16):
        n = min(maps.shape[0], max_channels)
        cols = 8; rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
        axes = axes.flatten()
        for i in range(n):
            fm = maps[i].numpy()
            axes[i].imshow(fm, cmap='viridis')
            axes[i].axis('off')
        for j in range(n, len(axes)):
            axes[j].axis('off')
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        return fig

    wandb.init(project="da6401_assignment2", name="q2_4_feature_maps",
               group="report_extras", reinit=True)
    orig_img = denormalize(img_tensor)
    wandb.log({
        "q2_4/original_image": wandb.Image(orig_img, caption=f"Label: {label}"),
        "q2_4/block1_feature_maps": wandb.Image(make_grid_fig(block1_maps, "Block 1 (Conv1): Low-level edges")),
        "q2_4/block5_feature_maps": wandb.Image(make_grid_fig(block5_maps, "Block 5 (Last Conv): High-level semantics")),
    })
    plt.close('all')
    wandb.finish()
    print("4 feature maps logged.")


#  5: Detection table 

def log_detection_table(localizer_ckpt_path: str):
    """
    Log a W&B table with GT (green) and predicted (red) bounding boxes,
    IoU, and a pseudo-confidence score for Q2.5.
    """
    if not os.path.exists(localizer_ckpt_path):
        print(f"Localizer checkpoint not found at {localizer_ckpt_path}, skipping Q2.5.")
        return

    print("\nLogging Q2.5 detection table...")
    localizer = VGG11Localizer(in_channels=3)
    ckpt = torch.load(localizer_ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    localizer.load_state_dict(ckpt, strict=False)
    localizer.eval()

    dataset = OxfordIIITPetDataset(root_dir="data", split="trainval")
    dataset.transform = val_transform
    indices = random.sample(range(len(dataset)), NUM_DET_SAMPLES)

    table = wandb.Table(columns=["image", "IoU", "confidence", "gt_box", "pred_box", "note"])

    wandb.init(project="da6401_assignment2", name="q2_5_detection_table",
               group="report_extras", reinit=True)

    for idx in indices:
        img_tensor, _, bbox, _ = dataset[idx]
        inp = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred_box = localizer(inp)[0].numpy()   # [xc, yc, w, h] pixels

        # GT box: dataset returns coco format [x, y, w, h] -> convert to [xc, yc, w, h]
        if len(bbox) == 0:
            continue
        bx, by, bw, bh = bbox[0]
        gt_box = np.array([bx + bw/2, by + bh/2, bw, bh])

        iou = iou_single(pred_box, gt_box)

        # Pseudo-confidence: use 1 - normalized regression uncertainty
        # (we use IoU itself as a proxy since we have no explicit confidence head)
        confidence = float(iou)   # honest proxy

        # Draw bounding boxes on image
        img_np = denormalize(img_tensor)   # [H,W,3] uint8
        H, W = img_np.shape[:2]
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(img_np)

        def draw_box(box, color, label):
            xc, yc, bw_, bh_ = box
            x1 = xc - bw_/2; y1 = yc - bh_/2
            rect = patches.Rectangle((x1, y1), bw_, bh_,
                                      linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-4, label, color=color, fontsize=7, fontweight='bold')

        draw_box(gt_box,   'green', 'GT')
        draw_box(pred_box, 'red',   f'Pred IoU={iou:.2f}')
        ax.axis('off')
        plt.tight_layout(pad=0.1)

        note = "high-conf low-IoU (failure)" if confidence > 0.3 and iou < 0.3 else ""
        table.add_data(
            wandb.Image(fig),
            round(iou, 3),
            round(confidence, 3),
            str(np.round(gt_box, 1).tolist()),
            str(np.round(pred_box, 1).tolist()),
            note
        )
        plt.close(fig)

    wandb.log({"q2_5/detection_table": table})
    wandb.finish()
    print("Detection table logged.")


#  Single segmentation training run (Q2.3 + Q2.6) 

def run_seg_experiment(strategy: str, train_loader, val_loader):
    """
    strategy: 'frozen' | 'partial' | 'full'
    Trains segmentation model and logs:
      - 3: val Dice, val loss, train loss curves
      - 6: pixel accuracy vs dice curves + 5-sample image grid each epoch
    """
    run_name = f"q2_3_seg_{strategy}"
    print(f"\n{'='*60}\n  Segmentation run: {run_name}\n{'='*60}")

    # Load pretrained encoder from classifier checkpoint
    encoder = VGG11Encoder(in_channels=3)
    if os.path.exists(CLASSIFIER_CKPT):
        ckpt = torch.load(CLASSIFIER_CKPT, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        enc_state = {k[len("encoder."):]: v for k, v in ckpt.items() if k.startswith("encoder.")}
        encoder.load_state_dict(enc_state, strict=True)
        print(f"  Loaded pretrained encoder from {CLASSIFIER_CKPT}")
    else:
        print(f"  WARNING: {CLASSIFIER_CKPT} not found — training from scratch")

    # Apply freezing strategy
    if strategy == "frozen":
        freeze_all(encoder)
        print("  Encoder: fully frozen")
    elif strategy == "partial":
        freeze_all(encoder)
        unfreeze_blocks(encoder, ["block4_conv", "block4_pool", "block5_conv", "block5_pool"])
        print("  Encoder: blocks 1-3 frozen, blocks 4-5 unfrozen")
    else:  # full
        unfreeze_all(encoder)
        print("  Encoder: fully unfrozen")

    model = VGG11UNet(num_classes=3, encoder=encoder).to(DEVICE)
    torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    wandb.init(
        project="da6401_assignment2",
        name=run_name,
        group="q2_3_transfer_showdown",
        config={
            "strategy": strategy,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
        },
        reinit=True
    )

    # Collect fixed val samples for 6 visualization (grabbed once)
    fixed_viz_samples = None

    for epoch in range(EPOCHS):

        #  Train 
        model.train()
        train_loss = 0.0
        for images, _, _, masks in train_loader:
            images = images.to(DEVICE)
            # masks from dataset: values 1/2/3 → remap to 0/1/2
            masks_long = (masks.long() - 1).clamp(0, 2).to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks_long)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        #  Validation 
        model.eval()
        val_loss = 0.0
        all_dice, all_pix_acc = [], []
        val_images_collected, val_gt_collected, val_pred_collected = [], [], []

        with torch.no_grad():
            for batch_idx, (images, _, _, masks) in enumerate(val_loader):
                images = images.to(DEVICE)
                masks_long = (masks.long() - 1).clamp(0, 2).to(DEVICE)
                logits = model(images)
                val_loss += criterion(logits, masks_long).item()

                pred_masks = logits.argmax(dim=1)  # [B, H, W]

                for i in range(images.shape[0]):
                    d  = dice_score(pred_masks[i].cpu(), masks_long[i].cpu())
                    pa = pixel_accuracy(pred_masks[i].cpu(), masks_long[i].cpu())
                    all_dice.append(d)
                    all_pix_acc.append(pa)

                # Collect samples for Q2.6 (only first 5 unique)
                if len(val_images_collected) < NUM_VIZ_SAMPLES:
                    needed = NUM_VIZ_SAMPLES - len(val_images_collected)
                    val_images_collected.extend([images[j].cpu() for j in range(min(needed, images.shape[0]))])
                    val_gt_collected.extend([masks_long[j].cpu() for j in range(min(needed, masks_long.shape[0]))])
                    val_pred_collected.extend([pred_masks[j].cpu() for j in range(min(needed, pred_masks.shape[0]))])

        val_loss    /= len(val_loader)
        mean_dice    = float(np.mean(all_dice))
        mean_pix_acc = float(np.mean(all_pix_acc))
        scheduler.step()

        print(f"[{run_name}] Epoch {epoch+1}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"dice={mean_dice:.4f}  pix_acc={mean_pix_acc:.4f}")

        # 6: segmentation sample grid (log every epoch) ───────────────
        seg_log = {}
        if epoch == 0 or epoch == EPOCHS - 1 or (epoch + 1) % 5 == 0:
            colormap = np.array([[0,0,0],[255,0,0],[0,255,0]], dtype=np.uint8)  # bg/fg/boundary
            fig, axes = plt.subplots(NUM_VIZ_SAMPLES, 3, figsize=(9, NUM_VIZ_SAMPLES*3))
            for i in range(NUM_VIZ_SAMPLES):
                orig   = denormalize(val_images_collected[i])
                gt_col = colormap[val_gt_collected[i].numpy()]
                pr_col = colormap[val_pred_collected[i].numpy()]
                axes[i][0].imshow(orig);    axes[i][0].set_title("Original" if i==0 else ""); axes[i][0].axis('off')
                axes[i][1].imshow(gt_col);  axes[i][1].set_title("GT Mask"  if i==0 else ""); axes[i][1].axis('off')
                axes[i][2].imshow(pr_col);  axes[i][2].set_title("Pred Mask" if i==0 else ""); axes[i][2].axis('off')
            plt.suptitle(f"{strategy} — Epoch {epoch+1}", fontsize=11)
            plt.tight_layout()
            seg_log[f"q2_6/seg_samples_{strategy}"] = wandb.Image(fig)
            plt.close(fig)

        wandb.log({
            "epoch":           epoch,
            "train_loss":      train_loss,
            "val_loss":        val_loss,
            "val_dice":        mean_dice,           # 3 + 6
            "val_pixel_acc":   mean_pix_acc,        # 6
            "lr":              scheduler.get_last_lr()[0],
            **seg_log
        })

    wandb.finish()
    print(f"Run {run_name} complete.")


#  Main 

if __name__ == "__main__":
    #  4: log feature maps (one-off, fast, ~30 sec) 
    if os.path.exists(CLASSIFIER_CKPT):
        log_feature_maps(CLASSIFIER_CKPT)
    else:
        print(f"Skipping 4: {CLASSIFIER_CKPT} not found.")

    #  5: log detection table (one-off, fast, ~1 min) 
    log_detection_table(LOCALIZER_CKPT)

    #  3 + 6: three segmentation training runs 
    train_loader, val_loader = make_loaders()

    for strategy in ["frozen", "partial", "full"]:
        run_seg_experiment(strategy, train_loader, val_loader)


    print("All done!")