"""
8 - Joint Multi-Task Training
Trains all 3 tasks (classification, localization, segmentation) jointly.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from models.multitask import MultiTaskPerceptionModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.layers import *
from models.vgg11 import *
from models.multitask import *
from data.pets_dataset import *
from models.classification import *
from models.localization import *
from models.segmentation import *
from losses.iou_loss import *


WANDB_KEY = "" # write the wandb key here


class Trainer:
    def __init__(self, model, bs, ep, device):

        self.model  = model.to(device)
        self.bs     = bs
        self.ep     = ep
        self.device = device

        # ── Transforms ────────────────────────────────────────────────────
        # bbox_params with clip=True: albumentations rescales bbox from
        # original pixel space → 224 space AND clips out-of-bounds coords automatically.
        train_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=[],
            min_area=0,
            min_visibility=0,
            clip=True        # clips out-of-bounds XML annotations safely
        ))

        val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=[],
            min_area=0,
            min_visibility=0,
            clip=True
        ))

        # ── Dataset ───────────────────────────────────────────────────────
        dataset    = OxfordIIITPetDataset(root_dir='data', split='trainval')
        train_size = int(0.8 * len(dataset))
        val_size   = len(dataset) - train_size
        generator  = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator)

        self.train_dataset.dataset.transform = train_transform
        self.val_dataset.dataset.transform   = val_transform

        print(f"Dataset: {train_size} train, {val_size} val samples")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.bs,
            shuffle=True,  num_workers=2, pin_memory=True)
        self.val_loader = DataLoader(
            self.val_dataset,   batch_size=self.bs,
            shuffle=False, num_workers=2, pin_memory=True)

        # ── Loss functions ─────────────────────────────────────────────────
        self.cls_loss       = nn.CrossEntropyLoss()
        self.loc_loss_train = IoULoss(reduction="mean")
        self.loc_loss_eval  = IoULoss(reduction="none")
        # CrossEntropy for segmentation: handles 3-class setup cleanly,
        # gives explicit gradient for the border class too
        self.seg_loss       = nn.CrossEntropyLoss()

        # ── Optimizer ──────────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1)

        # ── Loss weights ───────────────────────────────────────────────────
        # loc weighted 5x: IoU loss is bounded [0,1], CE losses unbounded
        self.w_cls = 1.0
        self.w_loc = 5.0
        self.w_seg = 2.0

    # ── bbox conversion ───────────────────────────────────────────────────
    def _prepare_bboxes(self, bboxes):
        """
        After albumentations (with bbox_params + clip=True), bbox is in
        COCO format [x_min, y_min, w, h] in 224x224 pixel space.
        Convert to PIXEL-SPACE center format [xc, yc, w, h] (NOT normalized).

        This matches the MultiTaskPerceptionModel.forward() output, which
        converts sigmoid outputs back to pixel coords:
            xc = loc[:, 0] * W,  yc = loc[:, 1] * H,  bw = loc[:, 2] * W,  bh = loc[:, 3] * H
        IoULoss works in any consistent space as long as pred & target match.
        """
        b  = bboxes.float().clone()
        xc = b[:, 0] + b[:, 2] / 2   # x_min + w/2  (pixel center x)
        yc = b[:, 1] + b[:, 3] / 2   # y_min + h/2  (pixel center y)
        w  = b[:, 2]                  # width  in pixels
        h  = b[:, 3]                  # height in pixels
        return torch.stack([xc, yc, w, h], dim=1).to(self.device)

    # ── Train one epoch ───────────────────────────────────────────────────
    def train_epoch(self):
        self.model.train()

        total_loss = total_cls_loss = total_loc_loss = total_seg_loss = 0.0
        all_preds  = []
        all_labels = []
        dice_total  = 0.0
        iou_correct = total_samples = 0

        for images, labels, bboxes, masks in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            masks  = masks.to(self.device)
            bboxes = self._prepare_bboxes(bboxes)

            self.optimizer.zero_grad()

            outputs  = self.model(images)
            cls_pred = outputs['classification']
            loc_pred = outputs['localization']
            seg_pred = outputs['segmentation']

            cls_loss = self.cls_loss(cls_pred, labels)
            loc_loss = self.loc_loss_train(loc_pred, bboxes)
            seg_loss = self.seg_loss(seg_pred, masks)
            loss     = (self.w_cls * cls_loss
                        + self.w_loc * loc_loss
                        + self.w_seg * seg_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Classification
            preds = torch.argmax(cls_pred, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Localization — Acc@IoU=0.5
            iou_score     = 1 - self.loc_loss_eval(loc_pred, bboxes)
            iou_correct  += (iou_score > 0.5).sum().item()
            total_samples += images.size(0)

            # Segmentation — Dice on pet foreground (class 1)
            # mask convention after (mask-1): 0=bg, 1=pet, 2=border
            seg_out    = torch.argmax(seg_pred, dim=1)
            valid_mask = (masks != 2)
            pred_bin   = ((seg_out == 1) & valid_mask).float()
            target_bin = ((masks   == 1) & valid_mask).float()
            inter = (pred_bin * target_bin).sum()
            union = pred_bin.sum() + target_bin.sum()
            dice_total += (2 * inter / (union + 1e-6)).item()

            total_loss     += loss.item()
            total_cls_loss += cls_loss.item()
            total_loc_loss += loc_loss.item()
            total_seg_loss += seg_loss.item()

        self.scheduler.step()
        n = len(self.train_loader)
        return (
            total_loss / n, total_cls_loss / n,
            total_loc_loss / n, total_seg_loss / n,
            f1_score(all_labels, all_preds, average='macro', zero_division=0),
            iou_correct / total_samples,
            dice_total / n,
        )

    # ── Validate ──────────────────────────────────────────────────────────
    def validate(self):
        self.model.eval()

        total_loss = total_cls_loss = total_loc_loss = total_seg_loss = 0.0
        all_preds  = []
        all_labels = []
        dice_total    = 0.0
        iou_correct   = total_samples = 0
        pixel_correct = pixel_total = 0

        with torch.no_grad():
            for images, labels, bboxes, masks in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                masks  = masks.to(self.device)
                bboxes = self._prepare_bboxes(bboxes)

                outputs  = self.model(images)
                cls_pred = outputs['classification']
                loc_pred = outputs['localization']
                seg_pred = outputs['segmentation']

                # Classification
                preds = torch.argmax(cls_pred, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Localization
                iou_score     = 1 - self.loc_loss_eval(loc_pred, bboxes)
                iou_correct  += (iou_score > 0.5).sum().item()
                total_samples += images.size(0)

                # Segmentation Dice on pet (class 1)
                seg_out    = torch.argmax(seg_pred, dim=1)
                valid_mask = (masks != 2)
                pred_bin   = ((seg_out == 1) & valid_mask).float()
                target_bin = ((masks   == 1) & valid_mask).float()
                inter = (pred_bin * target_bin).sum()
                union = pred_bin.sum() + target_bin.sum()
                dice_total += (2 * inter / (union + 1e-6)).item()

                # Pixel accuracy
                pixel_correct += (seg_out == masks).sum().item()
                pixel_total   += masks.numel()

                # Losses
                cls_loss = self.cls_loss(cls_pred, labels)
                loc_loss = self.loc_loss_train(loc_pred, bboxes)
                seg_loss = self.seg_loss(seg_pred, masks)
                loss     = (self.w_cls * cls_loss
                            + self.w_loc * loc_loss
                            + self.w_seg * seg_loss)

                total_loss     += loss.item()
                total_cls_loss += cls_loss.item()
                total_loc_loss += loc_loss.item()
                total_seg_loss += seg_loss.item()

        n = len(self.val_loader)
        return (
            total_loss / n, total_cls_loss / n,
            total_loc_loss / n, total_seg_loss / n,
            f1_score(all_labels, all_preds, average='macro', zero_division=0),
            iou_correct / total_samples,
            dice_total / n,
            pixel_correct / pixel_total,
        )

    # ── Full training loop ────────────────────────────────────────────────
    def train(self):
        best_val_dice = best_val_f1 = best_val_iou = 0.0

        for epoch in range(self.ep):
            (train_loss, train_cls_loss, train_loc_loss, train_seg_loss,
             train_f1, train_iou, train_dice) = self.train_epoch()

            (val_loss, val_cls_loss, val_loc_loss, val_seg_loss,
             val_f1, val_iou, val_dice, val_pixel_acc) = self.validate()

            best_val_dice = max(best_val_dice, val_dice)
            best_val_f1   = max(best_val_f1,   val_f1)
            best_val_iou  = max(best_val_iou,   val_iou)

            print(
                f"Epoch {epoch+1:02d}/{self.ep} | "
                f"Loss train={train_loss:.4f} val={val_loss:.4f} | "
                f"F1 train={train_f1:.4f} val={val_f1:.4f} | "
                f"IoU train={train_iou:.4f} val={val_iou:.4f} | "
                f"Dice train={train_dice:.4f} val={val_dice:.4f} | "
                f"PixAcc val={val_pixel_acc:.4f}"
            )

            if val_dice >= best_val_dice:
                torch.save(self.model.state_dict(),
                           "checkpoints/multitask_best.pth")

            wandb.log({
                "epoch": epoch + 1,

                # Total combined loss
                "loss/train_total":  train_loss,
                "loss/val_total":    val_loss,

                # Per-task losses
                "loss/train_cls":    train_cls_loss,
                "loss/val_cls":      val_cls_loss,
                "loss/train_loc":    train_loc_loss,
                "loss/val_loc":      val_loc_loss,
                "loss/train_seg":    train_seg_loss,
                "loss/val_seg":      val_seg_loss,

                # Per-task metrics
                "cls/train_f1":      train_f1,
                "cls/val_f1":        val_f1,
                "loc/train_iou_acc": train_iou,
                "loc/val_iou_acc":   val_iou,
                "seg/train_dice":    train_dice,
                "seg/val_dice":      val_dice,
                "seg/val_pixel_acc": val_pixel_acc,

                # Generalization gaps
                "gap/cls":           train_cls_loss - val_cls_loss,
                "gap/loc":           train_loc_loss - val_loc_loss,
                "gap/seg":           train_seg_loss - val_seg_loss,

                # Best so far
                "best/val_dice":     best_val_dice,
                "best/val_f1":       best_val_f1,
                "best/val_iou":      best_val_iou,

                # Learning rate
                "lr":                self.scheduler.get_last_lr()[0],
            })


def main():
    wandb.login(key=WANDB_KEY)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MultiTaskPerceptionModel(
        num_breeds=37,
        seg_classes=3,
        in_channels=3,
        classifier_path="checkpoints/classifier.pth",
        localizer_path="checkpoints/localizer.pth",
        unet_path="checkpoints/segmenter.pth",
    )

    # Free fragmented memory from loading 3 checkpoints
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    BS = 8    # Reduced: model has 3 VGG11 encoders = 3x memory
    EP = 30

    trainer = Trainer(model, bs=BS, ep=EP, device=device)

    wandb.init(
        project="da6401_assignment2",
        name=f"q2_8_multitask_joint_bs{BS}_ep{EP}",
        group="q2_8_meta_analysis",
        config={
            "batch_size":   BS,
            "epochs":       EP,
            "lr":           1e-4,
            "weight_decay": 1e-4,
            "w_cls":        trainer.w_cls,
            "w_loc":        trainer.w_loc,
            "w_seg":        trainer.w_seg,
            "optimizer":    "Adam",
            "scheduler":    "StepLR(step=10, gamma=0.1)",
            "cls_loss":     "CrossEntropyLoss",
            "loc_loss":     "Custom IoULoss",
            "seg_loss":     "CrossEntropyLoss",
            "backbone":     "VGG11 (pretrained from Task1/2/3 checkpoints)",
            "grad_clip":    1.0,
        }
    )

    os.makedirs("checkpoints", exist_ok=True)
    trainer.train()

    torch.save(model.state_dict(), "checkpoints/multitask_final.pth")
    print("Saved: checkpoints/multitask_final.pth")

    wandb.finish()
    print("Done.")


if __name__ == '__main__':
    main()
