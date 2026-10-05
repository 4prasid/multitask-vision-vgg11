'''
2. Internal Dynamics (Dropout Ablation Study)

Trains three models with identical architecture and hyperparameters,
varying only the dropout probability:
  Run A: No dropout   (p=0.0)
  Run B: Dropout p=0.2
  Run C: Dropout p=0.5

Logs Train vs. Validation Loss and F1 curves for all three runs to W&B
under the same group so they can be overlaid in a single panel.

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.vgg11 import VGG11Encoder

wandb.login(key="") # write the wandb key here

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 20          # 20 epochs is enough to reveal the generalisation gap
LR           = 5e-5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Three dropout conditions to compare
DROPOUT_RUNS = [
    {"p": 0.0,  "label": "no_dropout"},
    {"p": 0.2,  "label": "dropout_p0.2"},
    {"p": 0.5,  "label": "dropout_p0.5"},
]

# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco', label_fields=[]))

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco', label_fields=[]))


# ── Dataset ───────────────────────────────────────────────────────────────────
def make_loaders():
    dataset = OxfordIIITPetDataset(root_dir="data", split="trainval")
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # NOTE: both subsets share the same underlying dataset object,
    # so we set transforms AFTER splitting via dataset.transform directly.
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform   = val_transform
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ── Single training run ───────────────────────────────────────────────────────
def run_experiment(dropout_p: float, label: str, train_loader, val_loader):
    run_name = f"q2_2_{label}"
    print(f"\n{'='*60}")
    print(f"  Starting run: {run_name}  (dropout_p={dropout_p})")
    print(f"{'='*60}")

    # Fresh encoder + classifier for each run (no weight sharing)
    encoder = VGG11Encoder(in_channels=3)
    model   = VGG11Classifier(
        num_classes=37,
        dropout_p=dropout_p,
        encoder=encoder
    ).to(DEVICE)
    torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    wandb.init(
        project="da6401_assignment2",
        name=run_name,
        group="q2_2_dropout_ablation",      # ← same group → overlay in W&B
        config={
            "dropout_p":  dropout_p,
            "batch_size": BATCH_SIZE,
            "epochs":     EPOCHS,
            "lr":         LR,
        },
        reinit=True
    )

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss, all_preds, all_labels = 0.0, [], []

        for images, labels, _, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_f1   = f1_score(all_labels, all_preds, average="macro")
        train_loss /= len(train_loader)

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []

        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                val_loss += criterion(logits, labels).item()
                all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1   = f1_score(all_labels, all_preds, average="macro")
        val_loss /= len(val_loader)
        scheduler.step()

        overfit_gap = train_f1 - val_f1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

        print(f"[{run_name}] Epoch {epoch+1}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"train_f1={train_f1:.4f}  val_f1={val_f1:.4f}  "
              f"gap={overfit_gap:.4f}")

        wandb.log({
            "epoch":        epoch,
            "train_loss":   train_loss,
            "val_loss":     val_loss,
            "train_f1":     train_f1,
            "val_f1":       val_f1,
            "overfit_gap":  overfit_gap,
            "lr":           scheduler.get_last_lr()[0],
            "best_val_f1":  best_val_f1,
        })

    wandb.finish()
    print(f"Run {run_name} complete.  Best val F1 = {best_val_f1:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader = make_loaders()

    for cfg in DROPOUT_RUNS:
        run_experiment(
            dropout_p=cfg["p"],
            label=cfg["label"],
            train_loader=train_loader,
            val_loader=val_loader,
        )

    print("\nAll 2. runs complete.")
