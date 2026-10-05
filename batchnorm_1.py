'''
1. Regularization Effect of Dropout (BatchNorm Ablation Study)

Trains two models:
  Run A: VGG11 WITH BatchNorm    (standard config)
  Run B: VGG11 WITHOUT BatchNorm (ablation)

For each run, after every epoch, captures the activation distribution
of the 3rd convolutional layer (block2_conv) and logs it to W&B as a
histogram. Also logs train/val loss and F1 curves.

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy

from data.pets_dataset import OxfordIIITPetDataset
from models.layers import CustomDropout

wandb.login(key="") # write the wandb key here

# ── Config ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 20          # 20 epochs is enough to see convergence trend
LR         = 5e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROPOUT_P  = 0.5         # fixed for both runs (we're isolating BN here)

# ── Transforms ───────────────────────────────────────────────────────────────
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


# ── VGG11 Encoder variants ───────────────────────────────────────────────────

class VGG11EncoderNoBN(nn.Module):
    """VGG11 encoder WITHOUT BatchNorm — used only for the ablation run."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        def block(in_c, out_c, n_conv):
            layers = []
            for i in range(n_conv):
                layers += [nn.Conv2d(in_c if i == 0 else out_c, out_c,
                                     kernel_size=3, padding=1, bias=True),
                           nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.block1_conv = block(in_channels, 64,  1)
        self.block1_pool = nn.MaxPool2d(2, 2)
        self.block2_conv = block(64,  128, 1)    # ← 3rd conv layer overall
        self.block2_pool = nn.MaxPool2d(2, 2)
        self.block3_conv = block(128, 256, 2)
        self.block3_pool = nn.MaxPool2d(2, 2)
        self.block4_conv = block(256, 512, 2)
        self.block4_pool = nn.MaxPool2d(2, 2)
        self.block5_conv = block(512, 512, 2)
        self.block5_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.block1_pool(self.block1_conv(x))
        x = self.block2_pool(self.block2_conv(x))
        x = self.block3_pool(self.block3_conv(x))
        x = self.block4_pool(self.block4_conv(x))
        x = self.block5_pool(self.block5_conv(x))
        return x


class VGG11Classifier(nn.Module):
    """Classifier that accepts an encoder (with or without BN)."""

    def __init__(self, encoder, num_classes=37, dropout_p=0.5):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Sequential(nn.Linear(512*7*7, 4096), nn.ReLU(inplace=True), CustomDropout(p=dropout_p))
        self.fc2 = nn.Sequential(nn.Linear(4096, 4096),    nn.ReLU(inplace=True), CustomDropout(p=dropout_p))
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        return self.fc3(self.fc2(self.fc1(x)))


# ── Activation hook helper ───────────────────────────────────────────────────

class ActivationCapture:
    """Registers a forward hook on a module and stores the last output."""

    def __init__(self, module: nn.Module):
        self.activations = None
        self._hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.activations = output.detach().cpu()

    def remove(self):
        self._hook.remove()


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_loaders():
    dataset = OxfordIIITPetDataset(root_dir="data", split="trainval")
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


# ── Single training run ───────────────────────────────────────────────────────

def run_experiment(use_batchnorm: bool, train_loader, val_loader):
    run_name = f"q2_1_{'with_bn' if use_batchnorm else 'no_bn'}"
    print(f"\n{'='*60}")
    print(f"  Starting run: {run_name}")
    print(f"{'='*60}")

    # Build model
    if use_batchnorm:
        from models.vgg11 import VGG11Encoder
        encoder = VGG11Encoder(in_channels=3)
        # The 3rd conv layer is the first conv inside block2_conv (index 0 of that Sequential)
        hook_layer = encoder.block2_conv[0]   # Conv2d(64→128)
    else:
        encoder   = VGG11EncoderNoBN(in_channels=3)
        hook_layer = encoder.block2_conv[0]   # Conv2d(64→128), no BN after it

    model = VGG11Classifier(encoder, num_classes=37, dropout_p=DROPOUT_P).to(DEVICE)
    torch.backends.cudnn.benchmark = True

    # Activation hook on the 3rd conv (= first conv in block2)
    capture = ActivationCapture(hook_layer)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    wandb.init(
        project="da6401_assignment2",
        name=run_name,
        group="q2_1_batchnorm_ablation",
        config={
            "use_batchnorm": use_batchnorm,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "dropout_p": DROPOUT_P,
        },
        reinit=True
    )

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):

        # ── Train ──
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

        # ── Validation ──
        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        # capture activations on a fixed batch for stable histogram
        act_logged = False

        with torch.no_grad():
            for batch_idx, (images, labels, _, _) in enumerate(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                val_loss += criterion(logits, labels).item()
                all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Log activation histogram from first val batch once per epoch
                if batch_idx == 0 and not act_logged:
                    act_vals = capture.activations.numpy().flatten()
                    wandb.log({
                        "act_hist/block2_conv3": wandb.Histogram(act_vals),
                        "act_stats/mean":  float(act_vals.mean()),
                        "act_stats/std":   float(act_vals.std()),
                        "epoch": epoch,
                    }, commit=False)
                    act_logged = True

        val_f1   = f1_score(all_labels, all_preds, average="macro")
        val_loss /= len(val_loader)
        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

        print(f"[{run_name}] Epoch {epoch+1}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"train_f1={train_f1:.4f}  val_f1={val_f1:.4f}")

        wandb.log({
            "epoch":        epoch,
            "train_loss":   train_loss,
            "val_loss":     val_loss,
            "train_f1":     train_f1,
            "val_f1":       val_f1,
            "lr":           scheduler.get_last_lr()[0],
            "best_val_f1":  best_val_f1,
        })

    capture.remove()
    wandb.finish()
    print(f"Run {run_name} complete.  Best val F1 = {best_val_f1:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader = make_loaders()

    # Run 1: WITH BatchNorm
    run_experiment(use_batchnorm=True,  train_loader=train_loader, val_loader=val_loader)

    # Run 2: WITHOUT BatchNorm
    run_experiment(use_batchnorm=False, train_loader=train_loader, val_loader=val_loader)

    print("\nAll 1. runs complete.")