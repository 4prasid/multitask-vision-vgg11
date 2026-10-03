import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet
from models.classification import VGG11Classifier

wandb.login(key="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")

# config

BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# transforms

train_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.3),              
        A.Normalize(mean=(0.485, 0.456, 0.406), # normalize to ImageNet mean and std
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2() # convert to tensor
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=[]) # [x_min, y_min, width, height]
)


val_transform = A.Compose( 
    [   A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=[]) # [x_min, y_min, width, height]
)


# dataset
dataset = OxfordIIITPetDataset( root_dir="data", split="trainval")


# split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split( dataset,
[train_size, val_size],
generator=torch.Generator().manual_seed(42)
)


# apply transforms
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# create model and initialize segmetation model
model = VGG11UNet(in_channels=3, dropout_p=0.5).to(DEVICE)
torch.backends.cudnn.benchmark = True # speed up training

# load classifier weights
ckpt = torch.load("checkpoints/classifier.pth", map_location=DEVICE)

# load encoder weights from classifier
cls_model = VGG11Classifier(num_classes=37).to(DEVICE)
cls_model.load_state_dict(ckpt["state_dict"])

# transfer encoder weights from classifier to segmentation model
model.encoder.load_state_dict(cls_model.encoder.state_dict())

# unfreeze encoder
for param in model.encoder.parameters():
    param.requires_grad = True


# losses
weights = torch.tensor([0.2, 0.5, 0.3]).to(DEVICE)
ce_loss = nn.CrossEntropyLoss(weight=weights)

dice_weights = torch.tensor([0.2, 0.5, 0.3]).to(DEVICE)

def dice_loss_fn(logits, targets, smooth=1e-6):
    probs = torch.softmax(logits, dim=1)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=3).permute(0,3,1,2).float()

    intersection = (probs * targets_onehot).sum(dim=(2,3))
    union = probs.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3))

    dice = (2 * intersection + smooth) / (union + smooth)

    return 1 - (dice_weights * dice).mean() 

def combined_loss(logits, targets):
    return 0.3 * ce_loss(logits, targets) + 0.7 * dice_loss_fn(logits, targets)


# metrics
def compute_dice(logits, targets):
    preds = torch.argmax(logits, dim=1)

    dice = 0
    num_classes = logits.shape[1]
    valid_classes = 0

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        if union == 0:
            continue

        dice += (2 * intersection + 1e-6) / (union + 1e-6)
        valid_classes += 1

    return dice / (valid_classes + 1e-6)


# optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-5) 


# learning rate scheduler, reduces learning rate when validation loss stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)


# wandb init

wandb.init(
project="da6401_assignment2",
name=f"seg_vgg11_bs{BATCH_SIZE}_lr{LR}_ep{EPOCHS}",
config={
"batch_size": BATCH_SIZE,
"epochs": EPOCHS,
"lr": LR
}
)


# training loop
best_val_dice = 0

for epoch in range(EPOCHS):

    # train
    model.train()
    train_loss, train_dice = 0, 0

    for images, _, _, masks in train_loader:
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE).long()

        optimizer.zero_grad()

        logits = model(images)
        loss = combined_loss(logits, masks)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        optimizer.step()

        train_loss += loss.item()
        train_dice += compute_dice(logits, masks).item()

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)

    # validate

    model.eval()
    val_loss, val_dice = 0, 0

    with torch.no_grad():
        for images, _, _, masks in val_loader:
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE).long()

            logits = model(images)
            loss = combined_loss(logits, masks)

            val_loss += loss.item()
            val_dice += compute_dice(logits, masks).item()

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    scheduler.step(val_dice)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Dice: {val_dice:.4f}")

    wandb.log({
        "train_loss": train_loss,
        "train_dice": train_dice,
        "val_loss": val_loss,
        "val_dice": val_dice,
        "lr": optimizer.param_groups[0]['lr']
    })

    # save best model
    if val_dice > best_val_dice:
        best_val_dice = val_dice

        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, "checkpoints/segmenter.pth")

        print(f"Saved best model | Dice: {best_val_dice:.4f}")

wandb.finish()


