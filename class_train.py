'''
Code for training a VGG11 classifier on the Oxford-IIIT Pet Dataset.
Contains:
    - Data loading and preprocessing : loads the dataset and applies transformations
    - Transformation : applies transformations to the dataset
    - Model definition : defines the VGG11 classifier
    - Training loop : trains the model on the training data
    - Validation loop : validates the model on the validation data
    - Saving best model : saves the best model based on validation F1 score
    - Logging to wandb : logs the training and validation metrics to wandb
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

wandb.login(key="") # write the wandb key here

# config

BATCH_SIZE = 32
EPOCHS = 40
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# transforms

# data augmentation for training data, first resizes to 224x224, then applies random 
# horizontal flip, shift, scale, rotate, brightness, contrast, hue, saturation, noise
# then normalizes to ImageNet mean and std, then converts to tensor
train_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5), 
        A.ShiftScaleRotate(      
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.3), 
        A.HueSaturationValue(p=0.3),       
        A.GaussNoise(p=0.2),              

        A.Normalize(mean=(0.485, 0.456, 0.406), # normalize to ImageNet mean and std
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2() # convert to tensor
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=[]) 
)

# validation data transformation, only resize and normalize
val_transform = A.Compose( 
    [   A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=[])
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


# model
model = VGG11Classifier(num_classes=37, dropout_p=0.6).to(DEVICE)
torch.backends.cudnn.benchmark = True # speed up training


# loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# optimizer
optimizer = optim.Adam(
model.parameters(),
lr=LR,
weight_decay=5e-4
)

# learning rate scheduler, reduces learning rate by half every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# wandb init

wandb.init(
project="da6401_assignment2",
name=f"cls_vgg11_bs{BATCH_SIZE}_lr{LR}_ep{EPOCHS}",
config={
"batch_size": BATCH_SIZE,
"epochs": EPOCHS,
"lr": LR
}
)


# training loop

best_val_f1 = 0.0

for epoch in range(EPOCHS):

    # train
    model.train()
    train_loss = 0
    all_preds = []
    all_labels = []

    for images, labels, _, _ in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_f1 = f1_score(all_labels, all_preds, average="macro")
    train_loss /= len(train_loader)

    # validation
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0

    with torch.no_grad():
        for images, labels, _, _ in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(all_labels, all_preds, average="macro")
    val_loss /= len(val_loader)

    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")

    # save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({"state_dict": model.state_dict()}, "checkpoints/classifier.pth")
        print("Saved best model")

    # wandb log
    wandb.log({
        "epoch": epoch, # 0-indexed
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "lr": scheduler.get_last_lr()[0],
        "best_val_f1": best_val_f1,
        "overfit_gap": train_f1 - val_f1
    })

wandb.finish()
