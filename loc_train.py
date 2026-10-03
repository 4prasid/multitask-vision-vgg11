import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier

wandb.login(key="") # write the wandb key here

def coco_to_center(boxes):
    """
    Convert bounding boxes from (x_min, y_min, w, h) to (x_c, y_c, w, h)
    """
    x_min, y_min, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_c = x_min + w / 2
    y_c = y_min + h / 2
    return torch.stack([x_c, y_c, w, h], dim=1)

def normalize_bboxes(bboxes, H, W):
    """
    Normalize bounding boxes to [0, 1]
    """
    bboxes = bboxes.clone()
    bboxes[:, 0] /= W
    bboxes[:, 1] /= H
    bboxes[:, 2] /= W
    bboxes[:, 3] /= H
    return bboxes

def compute_iou(pred, target):
    # reuse logic from IoULoss but return IoU
    pred_x1 = pred[:,0] - pred[:,2]/2
    pred_y1 = pred[:,1] - pred[:,3]/2
    pred_x2 = pred[:,0] + pred[:,2]/2
    pred_y2 = pred[:,1] + pred[:,3]/2

    target_x1 = target[:,0] - target[:,2]/2
    target_y1 = target[:,1] - target[:,3]/2
    target_x2 = target[:,0] + target[:,2]/2
    target_y2 = target[:,1] + target[:,3]/2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    area_p = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    area_t = (target_x2 - target_x1) * (target_y2 - target_y1)

    union = area_p + area_t - inter + 1e-6

    return inter / union


# config

BATCH_SIZE = 32
EPOCHS = 70
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# transforms

train_transform = A.Compose(
    [
        # A.Resize(224, 224),
        A.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0), ratio=(0.9, 1.1), p=0.6),
        A.HorizontalFlip(p=0.5), 
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.RandomBrightnessContrast(p=0.2),              
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

# create model and initialize localizer
model = VGG11Localizer(in_channels=3, dropout_p=0.5, return_normalized_coords=True).to(DEVICE)
torch.backends.cudnn.benchmark = True # speed up training

# load classifier weights
ckpt = torch.load("checkpoints/classifier.pth", map_location=DEVICE)

# load encoder weights from classifier
cls_model = VGG11Classifier(num_classes=37).to(DEVICE)
cls_model.load_state_dict(ckpt["state_dict"])

# transfer encoder weights from classifier to localizer
model.encoder.load_state_dict(cls_model.encoder.state_dict())

# freeze encoder from classifier training 
for param in model.encoder.parameters():
    param.requires_grad = False


# optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=5e-4
)


# learning rate scheduler, reduces learning rate by half every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# losses
iou_loss = IoULoss()
l1_loss = nn.SmoothL1Loss()

# wandb init

wandb.init(
project="da6401_assignment2",
name=f"loc_vgg11_bs{BATCH_SIZE}_lr{LR}_ep{EPOCHS}",
config={
"batch_size": BATCH_SIZE,
"epochs": EPOCHS,
"lr": LR
}
)

# training loop

best_score = 0

for epoch in range(EPOCHS):

    if epoch == 10:
        print("Unfreezing encoder")
        for name, param in model.encoder.named_parameters():
            if "block5" in name:
                param.requires_grad = True

        # reinitialize optimizer
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR * 0.02, # small learning rate for fine-tuning
            weight_decay= 1e-3
        )

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    if epoch < 5:
        w_iou = 0.2
        w_l1 = 1.0
    elif epoch < 15:
        w_iou = 0.7
        w_l1 = 1.0
    else:
        w_iou = 1.2
        w_l1 = 0.5

    # train
    model.train()

    train_loss = 0
    train_ious = []

    for images, _, bboxes, _ in train_loader:

        images = images.to(DEVICE)
        bboxes = bboxes.to(DEVICE)

        B, _, H, W = images.shape

        # convert
        bboxes = coco_to_center(bboxes)
        bboxes = normalize_bboxes(bboxes, H, W)

        optimizer.zero_grad()

        preds = model(images)

        preds = torch.clamp(preds, 1e-6, 1 - 1e-6)

        loss_iou = iou_loss(preds, bboxes)
        loss_l1 = l1_loss(preds, bboxes)

        # add size loss
        size_loss = torch.abs(preds[:, 2:] - bboxes[:, 2:]).mean()

        loss = w_iou*loss_iou + w_l1*loss_l1 + 0.05 * size_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
 
        train_loss += loss.item()

        iou = compute_iou(preds, bboxes)
        train_ious.extend(iou.detach().cpu().numpy())

    train_loss /= len(train_loader)
    train_ious = torch.from_numpy(np.array(train_ious))
    train_acc_05 = (train_ious >= 0.5).float().mean().item()
    train_acc_075 = (train_ious >= 0.75).float().mean().item()
    
    train_score = 0.6 * train_acc_05 + 0.4 * train_acc_075

    # validate
    model.eval()

    val_ious = []
    val_loss = 0

    with torch.no_grad():
        for images, _, bboxes, _ in val_loader:

            images = images.to(DEVICE)
            bboxes = bboxes.to(DEVICE)

            B, _, H, W = images.shape

            bboxes = coco_to_center(bboxes)
            bboxes = normalize_bboxes(bboxes, H, W)

            preds = model(images)

            preds = torch.clamp(preds, 1e-6, 1 - 1e-6)

            # add size loss
            size_loss = torch.abs(preds[:, 2:] - bboxes[:, 2:]).mean()

            loss = w_iou*iou_loss(preds, bboxes) + w_l1*l1_loss(preds, bboxes) + 0.05 * size_loss
            val_loss += loss.item()

            iou = compute_iou(preds, bboxes)
            val_ious.extend(iou.detach().cpu().numpy())

    val_loss /= len(val_loader)

    val_ious = torch.from_numpy(np.array(val_ious))

    val_acc_05  = (val_ious >= 0.5).float().mean().item()
    val_acc_075 = (val_ious >= 0.75).float().mean().item()

    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS}: "
          f"Train_Loss: {train_loss:.4f} | Train_Acc_at_0.5: {train_acc_05:.4f}| Train_Acc_at_0.75: {train_acc_075:.4f}|"
          f"Val_Loss: {val_loss:.4f}| Val_Acc_at_0.5: {val_acc_05:.4f}| Val_Acc_at_0.75: {val_acc_075:.4f}")

    # save best model
    val_score = 0.6 * val_acc_05 + 0.4 * val_acc_075

    if val_score > best_score:
        best_score = val_score
        # save best model in pixel coordinates
        model.return_normalized_coords = False
        torch.save({"state_dict": model.state_dict()}, "checkpoints/localizer.pth")
        print("Best model saved")
        model.return_normalized_coords = True

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_val_gap": train_loss - val_loss,
        "train_acc_05": train_acc_05,
        "train_acc_075": train_acc_075,
        "val_acc_05": val_acc_05,
        "val_acc_075": val_acc_075,
        "acc_gap_05": train_acc_05 - val_acc_05,
        "acc_gap_075": train_acc_075 - val_acc_075,
        "mean_val_iou": val_ious.mean().item(),
        "train_score": train_score,
        "val_score": val_score,
        "best_score": best_score,
        "lr": scheduler.get_last_lr()[0],
        "val_iou_hist": wandb.Histogram(val_ious.cpu().numpy())
    })

wandb.finish()
