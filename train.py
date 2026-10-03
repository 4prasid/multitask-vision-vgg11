"""
Training entrypoint
"""

from models.multitask import MultiTaskPerceptionModel
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import wandb
from models.layers import *
from models.vgg11 import *
from models.multitask import *
from data.pets_dataset import *
from models.classification import *
from models.localization import *
from models.segmentation import *
from losses.iou_loss import *

class Trainer:
    def __init__(self, model, bs, ep, device):

        super().__init__()

        self.model = model.to(device)
        self.bs = bs # batch size
        self.ep = ep # num epochs
        self.device = device # device

        # load the data and then split it into train and val using 80 - 20 split
        dataset = OxfordIIITPetDataset(root_dir='data/oxford-iiit-pet', split='trainval')
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator = generator)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.bs, shuffle=False)

        # define loss functions
        self.cls_loss = nn.CrossEntropyLoss()
        self.loc_loss_train = IoULoss(reduction = "mean")
        self.loc_loss_eval = IoULoss(reduction = "none")
        self.seg_loss = nn.CrossEntropyLoss()

        # define optimizer
        # encoder -> lower lr, decoder -> higher lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # define weights for loss functions
        self.w_cls = 1.0
        self.w_loc = 5.0
        self.w_seg = 2.0

    def train_epoch(self):

        # set model to train mode
        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_loc_loss = 0.0
        total_seg_loss = 0.0

        all_preds = []
        all_labels = []
        
        dice_total = 0
        iou_correct = 0
        total_samples = 0

        
        for images, labels, bboxes, masks in self.train_loader:

            # move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            bboxes = bboxes.to(self.device)
            masks = masks.to(self.device)

            # zero the gradients
            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(images)

            cls_pred = outputs['classification']
            loc_pred = outputs['localization']
            seg_pred = outputs['segmentation']
        
            # calculate loss
            cls_loss = self.cls_loss(cls_pred, labels)
            loc_loss = self.loc_loss_train(loc_pred, bboxes)
            seg_loss = self.seg_loss(seg_pred, masks)

            # total loss
            loss = self.w_cls * cls_loss + self.w_loc * loc_loss + self.w_seg * seg_loss

            # backward pass
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update weights
            self.optimizer.step()

            # classification
            preds = torch.argmax(cls_pred, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # localization
            iou_score = 1 - self.loc_loss_eval(loc_pred, bboxes)
            iou_correct += (iou_score > 0.5).sum().item()
            total_samples += images.size(0)

            # segmentation
            seg_out = torch.argmax(seg_pred, dim = 1)
            valid_mask = (masks != 2)
            pred_bin = ((seg_out == 1) & valid_mask).float()
            target_bin = ((masks == 1) & valid_mask).float()

            intersection = (pred_bin * target_bin).sum()
            union = pred_bin.sum() + target_bin.sum()
            dice_score = (2 * intersection) / (union + 1e-6)
            dice_total += dice_score.item()

            # update trackers
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_loc_loss += loc_loss.item()
            total_seg_loss += seg_loss.item()
        
        # update learning rate
        self.scheduler.step()

        # compute average loss
        total_loss /= len(self.train_loader)
        total_cls_loss /= len(self.train_loader)
        total_loc_loss /= len(self.train_loader)
        total_seg_loss /= len(self.train_loader)

        # metrics
        cls_acc = f1_score(all_labels, all_preds, average='macro')
        loc_acc = iou_correct / total_samples
        seg_dice = dice_total / len(self.train_loader)

        return total_loss, total_cls_loss, total_loc_loss, total_seg_loss, cls_acc, loc_acc, seg_dice

    def validate(self):

        self.model.eval()

        all_preds = []
        all_labels = []
        
        dice_total = 0
        iou_correct = 0
        total_samples = 0

        total_loss = 0.0
        total_cls_loss = 0.0
        total_loc_loss = 0.0
        total_seg_loss = 0.0

        with torch.no_grad():
            for images, labels, bboxes, masks in self.val_loader:

                # move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                bboxes = bboxes.to(self.device)
                masks = masks.to(self.device)

                # forward pass
                outputs = self.model(images)

                cls_pred = outputs['classification']
                loc_pred = outputs['localization']
                seg_pred = outputs['segmentation']

                # classification
                preds = torch.argmax(cls_pred, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # localization
                iou_score = 1 - self.loc_loss_eval(loc_pred, bboxes)
                iou_correct += (iou_score > 0.5).sum().item()
                total_samples += images.size(0)

                # segmentation
                seg_out = torch.argmax(seg_pred, dim = 1)
                # pred_bin = (seg_out == 1).float()
                # target_bin = (masks == 1).float()
                valid_mask = (masks != 2)
                pred_bin = ((seg_out == 1) & valid_mask).float()
                target_bin = ((masks == 1) & valid_mask).float()

                intersection = (pred_bin * target_bin).sum()
                union = pred_bin.sum() + target_bin.sum()
                dice_score = (2 * intersection) / (union + 1e-6)
                dice_total += dice_score.item()

                # losses
                cls_loss = self.cls_loss(cls_pred, labels)
                loc_loss = self.loc_loss_train(loc_pred, bboxes)
                seg_loss = self.seg_loss(seg_pred, masks)

                # total loss
                loss = self.w_cls * cls_loss + self.w_loc * loc_loss + self.w_seg * seg_loss    

                # update trackers
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_loc_loss += loc_loss.item()
                total_seg_loss += seg_loss.item()
        
        # compute metrics
        f1 = f1_score(all_labels, all_preds, average='macro')
        iou_acc = iou_correct / total_samples
        dice = dice_total / len(self.val_loader)

        # compute average loss
        total_loss /= len(self.val_loader)
        total_cls_loss /= len(self.val_loader)
        total_loc_loss /= len(self.val_loader)
        total_seg_loss /= len(self.val_loader)

        return total_loss, total_cls_loss, total_loc_loss, total_seg_loss, f1, iou_acc, dice

    def train(self):

        best_dice = 0.0

        for epoch in range(self.ep):

            train_loss, train_cls_loss, train_loc_loss, train_seg_loss, train_f1, train_iou_acc, train_dice = self.train_epoch()
            val_loss, val_cls_loss, val_loc_loss, val_seg_loss, val_f1, val_iou_acc, val_dice = self.validate()

            print(f"Epoch {epoch}: "
                  f"Train Loss = {train_loss:.4f} | "
                  f"Train Cls Loss = {train_cls_loss:.4f} | Train Loc Loss = {train_loc_loss:.4f} | Train Seg Loss = {train_seg_loss:.4f} | "
                  f"Train F1 = {train_f1:.4f} | Train IoU = {train_iou_acc:.4f} | Train Dice = {train_dice:.4f} | "
                  f"Val Loss = {val_loss:.4f} | "
                  f"Val Cls Loss = {val_cls_loss:.4f} | Val Loc Loss = {val_loc_loss:.4f} | Val Seg Loss = {val_seg_loss:.4f} | "
                  f"Val F1 = {val_f1:.4f} | Val IoU = {val_iou_acc:.4f} | Val Dice = {val_dice:.4f}")

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(self.model.state_dict(), "best_model.pth")

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_f1": train_f1,
                "train_iou": train_iou_acc,
                "train_dice": train_dice,
                "val_f1": val_f1,
                "val_iou": val_iou_acc,
                "val_dice": val_dice,
                "lr": self.scheduler.get_last_lr()[0], 
                "train_cls_loss": train_cls_loss,
                "train_loc_loss": train_loc_loss,
                "train_seg_loss": train_seg_loss,
                "val_cls_loss": val_cls_loss,
                "val_loc_loss": val_loc_loss,
                "val_seg_loss": val_seg_loss,
                "best_dice": best_dice
            })

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskPerceptionModel()

    trainer = Trainer(model, bs=32, ep=20, device=device)

    wandb.init(
    project="assignment_2",
    name=f"MTL_VGG11_bs{trainer.bs}_lr1e-4_wloc{trainer.w_loc}_wseg{trainer.w_seg}_ep{trainer.ep}",
    config={
        "batch_size": trainer.bs,
        "epochs": trainer.ep,
        "lr": 1e-4,
        "w_cls": trainer.w_cls,
        "w_loc": trainer.w_loc,
        "w_seg": trainer.w_seg
    }
)

    trainer.train()

    torch.save(model.state_dict(), 'model.pth')

    wandb.finish()

if __name__ == '__main__':
    main()