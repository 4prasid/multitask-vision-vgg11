# VGG11 Multi-Task Vision Pipeline

A from-scratch **VGG11** pipeline in PyTorch which, in a single forward pass, predicts a pet's **breed** (37 classes), its **head bounding box**, and a pixel-wise **trimap segmentation**. Built on the Oxford-IIIT Pet dataset with a custom dropout layer, a custom IoU loss, and a U-Net decoder that upsamples with transposed convolutions.

🔗 [W&B Report](https://forge.coreweave.com/wandb/prasid-indian-institute-of-technology-madras/assignment_1/reports/DA6401-Assignment-1-PH21B007-PRASID--VmlldzoxNjEyODA5Ng?accessToken=7lf6abidol3880zy7aiflc38domgwf0gtrwlsqz0fhboc9dumm1bdqjfn0fs1042) &nbsp;|&nbsp; 🔗 [GitHub Repo](https://github.com/4prasid/multitask-vision-vgg11)

---

Pipeline output on an unseen image

<img width="257" height="277" alt="image" src="https://github.com/user-attachments/assets/4045216d-a9af-4b02-a42f-621effa09c66" />
<img width="251" height="266" alt="image" src="https://github.com/user-attachments/assets/908f5f09-0173-4b2d-8aad-dde329cff009" />


---


## Repository structure

```
.
├── data/
│   └── pets_dataset.py        # Oxford-IIIT Pet loader -> (image, label, bbox, mask)
├── losses/
│   └── iou_loss.py            # Custom IoU loss (nn.Module)
├── models/
│   ├── vgg11.py               # VGG11 encoder (BatchNorm, skip-feature output)
│   ├── layers.py              # CustomDropout (inverted dropout, nn.Module)
│   ├── classification.py      # VGG11Classifier
│   ├── localization.py        # VGG11Localizer
│   ├── segmentation.py        # VGG11UNet
│   └── multitask.py           # MultiTaskPerceptionModel (unified)
├── class_train.py             # Stage 1: classification
├── loc_train.py               # Stage 2: localization
├── seg_train.py               # Stage 3: segmentation
├── train.py                   # Stage 4: unified multi-task training
├── evaluate.py                # Metrics for the unified model on the validation split
├── requirements.txt
├── FINDINGS.md                # Full written analysis of the experiments
└── LICENSE
```

---

## Architecture

### VGG11 encoder (`models/vgg11.py`)

Standard VGG11 topology (8 convolutions in 5 blocks) with **BatchNorm after every convolution**. All convolutions are 3×3 with padding 1 and `bias=False` (BatchNorm makes the bias redundant). Each block ends in a 2×2 max-pool.

| Block | Convs | Channels | Pre-pool (skip) map | After pool |
|---|---|---|---|---|
| 1 | 1 | 64 | 64 × 224 × 224 | 64 × 112 × 112 |
| 2 | 1 | 128 | 128 × 112 × 112 | 128 × 56 × 56 |
| 3 | 2 | 256 | 256 × 56 × 56 | 256 × 28 × 28 |
| 4 | 2 | 512 | 512 × 28 × 28 | 512 × 14 × 14 |
| 5 | 2 | 512 | 512 × 14 × 14 | 512 × 7 × 7 |

`forward(x, return_features=True)` returns the bottleneck and a dict of the pre-pool maps (`block1`…`block5`), which the U-Net uses as skip connections. Layer order is **Conv → BatchNorm → ReLU**, so the normalization acts on pre-activations.

### Custom dropout (`models/layers.py`)

`CustomDropout(nn.Module)` is an inverted-dropout layer that does **not** use `torch.nn.Dropout` or `F.dropout`.
- In training it draws a Bernoulli keep-mask with probability `1 - p` and rescales the kept activations by `1 / (1 - p)`.
- In eval (`self.training == False`) it is an identity.
- It raises `ValueError` for `p` outside `[0, 1)`.

### Classification head (`models/classification.py`)

```
encoder → flatten (512·7·7 = 25088) → FC 4096 → ReLU → CustomDropout
                                    → FC 4096 → ReLU → CustomDropout
                                    → FC 37
```

### Localization head (`models/localization.py`)

```
encoder → AdaptiveAvgPool(7×7) → flatten → FC 1024 → ReLU → CustomDropout
                                         → FC 512  → ReLU → CustomDropout
                                         → FC 4 → Sigmoid → [x_c, y_c, w, h] in [0, 1]
```

Outputs are normalized, clamped to `[1e-6, 1 - 1e-6]`, and scaled by image width/height to pixels in the unified model.

**Custom IoU loss** (`losses/iou_loss.py`): converts `[x_c, y_c, w, h]` to corners, computes the intersection with clamped width/height (so disjoint boxes give 0), the union with an epsilon floor, and returns `1 - IoU`. It supports `mean`, `sum` and `none` reductions, and gradients flow through every step.

### Segmentation: U-Net decoder (`models/segmentation.py`)

Contracting path = the VGG11 encoder. The expansive path mirrors it, and **all upsampling uses `ConvTranspose2d` (kernel 2, stride 2)**. At every stage the upsampled map is concatenated with the matching encoder map before two Conv-BN-ReLU layers.

| Stage | Input | Transposed conv | Concat with | Output |
|---|---|---|---|---|
| 5 | 512 × 7 × 7 | → 512 × 14 × 14 | block 5 → 1024 ch | 512 × 14 × 14 |
| 4 | 512 × 14 × 14 | → 512 × 28 × 28 | block 4 → 1024 ch | 512 × 28 × 28 |
| 3 | 512 × 28 × 28 | → 256 × 56 × 56 | block 3 → 512 ch | 256 × 56 × 56 |
| 2 | 256 × 56 × 56 | → 128 × 112 × 112 | block 2 → 256 ch | 128 × 112 × 112 |
| 1 | 128 × 112 × 112 | → 64 × 224 × 224 | block 1 → 128 ch | 64 × 224 × 224 |
| out | 64 × 224 × 224 | 1×1 conv | | 3 × 224 × 224 |

Custom dropout is applied in decoder stages 5 and 4 only, to keep fine spatial detail near the output. Convolutions use He (Kaiming) initialization.

### Unified model (`models/multitask.py`)

```python
out = model(x)
out["classification"]   # [B, 37]            breed logits
out["localization"]     # [B, 4]             (x_c, y_c, w, h) in pixels
out["segmentation"]     # [B, 3, H, W]       logits; 0 = pet, 1 = background, 2 = border
```

Weights are loaded from the three single-task checkpoints by remapping state-dict key prefixes (`classifier.pth` → `encoder` + `cls_fc*`, `localizer.pth` → `loc_encoder` + `localizer_fc*`, `segmenter.pth` → `seg_encoder` + decoder). If the checkpoint files are missing, they are downloaded with `gdown`.

---

## Dataset

[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/): 37 breeds (25 dog, 12 cat), with breed labels, head bounding boxes (XML), and trimaps.

```bash
mkdir -p data && cd data
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf images.tar.gz && tar -xzf annotations.tar.gz
```

Expected layout (the dataset is extracted into `data/`, next to `pets_dataset.py`):

```
data/
├── images/*.jpg
└── annotations/
    ├── list.txt
    ├── trimaps/*.png
    └── xmls/*.xml
```

`OxfordIIITPetDataset` reads `annotations/list.txt` and keeps only images that have an XML box annotation (≈3.7k images). Each item is `(image, label, bbox, mask)`:

| Item | Format |
|---|---|
| `image` | float tensor `[3, H, W]` (after transforms: 224×224, ImageNet-normalized) |
| `label` | long, breed index 0–36 |
| `bbox` | float `[x_min, y_min, w, h]` (COCO format, pixels) |
| `mask` | long `[H, W]`: **0 = pet, 1 = background, 2 = border** (raw trimap values 1/2/3 minus 1) |

Albumentations transforms are supported through the `transform` argument and must handle `image`, `mask` and `bboxes` together. All scripts use an 80/20 random split with seed 42.

---

## Setup & usage

```bash
git clone https://github.com/4prasid/multitask-vision-vgg11.git
cd multitask-vision-vgg11
pip install -r requirements.txt
mkdir -p checkpoints
wandb login                      # training scripts log to Weights & Biases
```

Train the stages in order (each later stage loads the earlier checkpoints from `checkpoints/`):

```bash
python class_train.py            # -> checkpoints/classifier.pth
python loc_train.py              # -> checkpoints/localizer.pth
python seg_train.py              # -> checkpoints/segmenter.pth
python train.py                  # -> checkpoints/multitask_best.pth, multitask_last.pth
python evaluate.py --weights checkpoints/multitask_best.pth
```

Hyperparameters are constants at the top of each script; only `evaluate.py` takes command-line arguments.

Inference with a trained model:

```python
import torch
from models.multitask import MultiTaskPerceptionModel

model = MultiTaskPerceptionModel(load_pretrained=False)
model.load_state_dict(torch.load("checkpoints/multitask_best.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    out = model(images)          # images: [B, 3, 224, 224], ImageNet mean/std normalized
```

### Training configuration

All stages use gradient clipping at 1.0, an 80/20 split (seed 42) and input size 224×224.

| Stage | Init | Loss | Optimizer / schedule | Epochs × batch | Checkpoint chosen by |
|---|---|---|---|---|---|
| `class_train.py` | scratch (dropout 0.6) | CE, label smoothing 0.1 | Adam 5e-5, wd 5e-4, StepLR ×0.5 / 10 ep | 40 × 32 | val macro-F1 |
| `loc_train.py` | encoder ← classifier (dropout 0.5) | `w_iou·(1−IoU) + w_l1·SmoothL1 + 0.05·size-L1`, with (w_iou, w_l1) = (0.2, 1.0) → (0.7, 1.0) at ep 5 → (1.2, 0.5) at ep 15 | Adam 5e-5, wd 5e-4, StepLR ×0.5 / 5 ep. Encoder frozen until ep 10, then block 5 unfrozen and the optimizer re-created for head + block 5 at lr 1e-6, wd 1e-3 | 70 × 32 | 0.6·Acc@0.5 + 0.4·Acc@0.75 |
| `seg_train.py` | encoder ← classifier, fully fine-tuned (dropout 0.5) | 0.3·class-weighted CE + 0.7·class-weighted soft-Dice (3 classes) | Adam 1e-4, wd 1e-5, ReduceLROnPlateau (max Dice, patience 3, ×0.5) | 40 × 16 | val macro Dice |
| `train.py` | all three checkpoints (dropout 0.5 / 0.3 / 0.3) | `1·CE + 5·IoU + 2·CE` (classification, localization, segmentation) | Adam 1e-4, StepLR ×0.1 / 10 ep | 20 × 32 | val macro Dice |

---

## Results

All numbers are on a held-out 20% validation split (random, seed 42) of the annotated Oxford-IIIT images. The same split was used to select checkpoints, and the official test split is not used for any reported number.

| Task | Metric | Result |
|---|---|---|
| Breed classification (37 classes) | Macro F1 | **0.46** (train 0.98) |
| Head localization | Fraction of boxes with IoU ≥ 0.5 | **0.79** |
| Trimap segmentation (pet / background / border) | Macro Dice · pixel accuracy | **0.82** · 0.89 |

---

## How it works

```
                      ┌─ VGG11 encoder ① → FC head ───────────→ breed logits        [B, 37]
 image [B,3,224,224] ─┼─ VGG11 encoder ② → regression head ───→ box (x_c,y_c,w,h)   [B, 4]
                      └─ VGG11 encoder ③ → U-Net decoder ─────→ trimap logits       [B, 3, 224, 224]
```

The model is built in four stages. Each stage starts from the previous stage's weights:

1. **Classification**: VGG11 + BatchNorm + custom dropout, trained from scratch on 37 breeds.
2. **Localization**: the classifier's encoder plus a new regression head. The encoder is frozen for 10 epochs, then block 5 is unfrozen at a much smaller learning rate.
3. **Segmentation**: the classifier's encoder (fully fine-tuned) plus a symmetric U-Net decoder with transposed-convolution upsampling and skip connections.
4. **Unified model**: `MultiTaskPerceptionModel` loads the three trained checkpoints and fine-tunes them jointly with a weighted multi-task loss. A single `forward(x)` returns all three outputs.

The unified model holds three task-specific encoders rather than one shared backbone (see [Design decisions](#design-decisions)).

---


## Experiments & findings

Every experiment is logged to Weights & Biases. The detailed write-ups are in **[FINDINGS.md](FINDINGS.md)**, and the interactive plots are in the **[W&B report](https://forge.coreweave.com/wandb/prasid-indian-institute-of-technology-madras/da6401_assignment2/reports/DA6401-Assignment-2-PH21B007-PRASID--VmlldzoxNjQ4OTM5Mw)**.

| Topic | Headline finding |
|---|---|
| BatchNorm and trainability | At LR 5e-5, VGG11 without BatchNorm never trained (F1 ≈ 0) while the BatchNorm run reached val F1 ≈ 0.4 in 20 epochs. Without BN the activation std collapsed toward zero. |
| Dropout and the generalization gap | With dropout 0 / 0.2 / 0.5 all runs ended at val loss ≈ 2.5. Dropout 0.5 slowed the decline of training loss but did not clearly improve final validation loss in 25 epochs. |
| Transfer-learning strategies (segmentation) | Full fine-tuning (Dice 0.82) > partial fine-tuning of blocks 4–5 (0.79) > frozen backbone (0.73). |
| Feature maps | Block 1 keeps edges, silhouettes and texture. Block 5 maps are sparse and abstract. |
| Localization: confidence vs IoU | On 15 sampled images, mean IoU was 0.70 and 13 of 15 boxes had IoU ≥ 0.5. Both failures came with high classifier confidence, which shows that breed confidence says little about box quality. |
| Dice vs pixel accuracy | Pixel accuracy (0.89) sits well above macro Dice (0.82) because the majority background class dominates the pixel count. |
| In-the-wild images | Segmentation held up best. Breed predictions were wrong on all three images (one dog is a breed outside the 37 classes, and both cats were labeled as dog breeds). The box missed the subject on a cluttered outdoor scene. |
| Multi-task retrospective | Localization and segmentation show near-zero train/val gaps. Classification is the only overfitting source. |

---

## Design decisions

- **BatchNorm before ReLU, `bias=False` convs.** Normalizing pre-activations stabilized training. The BatchNorm ablation shows the difference between a model that learns and one that doesn't.
- **Dropout placement.** Dropout sits after the FC layers of the classification and localization heads, where most of the parameters are, and in the two deepest decoder stages. The convolutional backbone and the shallow decoder stages are left undropped to protect spatial detail.
- **Localization loss.** IoU loss is scale-invariant and optimizes the evaluation metric directly, but it gives no gradient for non-overlapping boxes. It is therefore mixed with SmoothL1 and a size term, ramping the IoU weight up and the L1 weight down over training.
- **Segmentation loss.** Cross-entropy provides stable per-pixel gradients, and soft Dice optimizes region overlap and is less dominated by the majority class. Dice is computed over all three classes, so the border class is supervised too.
- **Transposed-convolution upsampling.** The decoder learns its own upsampling kernels instead of using fixed interpolation.
- **Three encoders in the unified model.** Each task's encoder starts from its own specialized checkpoint, which avoids gradient interference between tasks by construction and keeps each task's accuracy. The cost is roughly 3× the encoder parameters and no feature sharing between tasks.

---

## Known limitations & next steps

- **Augmentation is defined but not effectively applied.** The training scripts assign `train_transform` and `val_transform` to the same underlying dataset object, so the second assignment wins and training runs with resize + normalize only. This is the most likely reason for the large classification train/validation gap. The fix is to build two dataset instances (one per transform) and index both with the same split indices.
- **Classification overfits.** Validation macro-F1 is 0.46 against 0.98 on train. The 25088→4096→4096 head is very large for ≈3k training images. Candidate fixes are real augmentation, a smaller head or global pooling, and stronger regularization.
- **Segmentation class weights were not tuned.** The weights `[0.2, 0.5, 0.3]` are indexed by `[pet, background, border]`, so background is weighted highest.
- **No shared backbone.** A true shared encoder with light task adapters would cut parameters and test whether the tasks help or hurt each other.
- **Validation, not test.** Reported numbers come from a 20% split that also drove checkpoint selection.
- **No CLI configuration yet.** Hyperparameters live as constants at the top of each training script.

---

## Background

Built using concepts taught in the course *DA6401: Introduction to Deep Learning* (IIT Madras).

Part of a deep learning project series:
[MLP from Scratch](https://github.com/4prasid/MLP-from-scratch) · [Multi-task Vision VGG11](https://github.com/4prasid/multitask-vision-vgg11) · Transformer NMT

## License

[MIT](LICENSE)
