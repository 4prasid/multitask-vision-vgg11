# VGG11 Multi-Task Vision Pipeline

A from-scratch **VGG11** pipeline in PyTorch that, in a single forward pass, predicts a pet's **breed** (37 classes), its **head bounding box**, and a pixel-wise **trimap segmentation**. Built on the Oxford-IIIT Pet dataset with a custom dropout layer, a custom IoU loss, and a U-Net decoder that upsamples with transposed convolutions.

**📊 [Interactive W&B report](https://api.wandb.ai/links/prasid-indian-institute-of-technology-madras/u2yoef01)**

<p align="center">
  <img src="https://github.com/user-attachments/assets/4045216d-a9af-4b02-a42f-621effa09c66" alt="Pipeline output on an unseen image" width="48%" />
  <img src="https://github.com/user-attachments/assets/908f5f09-0173-4b2d-8aad-dde329cff009" alt="Pipeline output on an unseen image" width="48%" />
  <br />
  <em>Pipeline output on unseen images</em>
</p>

---

## Results

All numbers are on a held-out 20% validation split (random, seed 42) of the annotated Oxford-IIIT images. The same split was used to select checkpoints, and the official test split is not used for any reported number.

| Task | Metric | Result |
|---|---|---|
| Breed classification (37 classes) | Macro F1 | **0.46** (train 0.98) |
| Head localization | Fraction of boxes with IoU ≥ 0.5 | **0.79** |
| Trimap segmentation (pet / background / border) | Macro Dice · pixel accuracy | **0.82** · 0.89 |

Classification and localization come from the unified model (`multitasking_8.py`, epoch 20). Segmentation comes from the full fine-tuning run of `transfer_learning_3_4_6.py` (plain cross-entropy, 20 epochs), with Dice computed per image as a macro average over the three classes and then averaged.

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

## Repository structure

```
multitask-vision-vgg11/
├── data/
│   └── pets_dataset.py              # Oxford-IIIT Pet loader -> (image, label, bbox, mask)
├── losses/
│   └── iou_loss.py                  # Custom IoU loss (nn.Module)
├── models/
│   ├── vgg11.py                     # VGG11 encoder (BatchNorm, skip-feature output)
│   ├── layers.py                    # CustomDropout (inverted dropout, nn.Module)
│   ├── classification.py            # VGG11Classifier
│   ├── localization.py              # VGG11Localizer
│   ├── segmentation.py              # VGG11UNet
│   └── multitask.py                 # MultiTaskPerceptionModel (unified)
├── Experiments/
│   ├── multitasking_8.py                # joint multi-task training
│   ├── inference_7.py                   # Run the pipeline on your own images
│   ├── batchnorm_1.py                   # BatchNorm ablation
│   ├── dropout_2.py                     # dropout sweep
│   ├── transfer_learning_3_4_6.py       # transfer learning, feature maps, Dice vs pixel accuracy
│   └── object_detection_5.py            # detection IoU gallery
├── class_train.py                   # Stage 1: classification
├── loc_train.py                     # Stage 2: localization
├── seg_train.py                     # Stage 3: segmentation (CE + Dice)
├── requirements.txt
├── FINDINGS.md                      # Full written analysis of the experiments
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

Custom dropout is applied in decoder stages 5 and 4 only, to keep fine spatial detail near the output.

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
wandb login                      # scripts log to Weights & Biases
```

Train the stages in order (each later stage loads the earlier checkpoints from `checkpoints/`):

```bash
python class_train.py            # -> checkpoints/classifier.pth
python loc_train.py              # -> checkpoints/localizer.pth
python seg_train.py              # -> checkpoints/segmenter.pth
python multitasking_8.py         # -> checkpoints/multitask_best.pth, multitask_final.pth
```

Hyperparameters are constants at the top of each training script; the training scripts take no command-line flags.

### Experiment scripts

Each script reproduces one part of the [W&B report](https://forge.coreweave.com/wandb/prasid-indian-institute-of-technology-madras/da6401_assignment2/reports/VGG11-Multi-Task-Vision--VmlldzoxNjQ4OTM5Mw) and logs to the `da6401_assignment2` project.

| Script | What it runs |
|---|---|
| `batchnorm_1.py` | VGG11 with vs without BatchNorm; activation histograms of the 2nd conv layer |
| `dropout_2.py` | Dropout p = 0 / 0.2 / 0.5 on the classifier; train/val loss and F1 gap |
| `transfer_learning_3_4_6.py` | Frozen / partial / full fine-tuning for segmentation; feature-map figures; Dice vs pixel accuracy and sample masks (needs `checkpoints/classifier.pth`) |
| `object_detection_5.py` | Predicted vs ground-truth boxes with IoU and confidence on 15 random images (needs `checkpoints/localizer.pth`) |

### Inference on your own images

```bash
python inference_7.py --images dog.jpg cat1.jpg cat2.jpg \
                      --names "Dog" "Cat 1" "Cat 2" --output_dir wild_outputs
```

The script takes exactly three images and builds `MultiTaskPerceptionModel` from the three single-task checkpoints (downloaded automatically if missing). It saves a figure per image (box, trimap, overlay, top-3 breeds) and logs them to W&B. It does not load the jointly fine-tuned weights.

### Training configuration

All stages use gradient clipping at 1.0, an 80/20 split (seed 42) and input size 224×224.

| Stage | Init | Loss | Optimizer / schedule | Epochs × batch | Checkpoint chosen by |
|---|---|---|---|---|---|
| `class_train.py` | scratch (dropout 0.6) | CE, label smoothing 0.1 | Adam 5e-5, wd 5e-4, StepLR ×0.5 / 10 ep | 40 × 32 | val macro-F1 |
| `loc_train.py` | encoder ← classifier (dropout 0.5) | `w_iou·(1−IoU) + w_l1·SmoothL1 + 0.05·size-L1`, with (w_iou, w_l1) = (0.2, 1.0) → (0.7, 1.0) at ep 5 → (1.2, 0.5) at ep 15 | Adam 5e-5, wd 5e-4, StepLR ×0.5 / 5 ep. Encoder frozen until ep 10, then block 5 unfrozen and the optimizer re-created for head + block 5 at lr 1e-6, wd 1e-3 | 70 × 32 | 0.6·Acc@0.5 + 0.4·Acc@0.75 |
| `seg_train.py` | encoder ← classifier, fully fine-tuned (dropout 0.5) | 0.3·class-weighted CE + 0.7·class-weighted soft-Dice (3 classes) | Adam 1e-4, wd 1e-5, ReduceLROnPlateau (max Dice, patience 3, ×0.5) | 40 × 16 | val macro Dice |
| `multitasking_8.py` | all three checkpoints (dropout 0.5 / 0.3 / 0.3) | `1·CE + 5·IoU + 2·CE` (classification, localization, segmentation) | Adam 1e-4, wd 1e-4, StepLR ×0.1 / 10 ep | 20 × 8 | val Dice¹ |
| `transfer_learning_3_4_6.py` | encoder ← classifier; frozen / blocks 4–5 / full | plain CE | Adam 1e-4, wd 1e-4, StepLR ×0.5 / 8 ep | 20 × 16 | final epoch |

¹ Class 1 vs rest with border pixels ignored. Class 1 is the background, so this score is not a pet-segmentation Dice.

---

## Design decisions

- **BatchNorm before ReLU, `bias=False` convs.** Normalizing pre-activations stabilized training. The BatchNorm ablation shows the difference between a model that learns and one that doesn't.
- **Dropout placement.** Dropout sits after the FC layers of the classification and localization heads, where most of the parameters are, and in the two deepest decoder stages. The convolutional backbone and the shallow decoder stages are left undropped to protect spatial detail.
- **Localization loss.** IoU loss is scale-invariant and optimizes the evaluation metric directly, but it gives no gradient for non-overlapping boxes. It is therefore mixed with SmoothL1 and a size term, ramping the IoU weight up and the L1 weight down over training.
- **Segmentation loss.** `seg_train.py` (the checkpoint that initializes the unified model) combines class-weighted cross-entropy with a soft Dice loss over all three classes, so the border class is supervised too. The transfer-learning and joint-training experiments use plain cross-entropy for simplicity.
- **Transposed-convolution upsampling.** The decoder learns its own upsampling kernels instead of using fixed interpolation.
- **Three encoders in the unified model.** Each task's encoder starts from its own specialized checkpoint, which avoids gradient interference between tasks by construction and keeps each task's accuracy. The cost is roughly 3× the encoder parameters and no feature sharing between tasks.

---

## Experiments & findings

Every experiment is logged to Weights & Biases. The detailed write-ups, including how each metric is defined and the caveats of each run, are in **[FINDINGS.md](FINDINGS.md)**, and the interactive plots are in the **[W&B report](https://forge.coreweave.com/wandb/prasid-indian-institute-of-technology-madras/da6401_assignment2/reports/VGG11-Multi-Task-Vision--VmlldzoxNjQ4OTM5Mw)**.

| Topic | key finding |
|---|---|
| BatchNorm and trainability | At LR 5e-5 with PyTorch default initialization, VGG11 without BatchNorm never trained (F1 ≈ 0), while the BatchNorm run reached val F1 ≈ 0.4 in 20 epochs. Without BN the std of the 2nd conv layer's output collapsed toward zero. |
| Dropout and the generalization gap | With dropout 0 / 0.2 / 0.5 all runs ended at val loss ≈ 2.5 after 25 epochs. Dropout 0.5 slowed the decline of training loss but did not clearly improve final validation loss. |
| Transfer-learning strategies (segmentation) | Full fine-tuning (Dice 0.82) > partial fine-tuning of blocks 4–5 (0.79) > frozen backbone (0.73). |
| Feature maps | Block 1 keeps edges, silhouettes and texture. Block 5 maps are sparse and abstract. |
| Localization quality | On 15 randomly sampled annotated images (not restricted to the validation split), mean IoU was 0.70 and 13 of 15 boxes had IoU ≥ 0.5. The two failures (IoU 0.36 and 0.25) came with high reported confidence (0.94 and 0.76), so confidence did not flag the poor boxes. Both are boxes that extend beyond the head. |
| Dice vs pixel accuracy | Pixel accuracy (0.89) sits well above macro Dice (0.82) because the majority background class dominates the pixel count. |
| In-the-wild images | Run with the unified architecture loaded from the three single-task checkpoints (no joint fine-tuning). Segmentation held up best. Breed predictions were wrong on all three images, and the box missed the subject on a cluttered outdoor scene. |
| Multi-task retrospective | Localization and segmentation show near-zero train/val gaps. Classification is the only overfitting source. |

---

## Reproducing the experiments

The experiment scripts import from the repository root, so run them from there with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python Experiments/batchnorm_1.py                 # BatchNorm ablation
PYTHONPATH=. python Experiments/dropout_2.py                   # dropout sweep
PYTHONPATH=. python Experiments/transfer_learning_3_4_6.py     # transfer learning, feature maps, Dice vs pixel accuracy
PYTHONPATH=. python Experiments/object_detection_5.py          # detection IoU gallery
PYTHONPATH=. python Experiments/inference_7.py                 # Run the pipeline on your own images
PYTHONPATH=. python Experiments/multitasking_8.py              # joint multi-task training
```

---

## Background

Built using concepts taught in the course *DA6401: Introduction to Deep Learning* (IIT Madras).

Part of a deep learning project series:
[MLP from Scratch](https://github.com/4prasid/MLP-from-scratch) · [Multi-task Vision](https://github.com/4prasid/multitask-vision-vgg11) · [Transformer NMT](https://github.com/4prasid/transformer-nmt-from-scratch)

---

## License

[MIT](LICENSE)
