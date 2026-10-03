# DA6401 Assignment 2 - Building a Complete Visual Perception Pipeline

**Student:** Prasid | **Roll No:** PH21B007

🔗 [W&B Report](https://api.wandb.ai/links/prasid-indian-institute-of-technology-madras/u2yoef01) &nbsp;|&nbsp; 🔗 [GitHub Repository](https://github.com/Prasid7/da6401_assignment_2_ph21b007_prasid)

---

## Overview

This project implements a complete **multi-task visual perception pipeline** built on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). Starting from a VGG11 encoder built from scratch, the pipeline is progressively extended to handle three vision tasks simultaneously:

1. **Classification** - Predict the pet breed (37 classes)
2. **Localization** - Predict the head bounding box `[x_center, y_center, width, height]`
3. **Segmentation** - Produce a pixel-wise trimap mask (foreground / background / border)

All tasks share the same VGG11 convolutional backbone and are unified into a single `MultiTaskPerceptionModel` that processes all three outputs in one forward pass.

---

## Repository Structure

```
Assignment_2/
├── data/
│   └── pets_dataset.py          # Oxford-IIIT Pet multi-task dataset loader
├── losses/
│   ├── __init__.py
│   └── iou_loss.py              # Custom IoU loss (nn.Module)
├── models/
│   ├── __init__.py
│   ├── vgg11.py                 # VGG11 encoder with skip-connection support
│   ├── layers.py                # Custom Dropout layer (nn.Module)
│   ├── classification.py        # VGG11Classifier (encoder + FC head)
│   ├── localization.py          # VGG11Localizer (encoder + regression head)
│   ├── segmentation.py          # VGG11UNet (encoder + U-Net decoder)
│   └── multitask.py             # MultiTaskPerceptionModel (unified pipeline)
├── checkpoints/                 # Saved model weights (generated during training)
├── train.py                     # Training entry point
├── inference.py                 # Inference and evaluation entry point
├── requirements.txt
└── README.md
```

---

## Dataset

**Oxford-IIIT Pet Dataset** : [Download here](https://www.robots.ox.ac.uk/~vgg/data/pets/)

The dataset provides:
- **Images** - JPEG photographs of 37 pet breeds
- **Class labels** - Breed index (1–37)
- **Bounding boxes** - XML annotations marking the head region
- **Trimaps** - Pixel-level masks with 3 classes: `1 = foreground`, `2 = background`, `3 = border`

### Expected Directory Layout

After downloading and extracting, the dataset should be organized as:

```
<root_dir>/
├── images/
│   ├── Abyssinian_1.jpg
│   └── ...
└── annotations/
    ├── list.txt            # trainval split
    ├── test.txt            # test split
    ├── trimaps/
    │   ├── Abyssinian_1.png
    │   └── ...
    └── xmls/
        ├── Abyssinian_1.xml
        └── ...
```

> **Note:** Only images that have a corresponding XML bounding-box annotation are included. Samples missing an XML file are automatically excluded by the dataset loader.

---

## Architecture

### VGG11 Encoder (`models/vgg11.py`)

The backbone follows the standard VGG11 topology with **Batch Normalization** added after every convolutional layer. All five convolutional blocks use `3×3` filters with `padding=1` to preserve spatial dimensions before each `2×2` max-pool.

| Block | Filters | Output Shape (224×224 input) |
|---|---|---|
| Block 1 | 64 | `(B, 64, 112, 112)` after pool |
| Block 2 | 128 | `(B, 128, 56, 56)` after pool |
| Block 3 | 256 × 2 | `(B, 256, 28, 28)` after pool |
| Block 4 | 512 × 2 | `(B, 512, 14, 14)` after pool |
| Block 5 | 512 × 2 | `(B, 512, 7, 7)` after pool |

The encoder's `forward()` method accepts a `return_features` flag. When `True`, it returns both the bottleneck tensor and a dictionary of pre-pool feature maps (`block1` through `block5`) used as skip connections by the U-Net decoder.

**Design Choices:**
- `bias=False` on all `Conv2d` layers — Batch Normalization makes the bias redundant, saving parameters.
- BatchNorm placed immediately after `Conv2d` and before `ReLU` — normalizes activations before the non-linearity to stabilize gradients and allow higher stable learning rates.

---

### Custom Dropout (`models/layers.py`)

```python
class CustomDropout(nn.Module):
```

A fully custom inverted-dropout implementation **without** using `torch.nn.Dropout`:

- During training: draws a Bernoulli mask with keep-probability `(1-p)`, applies it element-wise, and scales the output by `1/(1-p)` so that the expected activation magnitude is preserved at test time.
- During evaluation (`self.training == False`): passes the input through unchanged.
- Raises `ValueError` if `p` is outside `[0, 1)` — `p=1.0` is disallowed to prevent division by zero in the scaling step.

---

### Task 1 - Classification (`models/classification.py`)

```
VGG11Encoder → Flatten → FC(25088→4096) + ReLU + Dropout
                       → FC(4096→4096)  + ReLU + Dropout
                       → FC(4096→37)    [logits]
```

`CustomDropout` is placed after each of the first two fully-connected layers. This targets the densest part of the network where overfitting is most likely, while leaving the convolutional feature extractor intact.

**Key parameters:** `num_classes=37`, `dropout_p=0.5`

---

### Task 2 - Localization (`models/localization.py`)

```
VGG11Encoder → AdaptiveAvgPool(7×7) → Flatten
             → FC(25088→1024) + ReLU + Dropout
             → FC(1024→512)   + ReLU + Dropout
             → FC(512→4)      + Sigmoid
             → [x_center, y_center, width, height]
```

- The `Sigmoid` output keeps predicted coordinates in `[0, 1]` (normalized).
- Coordinates are clamped to `[ε, 1-ε]` to avoid degenerate boxes with zero area.
- The forward method converts to absolute pixel coordinates by multiplying against the input image's `H` and `W`.
- **Encoder strategy** (frozen vs fine-tuned): controlled externally in `train.py` via `requires_grad` flags on `model.encoder` — not hardcoded in the model class.

**Loss:** Custom `IoULoss` (see below). `dropout_p=0.3` (lighter regularization than classification since the regression head is shallower).

---

### Custom IoU Loss (`losses/iou_loss.py`)

```python
class IoULoss(nn.Module):
```

Mathematically correct IoU loss for bounding box regression:

1. Converts input `[x_c, y_c, w, h]` → `[x_min, y_min, x_max, y_max]`
2. Computes intersection area using clamped width/height (`min=0`) to handle non-overlapping boxes gracefully
3. Computes union area as `pred_area + target_area - inter_area`, clamped to `ε` to prevent division by zero
4. Returns `loss = 1 - IoU`

Supports `reduction` modes: `"mean"` (default), `"sum"`, or `"none"`. Gradient flow is preserved through all operations.

---

### Task 3 - Segmentation (`models/segmentation.py`)

A **U-Net style** network using the VGG11 encoder as the contracting path and a symmetric decoder as the expansive path.

**Decoder (expansive path):**

| Decoder Block | Input | Transposed Conv | After skip concat | Output |
|---|---|---|---|---|
| Block 5 | `(B, 512, 7, 7)` | `up5` → `(B, 512, 14, 14)` | concat `block5` → `(B, 1024, 14, 14)` | `(B, 512, 14, 14)` |
| Block 4 | `(B, 512, 14, 14)` | `up4` → `(B, 512, 28, 28)` | concat `block4` → `(B, 1024, 28, 28)` | `(B, 512, 28, 28)` |
| Block 3 | `(B, 512, 28, 28)` | `up3` → `(B, 256, 56, 56)` | concat `block3` → `(B, 512, 56, 56)` | `(B, 256, 56, 56)` |
| Block 2 | `(B, 256, 56, 56)` | `up2` → `(B, 128, 112, 112)` | concat `block2` → `(B, 256, 112, 112)` | `(B, 128, 112, 112)` |
| Block 1 | `(B, 128, 112, 112)` | `up1` → `(B, 64, 224, 224)` | concat `block1` → `(B, 128, 224, 224)` | `(B, 64, 224, 224)` |
| Final | `(B, 64, 224, 224)` | `Conv2d(64, 3, 1×1)` | — | `(B, 3, H, W)` |

- **Upsampling:** All upsampling is done with `ConvTranspose2d` (learnable, `kernel_size=2, stride=2`). No bilinear interpolation is used for the primary upsampling steps.
- **Skip connections:** Pre-pool feature maps from each encoder block are concatenated channel-wise with the decoder's upsampled output at the matching spatial resolution.
- **Dropout** is applied only in the deeper decoder blocks (5 and 4) where feature maps are densest. Shallower blocks closer to the output are left without dropout to preserve fine spatial detail.
- **Loss:** Cross-Entropy Loss over 3 classes (`background=0`, `foreground=1`, `border=2`). Cross-entropy is well-suited to multi-class pixel classification and naturally handles class imbalance better than pixel-accuracy-based objectives alone.
- Weights initialized with `kaiming_normal_` (He initialization) for convolutional layers.

---

### Task 4 - Unified Multi-Task Pipeline (`models/multitask.py`)

```python
class MultiTaskPerceptionModel(nn.Module):
```

The unified model hosts **three independent VGG11 encoders**, each loaded from its own pre-trained single-task checkpoint:

| Encoder | Source Checkpoint | Paired Head |
|---|---|---|
| `self.encoder` | `classifier.pth` | Classification FC layers |
| `self.loc_encoder` | `localizer.pth` | Localization FC layers |
| `self.seg_encoder` | `segmenter.pth` | U-Net decoder |

A single `forward(x)` call runs all three branches in parallel and returns a dictionary:

```python
{
    'classification': torch.Tensor,   # [B, 37]   breed logits
    'localization':   torch.Tensor,   # [B, 4]    bbox in pixel space
    'segmentation':   torch.Tensor,   # [B, 3, H, W] trimap logits
}
```

**Weight loading** (`_load_weights`): Remaps key prefixes from each single-task state dict (e.g., `encoder.*`, `fc1.*`) to the unified model's namespaced attributes. Uses `strict=False` to allow partial loading — existing weights not found in the checkpoint are silently retained. Pre-trained checkpoints are automatically downloaded via `gdown` if missing or incomplete.

---

## Dataset Loader (`data/pets_dataset.py`)

`OxfordIIITPetDataset` supports two splits: `'trainval'` and `'test'`.

Each item returns a 4-tuple:
- `image`: `FloatTensor [3, H, W]`
- `label`: `LongTensor []` — zero-indexed class (0–36)
- `bbox`: `FloatTensor [4]` — `[x_center, y_center, width, height]` in unnormalized pixel coordinates
- `mask`: `LongTensor [H, W]` — values `0` (background), `1` (foreground), `2` (border)

Masks are remapped from the raw trimap values `{1, 2, 3}` → `{0, 1, 2}`.

The loader is fully compatible with [Albumentations](https://albumentations.ai/) transforms via the `transform` argument, which must handle `image`, `mask`, and `bboxes` keys simultaneously.

---

## Training

```bash
python train.py
```

Edit `train.py` to configure:
- Dataset root path
- Task to train (`classification`, `localization`, `segmentation`)
- Encoder freezing strategy (freeze/unfreeze `model.encoder` parameters)
- Optimizer, scheduler, number of epochs
- W&B project and run name

---

## Inference

```bash
python inference.py
```

Edit `inference.py` to specify the checkpoint path and the path to input images.

---

## Experiment Tracking

All training runs are logged to [Weights & Biases](https://wandb.ai/). The public W&B report documents:

- Activation distributions with/without Batch Normalization
- Training vs. validation loss curves under three dropout settings (`p=0`, `0.2`, `0.5`)
- Transfer learning comparison: frozen backbone vs. partial fine-tuning vs. full fine-tuning
- Feature map visualizations from the first and last convolutional layers
- Bounding box prediction overlays with IoU scores on test images
- Segmentation outputs: original image, ground-truth trimap, predicted trimap
- Final pipeline results on novel "in-the-wild" pet images

🔗 [View the full W&B Report](https://api.wandb.ai/links/prasid-indian-institute-of-technology-madras/u2yoef01)

---

## Evaluation Metrics

| Task | Metric |
|---|---|
| Classification | Macro F1-Score (37 classes) |
| Localization | Mean Average Precision (mAP) |
| Segmentation | Dice Similarity Coefficient |

---

## Key Design Decisions

**BatchNorm before ReLU:** Normalizing pre-activations stabilizes training, acts as an implicit regularizer, and allows the use of larger learning rates without divergence.

**Custom Dropout placement:** Applied after FC layers in classification/localization heads (where overfitting is concentrated) and in the two deepest decoder blocks in the U-Net. Shallower layers and the convolutional backbone are left undropped to preserve spatial features and gradient signal quality.

**IoU Loss for localization:** Unlike MSE, IoU loss is scale-invariant and directly optimizes the detection metric, making it more robust to variation in box sizes across the dataset.

**Transposed Convolutions for upsampling:** Learnable upsampling allows the decoder to learn task-specific upsampling kernels rather than relying on fixed interpolation, giving the network more expressive power in the expansive path.

**Separate encoders in MultiTaskModel:** Rather than a hard-shared backbone (which can suffer from gradient interference between tasks), each task's encoder is initialized from its own specialized checkpoint. This avoids task competition while still enabling a single unified inference forward pass.
