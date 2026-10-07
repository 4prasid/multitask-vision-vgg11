"""
7. Wild Image Inference & W&B Logging

Runs the full MultiTaskPerceptionModel pipeline on 3 novel pet images
and logs results to W&B.

"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.multitask import MultiTaskPerceptionModel

# ── Config ────────────────────────────────────────────────────────────────────
WANDB_KEY  = "" # <-- insert W&B API key here
PROJECT    = "da6401_assignment2"
RUN_NAME   = "q2_7_wild_inference"

CLASSIFIER_CKPT = "checkpoints/classifier.pth"
LOCALIZER_CKPT  = "checkpoints/localizer.pth"
UNET_CKPT       = "checkpoints/segmenter.pth"

# Oxford-IIIT Pet 37 breed names (index matches dataset label)
BREED_NAMES = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier",
    "Basset Hound", "Beagle", "Bengal", "Birman", "Bombay",
    "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau",
    "English Cocker Spaniel", "English Setter", "German Shorthaired",
    "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond",
    "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue",
    "Saint Bernard", "Samoyed", "Scottish Terrier", "Shiba Inu",
    "Siamese", "Sphynx", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier"
]

# Segmentation color map: 0=pet(black), 1=background(red), 2=border(green)
SEG_COLORS = np.array([
    [0,   0,   0  ],   # class 0 — pet foreground
    [220, 50,  50 ],   # class 1 — background
    [50,  220, 50 ],   # class 2 — border
], dtype=np.uint8)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── Transform ─────────────────────────────────────────────────────────────────
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])


def denormalize(tensor):
    """Convert normalized tensor back to uint8 numpy image."""
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img  = tensor.cpu().numpy().astype(np.float32)
    img  = img * std + mean
    img  = np.clip(img, 0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


def run_inference(model, image_path):
    """Load image, run pipeline, return all predictions."""
    img_pil = Image.open(image_path).convert("RGB")
    img_np  = np.array(img_pil)

    transformed = transform(image=img_np)
    img_t = transformed["image"]           # [3, 224, 224]
    inp   = img_t.unsqueeze(0)             # [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(inp)

    # Classification
    cls_logits = outputs["classification"]          # [1, 37]
    probs      = F.softmax(cls_logits, dim=1)[0]
    pred_class = int(probs.argmax().item())
    confidence = float(probs.max().item())
    breed_name = BREED_NAMES[pred_class] if pred_class < len(BREED_NAMES) else f"Class {pred_class}"

    # Top-3 predictions
    top3_vals, top3_idx = probs.topk(3)
    top3 = [(BREED_NAMES[i] if i < len(BREED_NAMES) else f"Class {i}",
             float(v)) for i, v in zip(top3_idx.tolist(), top3_vals.tolist())]

    # Localization — already in pixel space (224x224) from model forward()
    bbox = outputs["localization"][0].cpu().numpy()   # [xc, yc, w, h] pixels

    # Segmentation
    seg_logits = outputs["segmentation"][0]           # [3, 224, 224]
    seg_mask   = seg_logits.argmax(dim=0).cpu().numpy()  # [224, 224]

    return {
        "img_t":      img_t,
        "img_np":     img_np,
        "pred_class": pred_class,
        "breed_name": breed_name,
        "confidence": confidence,
        "top3":       top3,
        "bbox":       bbox,
        "seg_mask":   seg_mask,
    }


def make_figure(result, title, save_path):
    """
    Create a 3-panel figure:
      Left:   Original image (resized to 224) with predicted bbox
      Middle: Predicted segmentation mask (coloured)
      Right:  Segmentation overlay on original
    """
    img_vis  = denormalize(result["img_t"])       # [224, 224, 3] uint8
    seg_mask = result["seg_mask"]                  # [224, 224]
    seg_rgb  = SEG_COLORS[seg_mask]                # [224, 224, 3]

    # Overlay: blend seg on original
    overlay = (0.55 * img_vis + 0.45 * seg_rgb).astype(np.uint8)

    xc, yc, bw, bh = [float(v) for v in result["bbox"]]
    x1, y1 = xc - bw / 2, yc - bh / 2

    breed = result["breed_name"]
    conf  = result["confidence"]
    top3  = result["top3"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    # Panel 1: Image + BBox
    axes[0].imshow(img_vis)
    rect = patches.Rectangle(
        (x1, y1), bw, bh,
        linewidth=2.5, edgecolor='#FF3333', facecolor='none'
    )
    axes[0].add_patch(rect)
    axes[0].set_title(
        f"Pred: {breed}\nConf: {conf:.3f}\n"
        f"Top3: {top3[1][0]} ({top3[1][1]:.2f}), {top3[2][0]} ({top3[2][1]:.2f})",
        fontsize=8
    )
    axes[0].axis('off')

    # Panel 2: Segmentation mask
    axes[1].imshow(seg_rgb)
    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='black',          label='Pet (foreground)'),
        Patch(facecolor='#DC3232',        label='Background'),
        Patch(facecolor='#32DC32',        label='Border'),
    ]
    axes[1].legend(handles=legend, loc='lower left', fontsize=7,
                   framealpha=0.8)
    axes[1].set_title("Predicted Segmentation Mask", fontsize=9)
    axes[1].axis('off')

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Seg Overlay on Original", fontsize=9)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True,
                        help="Paths to 3 wild pet images")
    parser.add_argument("--names",  nargs="+", default=None,
                        help="Descriptive names for each image (optional)")
    parser.add_argument("--output_dir", default="wild_outputs",
                        help="Directory to save output figures")
    args = parser.parse_args()

    assert len(args.images) == 3, "Provide exactly 3 images."
    names = args.names if args.names and len(args.names) == 3 else \
            [f"Wild Sample {i+1}" for i in range(3)]

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading MultiTaskPerceptionModel...")
    model = MultiTaskPerceptionModel(
        num_breeds=37,
        seg_classes=3,
        in_channels=3,
        classifier_path=CLASSIFIER_CKPT,
        localizer_path=LOCALIZER_CKPT,
        unet_path=UNET_CKPT,
    )
    model.eval()
    print("Model loaded.\n")

    # Run inference on all 3 images
    all_results = []
    all_figures = []

    for i, (img_path, name) in enumerate(zip(args.images, names)):
        print(f"[{i+1}/3] Processing: {img_path}  ({name})")
        result = run_inference(model, img_path)
        print(f"  Breed: {result['breed_name']}  Conf: {result['confidence']:.3f}")
        print(f"  BBox:  xc={result['bbox'][0]:.1f} yc={result['bbox'][1]:.1f} "
              f"w={result['bbox'][2]:.1f} h={result['bbox'][3]:.1f}")
        print(f"  Top3:  {result['top3']}")

        fig_path = os.path.join(args.output_dir, f"wild_sample_{i+1}.png")
        make_figure(result, name, fig_path)

        all_results.append(result)
        all_figures.append((name, fig_path, result))

    # Log to W&B
    print("\nLogging to W&B...")
    wandb.login(key=WANDB_KEY)
    run = wandb.init(
        project=PROJECT,
        name=RUN_NAME,
        group="report_extras",
        reinit="allow",
        config={"num_wild_images": 3}
    )

    # Log individual panels
    wandb_images = []
    for name, fig_path, result in all_figures:
        caption = (
            f"{name} | "
            f"Pred: {result['breed_name']} (conf={result['confidence']:.3f}) | "
            f"BBox: [{result['bbox'][0]:.1f}, {result['bbox'][1]:.1f}, "
            f"{result['bbox'][2]:.1f}, {result['bbox'][3]:.1f}]"
        )
        wandb_images.append(wandb.Image(fig_path, caption=caption))

    # Summary table
    table = wandb.Table(columns=[
        "image", "sample_name", "pred_breed", "confidence",
        "top2_breed", "top2_conf", "top3_breed", "top3_conf",
        "bbox_xc", "bbox_yc", "bbox_w", "bbox_h"
    ])
    for name, fig_path, result in all_figures:
        table.add_data(
            wandb.Image(fig_path),
            name,
            result["breed_name"],
            round(result["confidence"], 3),
            result["top3"][1][0], round(result["top3"][1][1], 3),
            result["top3"][2][0], round(result["top3"][2][1], 3),
            round(float(result["bbox"][0]), 1),
            round(float(result["bbox"][1]), 1),
            round(float(result["bbox"][2]), 1),
            round(float(result["bbox"][3]), 1),
        )

    wandb.log({
        "q2_7_wild_pipeline_outputs": wandb_images,
        "q2_7_summary_table":         table,
    })

    wandb.finish()

    print("DONE.")


if __name__ == "__main__":
    main()
