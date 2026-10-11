# Findings

Written analysis of the experiments behind the [VGG11 multi-task vision pipeline](README.md). Interactive plots are in the [W&B report](https://forge.coreweave.com/wandb/prasid-indian-institute-of-technology-madras/da6401_assignment2/reports/VGG11-Multi-Task-Vision--VmlldzoxNjQ4OTM5Mw); this file is the durable, self-contained version of the analysis.

---

Every training run uses the same random 80/20 split (seed 42) of the annotated Oxford-IIIT Pet images, and every validation number comes from the 20% hold-out (the one exception is the 15-image detection gallery, which is sampled from the whole annotated set). All runs log to the W&B.

**Mask convention:** The dataset loader maps the raw trimap values 1/2/3 to **0 = pet, 1 = background, 2 = border**. Segmentation figures use black for the pet, red for the background and green for the border.

**Metric definitions:** Several segmentation numbers appear below, and they are not interchangeable.

| Metric | Definition | Where it is used |
|---|---|---|
| Macro F1 | `sklearn` macro-averaged F1 over the 37 breeds | BatchNorm, dropout, joint training |
| IoU accuracy @ 0.5 | Fraction of validation images whose predicted box has IoU > 0.5 with the ground-truth head box | Joint training |
| Pixel accuracy | Correctly classified pixels / all pixels, over all three classes (border included). Averaged per image in the transfer-learning runs, pooled over all validation pixels in joint training | Transfer learning, Dice vs pixel accuracy, joint training |
| Macro Dice (3-class) | Dice of pet, background and border computed per image, averaged over the three classes, then averaged over images (ε = 1e-6 in numerator and denominator) | Transfer learning, Dice vs pixel accuracy |
| Class-1 Dice (joint training) | Dice of class index 1 only, with border pixels ignored, computed per batch and averaged. | Joint training (the logged `val_dice`) |

---

## 1. Internal Dynamics

**Setup:** Two VGG11 classifiers were trained under identical conditions: Adam, LR 5e-5, weight decay 5e-4, StepLR ×0.5 every 10 epochs, batch size 32, 20 epochs, label smoothing 0.1, dropout 0.5 in the two FC layers. One model uses the repository's VGG11 encoder (Conv → BatchNorm → ReLU, bias-free convolutions). The other is the same architecture without BatchNorm (convolutions with bias, PyTorch's default initialization). After every epoch a fixed validation batch (the first one) is passed through each model in eval mode, and the output of the **second convolutional layer** (`Conv2d 64→128`, the first conv of block 2) is recorded. That tensor is the raw convolution output, captured before BatchNorm and ReLU.

**Results:**

| | Without BatchNorm | With BatchNorm |
|---|---|---|
| Std of probed activations | ≈ 0.24 at epoch 1, falls to ≈ 0 by epoch 20 | ≈ 0.50 at epoch 1, ≈ 0.43 at epoch 20 |
| Mean of probed activations | ≈ +0.02, falls to ≈ 0 | ≈ −0.01, drifts to ≈ −0.09 |
| Histogram | Wide at epoch 1, collapses to a spike at zero | Broad and roughly symmetric around zero at every epoch |
| Train loss | Flat at ≈ 3.6 | 3.7 → ≈ 1.1 |
| Val loss | Flat at ≈ 3.6 | 3.55 → ≈ 2.5–2.7 |
| Train F1 | ≈ 0 throughout | ≈ 0.04 → ≈ 0.9 |
| Val F1 | ≈ 0 throughout | ≈ 0.02 → ≈ 0.4 |

**without BatchNorm:** The loss stays at ≈ 3.6, which is ln 37, the cross-entropy of a uniform prediction over 37 classes. The network never moves off chance. PyTorch's default convolution and linear initialization has weight variance of about 1/(3·fan_in), which shrinks the second moment of the signal by roughly a factor of 6 at every Conv/Linear + ReLU pair. After 8 convolutions and 3 FC layers, almost nothing reaches the logits, and the gradients flowing back are equally small, so the weights barely update. The probed layer shows the same thing from the inside: its output standard deviation decays to zero as training proceeds and the histogram collapses onto zero.

**with BatchNorm:** BatchNorm in block 1 re-centres and re-scales the signal that enters the probed layer at every step, so its statistics stay in a healthy range for all 20 epochs (the plotted tensor is the pre-BatchNorm conv output, so the stability seen here comes from the normalized inputs it receives, not from BatchNorm acting on that tensor itself). The model learns from the first epoch: validation F1 climbs from ≈ 0.02 to ≈ 0.35 by epoch 11 and ends near 0.4, while train F1 reaches ≈ 0.9. The gap between those two curves is the over-fitting examined in the next section.

**Convergence speed and maximum stable learning rate:** At the single learning rate tested (5e-5), the BatchNorm network converged and the unnormalized one did not move. This experiment did not sweep the learning rate and did not try He initialization for the unnormalized network, so it shows that BatchNorm makes training robust to a poor initialization and a small step size; it does not measure how much larger a learning rate each variant could tolerate. The usual explanation for that further benefit is that normalization bounds activation scale and smooths the loss surface, so steps stay predictable at higher learning rates.

---

## 2. The Regularization Effect of Dropout

**Setup:** Three BatchNorm VGG11 classifiers were trained for 25 epochs with identical settings (Adam, LR 5e-5, weight decay 5e-4, StepLR ×0.5 every 10 epochs, batch 32, label smoothing 0.1). The only difference is the dropout probability in the two 4096-unit FC layers, implemented with the repository's `CustomDropout` (inverted Bernoulli mask during training, identity in eval): p = 0, 0.2 and 0.5. Training used resize and normalization only: the augmentation pipeline defined in the script (flip, shift/scale/rotate, colour jitter, noise) is not applied, because the train and validation subsets share one dataset object and the validation transform overwrites the training one.

**Results:**

| | p = 0 | p = 0.2 | p = 0.5 |
|---|---|---|---|
| Train loss at epoch 25 | ≈ 0.7 | ≈ 0.7 | ≈ 0.9, still falling |
| Val loss at epoch 25 | ≈ 2.45–2.55 | ≈ 2.45–2.55 | ≈ 2.45–2.55 |
| Gap (train F1 − val F1) | ≈ 0.6, plateau from epoch ≈ 11 | ≈ 0.6, plateau from epoch ≈ 11 | ≈ 0.6, plateau reached at epoch ≈ 17–20 |

The overfit gap in these plots is the **train F1 minus validation F1**, not a loss difference.

**Curves:**

- **Training loss:** With label smoothing 0.1 over 37 classes, the smallest attainable cross-entropy is ≈ 0.67, the entropy of the smoothed target. The p = 0 and p = 0.2 runs end at ≈ 0.7, so they fit the training set essentially perfectly. The p = 0.5 run is slower. Part of that is regularization (each update trains a random sub-network), and part is measurement: training loss and F1 are computed with dropout active, so they are depressed in proportion to p.
- **Validation loss:** All three runs finish within about 0.1 of each other. The p = 0 and p = 0.2 runs have visible validation-loss spikes around epochs 5–8; the p = 0.5 run has no spike of comparable size early on (it shows a smaller bump to ≈ 2.8 near epoch 17). None of the three is clearly better at the end.
- **Generalization gap:** Dropout changed *when* the gap opened, not how large it ended up. The p = 0.5 gap grows more gradually and reaches the same ≈ 0.6 plateau about six to nine epochs later than the other two.

**How the custom dropout alters generalization:** During training the layer zeroes each activation with probability p and rescales the survivors by 1/(1−p), so the expected activation is unchanged and no unit can rely on a specific partner being present. That slows memorization, which is what the delayed gap shows. At evaluation the layer is the identity, so the validation numbers describe the full network deterministically; this train/eval split is what makes the gap comparable across the three runs.

**Why dropout alone did not close the gap:** The classifier head has about 120M parameters (25088→4096→4096→37), against ≈ 3k training images and a convolutional encoder of only ≈ 9.2M parameters. The head can memorize the training set under any of these dropout rates, and the only other regularizers are weight decay and label smoothing. Closing the gap would need more effective regularization of the data side (working augmentation) or a smaller head; p = 0.5 on the FC layers alone is not enough here. The joint-training section shows the same ≈ 2.5 validation loss for the classifier.

---

## 3. Transfer Learning Showdown

**Setup:** A VGG11-encoder U-Net (transposed-convolution decoder with skip connections) was trained for 20 epochs with each of three strategies. In every case the encoder starts from the classifier checkpoint and the decoder is randomly initialized. Settings: plain cross-entropy, Adam LR 1e-4, weight decay 1e-4, StepLR ×0.5 every 8 epochs, batch 16, gradient clipping at 1.0.

| Strategy | What is trained |
|---|---|
| Frozen (strict feature extractor) | Decoder and final 1×1 conv only; all encoder weights frozen |
| Partial fine-tuning | Decoder plus encoder blocks 4–5; blocks 1–3 frozen |
| Full fine-tuning | Everything |

In all three runs the model is in training mode, so the BatchNorm layers of a "frozen" encoder still normalize with batch statistics and update their running statistics; the frozen weights do not change, but the encoder's normalization adapts to the segmentation data.

**Results (validation, final epoch):**

| Strategy | Dice (3-class macro) | Val loss | Convergence |
|---|---|---|---|
| Full fine-tuning | **0.82** | **0.34** | Fast and stable |
| Partial fine-tuning | 0.79 | 0.42 | Moderate and stable |
| Frozen | 0.73 | 0.49 | Moderate, with oscillating val loss from epoch ≈ 10 |

**Empirical comparison:**

- **Convergence speed:** Full fine-tuning reaches Dice ≈ 0.73 by epoch 2–3. At the same point the partial run is at ≈ 0.70 and the frozen run at ≈ 0.62. The ordering never changes afterwards.
- **Stability:** The frozen run's validation loss oscillates between ≈ 0.43 and ≈ 0.49 from about epoch 10 on. The full run's validation loss bottoms out at ≈ 0.31–0.32 around epochs 12–15 and drifts up slightly to ≈ 0.34 at epoch 20 while Dice keeps improving slowly; the partial run behaves similarly (≈ 0.35 rising to ≈ 0.42).
- **Final performance:** Full > partial > frozen on both Dice and loss. The gap between full and frozen (0.09 Dice) is the clear effect. The 0.03 gap between full and partial is small enough that a single run per strategy cannot separate it from run-to-run noise.
- **Cost per epoch:** Structurally, a frozen encoder needs no backward pass through its weights, partial fine-tuning back-propagates through two blocks, and full fine-tuning through all five, so cost per epoch rises in that order.

**Why unfreezing helps:** The early blocks learn generic, low-level features (edges, colour gradients, simple textures) that are useful for any image task, so freezing them (as the frozen and partial strategies do) costs little. The later blocks are different. They were trained to separate 37 breeds, which rewards global, discriminative patterns and actively discards spatial detail through repeated pooling. A segmentation decoder needs the opposite: dense, spatially precise features at every skip-connection depth. With a frozen encoder the decoder must make do with representations built for classification, which caps the achievable Dice. Unfreezing the late blocks lets the segmentation loss reshape the semantic features that feed the deepest skip connections, and unfreezing everything lets it also adjust mid-level features. Catastrophic forgetting is not a practical risk here because the pretraining and target data are the same images, so the pretrained features are already relevant; in a cross-domain transfer, partial fine-tuning would be the safer default.

---

## 4. Inside the Black Box: Feature Maps

**Setup:** The first image of the annotated list (an Abyssinian cat, label 0) is passed through the trained classifier. Two feature maps are extracted: the output of **block 1** (the first convolution, 64 channels at 224×224) and the output of the **last convolution of block 5, before the final max-pool** (512 channels at 14×14). The first 16 channels of each are shown with a viridis colormap. They are the first 16 channels in index order, not a selection of the most active ones.

**Block 1:** The maps look like filtered copies of the photograph. The cat's silhouette, the outline of its face, the fur texture and the edges of the surrounding blue fabric are all visible. Different channels do different low-level jobs: some respond to edges along the body contour, some to the orange-versus-blue colour boundary, and a few suppress the animal and highlight the background. Spatial layout is fully preserved at this depth.

**Block 5:** The maps are 14×14, mostly near zero, with a few bright blobs at isolated locations. They no longer resemble the input. Each cell has a stride of 16 pixels but a theoretical receptive field of about 134×134 pixels, more than half the 224×224 input, so a single cell integrates evidence from a large part of the image. The classifier then flattens the 7×7×512 map after the last pool (it does not use global average pooling) and feeds it to the FC layers.

**From edges to semantic shapes:** Each convolution combines the previous layer's outputs, so the effective receptive field grows with depth: edges and colour contrasts in block 1 combine into textures and corners in block 2, into parts such as fur patterns and eye or ear outlines in blocks 3–4, and into compact, sparse codes for part configurations in block 5. Max-pooling halves the resolution four times on the way, which trades exact position for tolerance to small shifts: a feature fires if it is present anywhere in its region. This hierarchy is not designed in; it emerges because separating 37 visually similar breeds under a cross-entropy loss requires increasingly abstract, breed-diagnostic features. The block 5 maps themselves cannot tell us which parts individual channels encode. That would need a dedicated probe such as activation maximization.

---

## 5. Object Detection: Confidence & IoU

**Setup:** Fifteen images were drawn at random from the annotated set (unseeded `random.sample`, so not restricted to the validation split). For each, the localizer predicts a head box and the breed classifier gives a softmax distribution on the same image. **IoU** compares the predicted box to the ground-truth head box (scaled to 224×224). **Confidence** is the classifier's top-1 softmax probability. The localizer and the classifier are separate networks.

| Sample | IoU | Confidence | IoU ≥ 0.5 |
|---|---|---|---|
| 1 | 0.888 | 0.959 | yes |
| 2 | 0.781 | 0.822 | yes |
| 3 | 0.796 | 0.944 | yes |
| 4 | 0.739 | 0.809 | yes |
| 5 | 0.362 | 0.937 | **no** |
| 6 | 0.252 | 0.755 | **no** |
| 7 | 0.644 | 0.958 | yes |
| 8 | 0.513 | 0.840 | yes |
| 9 | 0.799 | 0.848 | yes |
| 10 | 0.861 | 0.924 | yes |
| 11 | 0.763 | 0.969 | yes |
| 12 | 0.675 | 0.863 | yes |
| 13 | 0.785 | 0.885 | yes |
| 14 | 0.948 | 0.678 | yes |
| 15 | 0.656 | 0.941 | yes |
| **Mean** | **0.697** | **0.875** | 13 of 15 |

The W&B image captions use a stricter label (FAIL if IoU < 0.3, HIGH-CONF FAIL if confidence is also above 0.5), so under that rule only sample 6 is flagged and sample 5 reads "OK".

**Where the localizer works:** On clean, well-lit pets, boxes are tight: samples 1, 10 and 14 reach IoU 0.89, 0.86 and 0.95.

**Confidence says nothing about box quality:** Across the 15 samples, the correlation between IoU and confidence is essentially zero (Pearson r ≈ 0.05, Spearman ≈ 0.05). The two worst boxes have confidence 0.94 and 0.76, close to the average of the good ones (0.88), and the best box (IoU 0.948) has the lowest confidence of all (0.678). This is expected from the design. The confidence comes from the breed classifier, which judges fur colour, face shape and body shape over the whole image; the box comes from a different network that regresses four numbers. A classifier can be very sure of the breed while the box is wrong.

**Failure cases:** In both samples below 0.5 the predicted box is much larger than the ground-truth head box and covers part of the body.

- **Sample 5** (IoU 0.362, confidence 0.937): a white kitten resting on a bed in front of a plaid blanket. The ground-truth box is tight on the face. The prediction extends above the head onto the blanket and down over the chest.
- **Sample 6** (IoU 0.252, confidence 0.755): a large white dog standing in front of a wooden fence in snow. The prediction covers the head and extends down over the chest, while the ground truth is the head alone.

Plausible contributors, none of which was tested here: the ground truth is a *head* box while the salient object is the whole animal, and a regression head that sees the image through a flattened 7×7 map has no explicit mechanism for separating the head from the body; both subjects are large and bright in the frame; and the head is a small, tight target, so small absolute errors cost a lot of IoU.

---

## 6. Segmentation Evaluation: Dice v/s Pixel Accuracy

**Setup:** The full fine-tuning run from the transfer-learning experiment (plain cross-entropy, 3 classes, 20 epochs). Pixel accuracy and macro Dice are computed per image on the validation set and averaged. Predicted and ground-truth masks for the first five validation images are logged at epochs 1, 5, 10, 15 and 20.

**Observation:**

| Epoch | Pixel accuracy | Macro Dice |
|---|---|---|
| 1 | ≈ 0.76 | ≈ 0.58 |
| 20 | ≈ 0.89 | ≈ 0.82 |

Pixel accuracy is higher than Dice throughout, and the gap is widest early (≈ 0.18 at epoch 1, ≈ 0.07 at epoch 20).

**Why, mathematically:** Pixel accuracy is Σ<sub>c</sub> TP<sub>c</sub> / N, so every pixel counts equally and classes contribute in proportion to their size. The Dice for class c is 2·TP<sub>c</sub> / (2·TP<sub>c</sub> + FP<sub>c</sub> + FN<sub>c</sub>), and macro Dice averages the three classes with equal weight whatever their size. The trimaps are imbalanced: the background is the largest class, the pet most of the rest, and the border is a thin ring around the animal (roughly 55%, 38% and 7% of pixels as typical orders of magnitude). Two consequences follow.

- A model that gets background and pet perfectly right but misses every border pixel scores ≈ 93% pixel accuracy, yet its macro Dice is at most 2/3, because the border Dice is 0.
- A model that predicts "background" everywhere scores a pixel accuracy equal to the background share (≈ 0.55), while its macro Dice is ≈ 0.24: the background Dice is ≈ 0.71 and the other two classes score 0.

**Early epochs:** In the first epoch the network already gets the large, easy regions right, so pixel accuracy starts high at ≈ 0.76 while macro Dice is ≈ 0.58. The remaining error sits in small or hard regions, such as the border ring and the pet's boundary. Per-class Dice was not logged in this run, so that attribution follows from the metric definitions, not from a per-class breakdown.

**Why Dice is the better metric here:** Per-class normalization means a failure on a small class cannot hide behind a large class, which is precisely the case for a thin border band. Dice also penalizes both missed pixels and spurious pixels relative to the class size. It has its own weakness: for classes that are tiny or absent, it is noisy, and the ε smoothing returns 1 when a class is absent from both prediction and ground truth. Reporting both metrics, as done here, is more informative than either alone.

**Sample predictions:** On most of the five samples the predicted mask follows the ground-truth silhouette closely and the border band sits in the right place. The weakest is a brown dog on a patterned rug: the predicted pet region is fragmented and the border band is thicker and less precise than the ground truth, which fits the low colour contrast between the dog and the rug.

---

## 7. The Final Pipeline Showcase

**Setup:** Three images that are not in the dataset were run through the unified architecture (`MultiTaskPerceptionModel`) loaded from the three single-task checkpoints; the jointly fine-tuned weights were not used. One image came from the internet and two were photographed on the IITM campus. Each is resized to 224×224. The classifier receives the full resized image, so the box is not used to crop the input, and a bad box does not affect the breed prediction.

| Image | Breed prediction (top-1) | Box | Segmentation |
|---|---|---|---|
| Dog (internet, clean photo) | Shiba Inu, 0.18 (then Beagle 0.14, Abyssinian 0.07) | Good: covers head and upper body | Excellent: clean body/grass separation with a well-defined border ring |
| Cat 1 (IITM, low-light indoor) | English Cocker Spaniel, 0.15 (then Wheaten Terrier 0.09, English Setter 0.09) | Moderate: covers the torso, larger than the head | Partial: noisy mask, thick and irregular border |
| Cat 2 (IITM, outdoor, white cat) | English Setter, 0.38 | Missed: box sits in the upper left near a tree trunk, the cat is lower centre | Good: clean silhouette and border |

**Classification:** Wrong on all three. The internet dog looks like a golden-coated retriever, a breed that is not among the 37 classes, so no breed prediction could be right; the low top-1 probability, with the remaining mass spread across unrelated breeds, is the appropriate response. Both cats were labelled as dog breeds, which is a species-level error. This is consistent with the overall picture of the classifier: a validation macro-F1 of 0.46 against 0.98 on the training set means it has learned the training distribution much better than it has learned breed features that transfer. Dark indoor light and an outdoor scene with a white cat both fall well outside that distribution.

**Localization:** Of the three outputs, the box is the most sensitive to context. It is right on a clean photo, oversized for a lying cat in low light, and misses entirely in a cluttered outdoor scene (soil, roots, a wall and branches). The regression head predicts four numbers from a flattened feature map, so competing salient regions can pull the prediction away from the subject.

**Segmentation:** Segmentation is the most robust output. It succeeded in two of three images, including the one where the box failed, and degraded only in the low-contrast indoor shot (tabby fur against a similarly toned floor). The U-Net decodes from skip connections at five scales and predicts at every pixel, so it relies on local boundaries and texture, which survive a change of background far better than a global breed signature or a four-number box regression.

**Summary:** Segmentation generalizes best, localization is sensitive to clutter, and classification is the least transferable of the three. A mask-derived box would be a natural fallback when the regressed box is unreliable.

---

## 8. Meta-Analysis and Reflection

**Setup:** `MultiTaskPerceptionModel` holds three task-specific VGG11 encoders and three heads. Each encoder starts from its own single-task checkpoint (`classifier.pth`, `localizer.pth`, `segmenter.pth`), and the full model is fine-tuned for 20 epochs with a weighted sum of the three losses:

`L = 1.0 · CE(breed) + 5.0 · (1 − IoU)(box) + 2.0 · CE(trimap)`

Adam LR 1e-4 with weight decay 1e-4, StepLR ×0.1 every 10 epochs (so 1e-4 → 1e-5 at epoch 10), batch size 8, gradient clipping at 1.0, dropout 0.5 / 0.3 / 0.3 in the classification head, localization head and decoder. At the start of training, after weighting, the three terms are roughly 2.4 (classification), 3.0 (localization) and 0.3–0.7 (segmentation).

**Loss curves:**

| | Train | Validation |
|---|---|---|
| Total weighted loss | ≈ 5.7 → ≈ 2.0 | ≈ 6.3 early, ≈ 4.7 right after the LR step, ≈ 5.1 by epoch 20 |
| Classification loss | ≈ 2.5 → ≈ 0.05 | ≈ 2.5–2.7, minimum ≈ 2.2 near epoch 12, ending ≈ 2.6 |
| Localization loss (1 − IoU) | ≈ 0.6 → ≈ 0.35 | tracks train closely, ends ≈ 0.37 |
| Segmentation loss | ≈ 0.15, rising to ≈ 0.2 over epochs 2–9, then ≈ 0.10 after the LR step | ≈ 0.25–0.30, one early spike |

The learning-rate step at epoch 10 is visible everywhere: the training loss falls sharply between epochs 10 and 12 (≈ 3.5 → ≈ 2.7), validation loss drops from ≈ 6.0 to ≈ 4.7, and train F1 jumps from ≈ 0.76 to ≈ 0.95 within one epoch. After that, validation loss drifts upward, driven by the classification term.

**Validation metrics at epoch 20:**

| Task | Metric | Train | Validation |
|---|---|---|---|
| Classification | Macro F1 | ≈ 0.98 | ≈ 0.46 (plateau from epoch ≈ 14) |
| Localization | IoU accuracy @ 0.5 | ≈ 0.84 | ≈ 0.79 |
| Segmentation | Pixel accuracy (3-class) | not logged | ≈ 0.91 |
| Segmentation | Class-1 (background) Dice | ≈ 0.98 | ≈ 0.94 |

The logged Dice is the class-1 Dice defined above, which measures the background, so it is high by construction and should not be read as pet-segmentation quality. The comparable pet-aware number is the 3-class macro Dice of 0.82 from the segmentation-only run, and the validation pixel accuracy of ≈ 0.91 (against ≈ 0.89 for that run) shows that the unified model's segmentation is at least as good.

**Generalization gaps** (train loss − validation loss, so more negative means more overfitting): localization stays at ≈ 0 for all 20 epochs, segmentation is slightly negative (validation loss modestly above train loss, a small gap of roughly 0.05–0.2), and classification diverges to ≈ −2.5 by epoch 20. Classification is the only task that overfits.

### Placement of BatchNorm and dropout

BatchNorm follows every convolution in all three encoders and in the decoder, and `CustomDropout` sits after each of the FC layers in the classification and localization heads and in decoder stages 5 and 4. For localization and segmentation that configuration works: both losses descend smoothly and their train/validation gaps are near zero. No joint run without BatchNorm was made, so its effect in the joint setting is not isolated; the observable result is that all three losses descend without instability. For classification it does not. The 25088→4096→4096→37 head has about 120M parameters, more than ten times the encoder's ≈ 9.2M, and 0.5 dropout together with BatchNorm cannot stop it from memorizing about 3k training images. This is the same ≈ 2.5 validation loss that the standalone dropout experiments reached, so the classification gap is inherited from the single-task stage and is not caused by joint training. Stage 1 also ran without effective augmentation (see the dropout section). A smaller head or global pooling, and a working augmentation pipeline, are the changes that this evidence points to.

### Encoder strategy and task interference

For localization, the encoder was frozen for the first 10 epochs, then block 5 was unfrozen together with the head at a learning rate of 1e-6 (LR × 0.02). The resulting checkpoint therefore keeps the weights of blocks 1–4 exactly as the classifier left them and only adapts the last block. In the unified model each task keeps its own encoder, so there is no shared backbone to interfere. Gradients from the localization and segmentation losses never reach the classification encoder, and vice versa. That is why the localization and segmentation gaps stay near zero while classification overfits: the three tasks do not compete, and the classification overfitting is not something the other tasks caused. The price is parameters (three encoders instead of one) and the absence of any feature sharing; whether a shared backbone with light task adapters would help or hurt the individual tasks was not tested.

### Segmentation loss

Standalone segmentation (`seg_train.py`) trains with 0.3 · class-weighted cross-entropy + 0.7 · class-weighted soft Dice over all three classes, so the border class is supervised explicitly. The transfer-learning and joint-training runs use plain cross-entropy. Cross-entropy supplies stable per-pixel gradients, and including the border as a class gives the model an explicit signal on the boundary. Plain cross-entropy was enough to reach a macro Dice of 0.82 and a pixel accuracy of 0.89–0.91, but no experiment compared cross-entropy with the Dice-augmented loss directly, so the benefit of the Dice term is not isolated by these runs.
