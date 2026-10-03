"""
Unified multi-task model
"""

import torch
import torch.nn as nn
import os
from .vgg11 import *
from .layers import *

class MultiTaskPerceptionModel(nn.Module):
    """
    Shared-backbone multi-task model.
    """

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "segmenter.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """

        super().__init__()

        def safe_download(path, file_id):
            if (not os.path.exists(path)) or os.path.getsize(path) < 1e6:
                import gdown
                gdown.download(id=file_id, output=path, quiet=False)

        safe_download(classifier_path, "1Wv-wSVGKvdZlyHg0xbTGQgoVmAvhDUNJ")
        safe_download(localizer_path, "1ptO48uAaxdhC9QGTYQjXYyU9mTONIn7n")
        safe_download(unet_path, "1VC-WQdIW69UbjjFqBgmyr3gV-4wy10I9")

        self.num_breeds = num_breeds
        self.seg_classes = seg_classes
        self.in_channels = in_channels

        self.eps = 1e-6

        p_cls = 0.5
        p_loc = 0.3
        p_seg = 0.3

        # Three separate encoders : each loaded from its own trained checkpoint:
        # self.encoder     -> classification head  (loaded from classifier.pth)
        # self.loc_encoder -> localization head    (loaded from localizer.pth)
        # self.seg_encoder -> segmentation head    (loaded from segmenter.pth)
        self.encoder     = VGG11Encoder(in_channels=in_channels)
        self.loc_encoder = VGG11Encoder(in_channels=in_channels)
        self.seg_encoder = VGG11Encoder(in_channels=in_channels)


        # Classification head                                                 
        
        c_out = 7
        self.cls_fc1 = nn.Sequential(
            nn.Linear(512 * c_out * c_out, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=p_cls))

        self.cls_fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=p_cls))

        self.cls_fc3 = nn.Linear(4096, num_breeds)

        
        # Localization head                                                   
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.localizer_fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            CustomDropout(p=p_loc))

        self.localizer_fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            CustomDropout(p=p_loc))

        self.localizer_fc3 = nn.Sequential(
            nn.Linear(512, 4),
            nn.Sigmoid())

       
        # Segmentation head                                                   
       
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)

        self.decoder_block_5 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=p_seg),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.decoder_block_4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=p_seg),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.decoder_block_3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.final_layer = nn.Conv2d(64, seg_classes, kernel_size=1)

        self._load_weights(classifier_path, localizer_path, unet_path)

    def _load_weights(self, classifier_path: str, localizer_path: str, unet_path: str):
        """
        Load weights from three single-task checkpoints.

        classifier.pth -> self.encoder     + cls_fc*
        localizer.pth  -> self.loc_encoder + localizer_fc* / avgpool
        segmenter.pth  -> self.seg_encoder + up* / decoder_block_* / final_layer
        """
        device = torch.device("cpu")

        
        # Classifier -> self.encoder + classification head                
        
        cls_ckpt = torch.load(classifier_path, map_location=device)
        if isinstance(cls_ckpt, dict) and "state_dict" in cls_ckpt:
            cls_ckpt = cls_ckpt["state_dict"]

        encoder_state  = {}
        cls_head_state = {}
        for k, v in cls_ckpt.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder."):]] = v
            elif k.startswith("fc1."):
                cls_head_state["cls_fc1." + k[len("fc1."):]] = v
            elif k.startswith("fc2."):
                cls_head_state["cls_fc2." + k[len("fc2."):]] = v
            elif k.startswith("fc3."):
                cls_head_state["cls_fc3." + k[len("fc3."):]] = v
            elif k.startswith("cls_fc"):
                cls_head_state[k] = v

        self.encoder.load_state_dict(encoder_state, strict=False)
        self.load_state_dict(cls_head_state, strict=False)

       
        # Localizer -> self.loc_encoder + localization head               
        
        loc_ckpt = torch.load(localizer_path, map_location=device)
        if isinstance(loc_ckpt, dict) and "state_dict" in loc_ckpt:
            loc_ckpt = loc_ckpt["state_dict"]

        loc_encoder_state = {}
        loc_head_state    = {}
        for k, v in loc_ckpt.items():
            if k.startswith("encoder."):
                loc_encoder_state[k[len("encoder."):]] = v
            elif k.startswith("localizer_fc1."):
                loc_head_state[k] = v
            elif k.startswith("localizer_fc2."):
                loc_head_state[k] = v
            elif k.startswith("localizer_fc3."):
                loc_head_state[k] = v
            elif k.startswith("avgpool."):
                loc_head_state[k] = v

        # Load localizer's own encoder — does NOT touch self.encoder
        self.loc_encoder.load_state_dict(loc_encoder_state, strict=False)
        self.load_state_dict(loc_head_state, strict=False)

        
        # UNet -> self.seg_encoder + segmentation decoder                 
        
        seg_ckpt = torch.load(unet_path, map_location=device)
        if isinstance(seg_ckpt, dict) and "state_dict" in seg_ckpt:
            seg_ckpt = seg_ckpt["state_dict"]

        seg_encoder_state = {}
        seg_head_state    = {}
        for k, v in seg_ckpt.items():
            if k.startswith("encoder."):
                seg_encoder_state[k[len("encoder."):]] = v
            elif (k.startswith("up") or
                  k.startswith("decoder_block") or
                  k.startswith("final_layer")):
                seg_head_state[k] = v

        # Load segmenter's own encoder — does NOT touch self.encoder or self.loc_encoder
        self.seg_encoder.load_state_dict(seg_encoder_state, strict=False)
        self.load_state_dict(seg_head_state, strict=False)


    def forward(self, x: torch.Tensor):
        """
        Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization':   [B, 4] bounding box tensor (pixel space).
            - 'segmentation':   [B, seg_classes, H, W] segmentation logits tensor.
        """

        _, _, H, W = x.shape

        
        # Classification : uses self.encoder (classifier weights)           
       
        cls_bottleneck, _ = self.encoder(x, return_features=True)
        flat = cls_bottleneck.reshape(cls_bottleneck.size(0), -1)
        cls  = self.cls_fc3(self.cls_fc2(self.cls_fc1(flat)))

        
        # Localization : uses self.loc_encoder (localizer weights)           
        # localizer.pth saved with return_normalized_coords=False,           
        # so we do the pixel-space conversion here ourselves.                
       
        loc_bottleneck, _ = self.loc_encoder(x, return_features=True)
        loc = self.avgpool(loc_bottleneck)
        loc = loc.reshape(loc.size(0), -1)
        loc = self.localizer_fc3(self.localizer_fc2(self.localizer_fc1(loc)))
        loc = torch.clamp(loc, min=self.eps, max=1 - self.eps)

        # Convert normalized [0,1] -> pixel coordinates
        xc  = loc[:, 0] * W
        yc  = loc[:, 1] * H
        bw  = loc[:, 2] * W
        bh  = loc[:, 3] * H
        loc = torch.stack([xc, yc, bw, bh], dim=1)   # [B, 4] pixel space

        
        # Segmentation :uses self.seg_encoder (segmenter weights)           
        
        seg_bottleneck, seg_features = self.seg_encoder(x, return_features=True)

        s1 = seg_features["block1"]
        s2 = seg_features["block2"]
        s3 = seg_features["block3"]
        s4 = seg_features["block4"]
        s5 = seg_features["block5"]

        x5 = self.up5(seg_bottleneck)
        x5 = self.decoder_block_5(torch.cat([x5, s5], dim=1))

        x4 = self.up4(x5)
        x4 = self.decoder_block_4(torch.cat([x4, s4], dim=1))

        x3 = self.up3(x4)
        x3 = self.decoder_block_3(torch.cat([x3, s3], dim=1))

        x2 = self.up2(x3)
        x2 = self.decoder_block_2(torch.cat([x2, s2], dim=1))

        x1 = self.up1(x2)
        x1 = self.decoder_block_1(torch.cat([x1, s1], dim=1))

        seg = self.final_layer(x1)
        seg = torch.nn.functional.interpolate(
            seg,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        return {
            'classification': cls,
            'localization':   loc,
            'segmentation':   seg
        }