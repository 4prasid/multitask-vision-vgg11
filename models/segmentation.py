"""
Segmentation model
"""

import torch
import torch.nn as nn
from .vgg11 import *
from .layers import *

class VGG11UNet(nn.Module):
    """
    U-Net style segmentation network.

    Uses VGG11 encoder and decoder with skip connections.
    Uses ConvTranspose2d for learnable upsampling.
    Uses skip connections from encoder blocks for feature concatenation.
    Returns raw logits for Cross Entropy Loss.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5, encoder=None):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_p = dropout_p

        # VGG11 encoder
        # coming from vgg11.py
        if encoder is None:
            self.encoder = VGG11Encoder(in_channels=in_channels)
        else:
            self.encoder = encoder

        # Upsample layers
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Decoder layers
        # Conv2D -> BN -> ReLU -> Conv2D -> BN -> ReLU

        # implemnted in Dense to Shallow layers that's why 
        # used dropout in block_5 and block_4 only

        self.decoder_block_5 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), 
            CustomDropout(p=dropout_p),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        
        self.decoder_block_4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), 
            CustomDropout(p=dropout_p), 
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.decoder_block_3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)   )

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
            nn.Conv2d(64, 64, 3, padding=1,     bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        # final layer
        self.final_layer = nn.Conv2d(64, num_classes, kernel_size=1)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Encoder forward
        bottleneck, features = self.encoder(x, return_features=True)

        # Extract skip connections
        skip_1 = features["block1"] # (B, 64, 224, 224)
        skip_2 = features["block2"] # (B, 128, 112, 112)
        skip_3 = features["block3"] # (B, 256, 56, 56)
        skip_4 = features["block4"] # (B, 512, 28, 28)
        skip_5 = features["block5"] # (B, 512, 14, 14)

        # Decoder forward pass
        # Upsample -> concat -> decoder_block
        # Assertions to check if the skip connections are of the same size as the decoder block

        # Block 5
        x_5 = self.up5(bottleneck) # (B, 512, 14, 14)
        x_5 = torch.cat([x_5, skip_5], dim=1) # (B, 1024, 14, 14)
        assert x_5.shape[2:] == skip_5.shape[2:] 
        x_5 = self.decoder_block_5(x_5) # (B, 512, 14, 14)

        # Block 4
        x_4 = self.up4(x_5) # (B, 512, 28, 28)
        x_4 = torch.cat([x_4, skip_4], dim=1) # (B, 1024, 28, 28)
        assert x_4.shape[2:] == skip_4.shape[2:] 
        x_4 = self.decoder_block_4(x_4) # (B, 512, 28, 28)

        # Block 3
        x_3 = self.up3(x_4) # (B, 256, 56, 56)
        x_3 = torch.cat([x_3, skip_3], dim=1) # (B, 512, 56, 56)
        assert x_3.shape[2:] == skip_3.shape[2:] 
        x_3 = self.decoder_block_3(x_3) # (B, 256, 56, 56)

        # Block 2
        x_2 = self.up2(x_3) # (B, 128, 112, 112)
        x_2 = torch.cat([x_2, skip_2], dim=1) # (B, 256, 112, 112)
        assert x_2.shape[2:] == skip_2.shape[2:] 
        x_2 = self.decoder_block_2(x_2) # (B, 128, 112, 112)

        # Block 1
        x_1 = self.up1(x_2) # (B, 64, 224, 224)
        x_1 = torch.cat([x_1, skip_1], dim=1) # (B, 128, 224, 224)
        assert x_1.shape[2:] == skip_1.shape[2:] 
        x_1 = self.decoder_block_1(x_1) # (B, 64, 224, 224)

        # Final layer
        logits = self.final_layer(x_1) # (B, num_classes, 224, 224)

        if return_features:
            return logits, {"decoder_output": x_1, "skip_connections": {"block_1": skip_1, "block_2": skip_2, "block_3": skip_3, "block_4": skip_4, "block_5": skip_5}}
        else:
            return logits
