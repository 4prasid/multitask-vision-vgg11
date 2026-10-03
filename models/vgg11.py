"""
VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """
    VGG11-style encoder with optional intermediate feature returns.
    5 convolutional blocks with increasing filter sizes (64, 128, 256, 512, 512).
    Each block consists of convolutional layers followed by batch normalization, ReLU activation, and max pooling.
    The forward method can return just the bottleneck features or also the intermediate features for skip connections in a U-Net decoder.
    """

    def __init__(self, in_channels : int = 3):
        """Initialize the VGG11Encoder model. 
        Args:
            in_channels: number of input channels (default: 3 for RGB images).
        """

        super().__init__()
        self.in_channels = in_channels

        # VGG11 architecture consists of 5 convolutional blocks, 
        # each block has a certain number of convolutional layers followed by a max pooling layer.
        # all filters are 3x3 with padding=1 to preserve spatial dimensions before pooling.
        # Max pooling layers have kernel size 2 and stride 2 to downsample the feature maps by a factor of 2.
        # covn. -> BN -> ReLU -> Maxpool

        # Block 1: 1 conv layer, 64 filters 
        self.block1_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # Block 2: 1 conv layer, 128 filters
        self.block2_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # Block 3: 2 conv layers, 256 filters
        self.block3_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # Block 4: 2 conv layers, 512 filters  
        self.block4_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # Block 5: 2 conv layers, 512 filters
        self.block5_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.block5_pool = nn.MaxPool2d(kernel_size=2, stride=2)



    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """

        # Dictionary to hold intermediate features for skip connections, keys are block names 
        # and values are feature tensors
        features = {}

        # Pass the input through each block, storing the output of each block before pooling 
        # in the features dictionary.
        # Block 1 -> store in features["block1"], then apply max pooling and 
        # pass to next block and so on for all 5 blocks.

        # Block 1: (B, 3, 224, 224) -> (B, 64, 224, 224) -> (B, 64, 112, 112)
        x_1 = self.block1_conv(x)
        features["block1"] = x_1
        x_1 = self.block1_pool(x_1)

        # Block 2: (B, 128, 112, 112) -> (B, 128, 112, 112) -> (B, 128, 56, 56)
        x_2 = self.block2_conv(x_1)
        features["block2"] = x_2
        x_2 = self.block2_pool(x_2)

        # Block 3: (B, 256, 56, 56) -> (B, 256, 56, 56) ->(B, 256, 28, 28)
        x_3 = self.block3_conv(x_2)
        features["block3"] = x_3
        x_3 = self.block3_pool(x_3)

        # Block 4: (B, 512, 28, 28) -> (B, 512, 28, 28) -> (B, 512, 14, 14)
        x_4 = self.block4_conv(x_3)
        features["block4"] = x_4
        x_4 = self.block4_pool(x_4)

        # Block 5: (B, 512, 14, 14) -> (B, 512, 14, 14) -> (B, 512, 7, 7)
        x_5 = self.block5_conv(x_4)
        features["block5"] = x_5 
        x_5 = self.block5_pool(x_5)

        # x_5 is the bottleneck feature tensor after the final block and pooling.
        # If return_features is True, we return both the bottleneck features (x_5) and the dictionary
        # of intermediate features for skip connections. 
        # If return_features is False, we only return the bottleneck features (x_5).
        if return_features:
            return x_5, features
        else:
            return x_5 
        