"""
Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import *
from .layers import *

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.3, return_normalized_coords: bool = False, epsilon: float = 1e-6, encoder=None):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
            return_normalized_coords: Whether to return normalized bounding box coordinates.
            epsilon: Small value to prevent output coordinates from being exactly 0 or 1 when using sigmoid activation.
        """

        super().__init__()
        self.in_channels = in_channels
        self.dropout_p = dropout_p
        self.return_normalized_coords = return_normalized_coords
        self.epsilon = epsilon

        # VGG11 encoder: encoder head same for both localization and classification branches, so we can reuse the same encoder for both tasks if needed.    
        if encoder is None:
            self.encoder = VGG11Encoder(in_channels=in_channels)
        else:
            self.encoder = encoder

        # fine-tune/freeze encoder is handeled in the train.py not here.

        # Localization head: takes the encoder's output and predicts bounding box coordinates.
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # ensure output feature map is always 7x7 regardless of input image size

        # FC1(1024)
        self.localizer_fc1 = nn.Sequential(
            nn.Linear(512*7*7, 1024),    
            nn.ReLU(),
            CustomDropout(dropout_p)
        )  

        # FC2(512)
        self.localizer_fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            CustomDropout(dropout_p)
        )  

        # final layer to predict bounding box coordinates
        self.localizer_fc3 = nn.Sequential(
            nn.Linear(512, 4),
            nn.Sigmoid()  # assuming coordinates should be normalized between 0 and 1
        )  


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """

        _, _, H, W = x.shape  # get input image height and width for later use in converting normalized coords to pixel coords

        # pass through encoder to get bottleneck features

        bottleneck_features = self.encoder(x)  # [B, 512, 7, 7] 
        if isinstance(bottleneck_features, tuple):
            bottleneck_features = bottleneck_features[0]  # get the feature map if encoder returns a tuple

        # apply adaptive average pooling to ensure a fixed-size feature map
        bottleneck_features_pooled = self.avgpool(bottleneck_features)  # [B, 512, H', W']

        # flatten the bottleneck features to shape [B, 512*7*7]
        bottleneck_feature_flat = bottleneck_features_pooled.reshape(bottleneck_features_pooled.size(0), -1)  # [B, 512*7*7]

        # pass through the localization head to get bounding box coordinates

        # [B, 512*7*7] -> [B, 1024]
        localizer_fc1_out = self.localizer_fc1(bottleneck_feature_flat)

        # [B, 1024] -> [B, 512]
        localizer_fc2_out = self.localizer_fc2(localizer_fc1_out)
        
        # [B, 512] -> [B, 4]
        localizer_fc3_out = self.localizer_fc3(localizer_fc2_out)   # output is in normalized coordinates [0, 1] due to sigmoid activation

         # clamp to ensure coordinates are not exactly 0 or 1 to avoid issues with certain loss functions
         #  (like IoU-based losses) that can have undefined behavior when boxes have zero area.
        
        bbox_norm = torch.clamp(localizer_fc3_out, min=self.epsilon, max=1 - self.epsilon) 

        # if return_normalized_coords is True, we return the normalized coordinates in [0, 1].
        # If False, we convert the normalized coordinates to pixel coordinates based on the input image size
        # (xc_pixel = xc * W, yc_pixel = yc * H, w_pixel  = w * W, h_pixel  = h * H) and return those.

        if self.return_normalized_coords:
            return bbox_norm  
        else:
            xc_pixel = bbox_norm[:, 0] * W
            yc_pixel = bbox_norm[:, 1] * H
            w_pixel  = bbox_norm[:, 2] * W
            h_pixel  = bbox_norm[:, 3] * H

            # stack to get final output shape [B, 4]
            bbox_coords_pixel = torch.stack([xc_pixel, yc_pixel, w_pixel, h_pixel], dim=1)  # [B, 4]

            return bbox_coords_pixel
        