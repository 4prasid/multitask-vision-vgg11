"""
Classification components
"""

import torch
import torch.nn as nn
from .vgg11 import *
from .layers import *


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5, encoder=None):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_p = dropout_p

        # vgg encoder to extract features from the input image

        if encoder is None:
            self.encoder = VGG11Encoder(in_channels=in_channels)
        else:
            self.encoder = encoder

        # classification head to produce class logits from the bottleneck features
        # we will flatten this to [B, 512*7*7] and then apply a linear layer to get the class logits.

        # Assumes input size = 224 x 224 -> output = 7 x 7 after 5 pooling layerss
        c_out = 7 
        
        # flatten to FC1(4096) -> FC2(4096) -> FC3(num_classes) with dropout in between

        # flatten to FC1(4096) 
        self.fc1 = nn.Sequential(
            nn.Linear(512 * c_out * c_out, 4096),  # input size is 512*7*7 (if i/p image is 224 * 224), output size is 4096
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p))  # apply custom dropout with the specified dropout
        
        # FC2(4096)
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),  
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p))  
        
        # final linear layer to get class logits
        self.fc3 = nn.Linear(4096, num_classes)  


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """

        # pass through the encoder to get bottleneck tensor
        bottleneck = self.encoder(x)  # shape [B, 512, 7, 7]

        if isinstance(bottleneck, tuple):
            bottleneck = bottleneck[0]  # if encoder returns (bottleneck, features), we only need the bottleneck for classification

        # flatten the bottleneck features to shape [B, 512*7*7] preserving the batch dimension
        bottleneck_flat = bottleneck.reshape(bottleneck.size(0), -1)  # shape [B, 512*7*7]

        # pass through the classifier head to get class logits

        # [B, 512*7*7] -> [B, 4096] 
        fc_1 = self.fc1(bottleneck_flat)  

        # [B, 4096] -> [B, 4096]
        fc_2 = self.fc2(fc_1)  

        # [B, 4096] -> [B, num_classes]
        logits = self.fc3(fc_2)  

        return logits