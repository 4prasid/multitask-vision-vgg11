"""
Reusable custom layers 

"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()
        self.p = p

        # check that p is a valid probability value between 0 and 1
        # (1 exclusive to avoid dropping all features and division by zero).
        if not (0 <= self.p < 1):
            raise ValueError(f"Dropout probability p must be between 0 and 1, but got {self.p}.")
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """

        # During training, apply dropout by randomly zeroing out some features
        # and scaling the remaining features to maintain the expected value.
        if self.training:
            if self.p == 0:
                return x  # No dropout applied for p=0, return input unchanged.
            else: 
                # Create a binary mask where each element is 1 with probability (1-p) and 0 with probability p
                mask = torch.bernoulli(torch.full_like(x, 1 - self.p))  # shape [B, C, H, W], values are 0 or 1
                mask = mask.float()  # convert to float for multiplication
                out = x * mask   # apply the dropout mask to the input 
                return out / (1 - self.p)  # apply scaling to maintain the expected value of the features (inverted dropout)
        # During evaluation, return the input unchanged (no dropout applied).
        else:
            return x
