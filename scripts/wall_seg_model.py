"""
model.py

Defines the FPN model, loss functions, and initialization utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from torchvision.models._utils import IntermediateLayerGetter


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) with a ResNet-34 backbone.
    Designed for single-channel (grayscale) input.
    """

    def __init__(self, out_channels=64):
        super(FPN, self).__init__()

        # Load a ResNet-34 backbone, weights are not pre-trained
        backbone = resnet34(weights=None)
        return_layers = {"layer1": "c1", "layer2": "c2", "layer3": "c3", "layer4": "c4"}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # Lateral convs (reduce channels)
        self.lateral4 = nn.Conv2d(512, out_channels, 1)
        self.lateral3 = nn.Conv2d(256, out_channels, 1)
        self.lateral2 = nn.Conv2d(128, out_channels, 1)
        self.lateral1 = nn.Conv2d(64, out_channels, 1)

        # Output convolution
        self.output_conv = nn.Conv2d(out_channels, 1, 1)

    def forward(self, x: torch.Tensor):
        # Input 'x' is [B, 1, H, W] (grayscale)
        # ResNet expects [B, 3, H, W], so we repeat the channel
        
        x = x.repeat(1, 3, 1, 1)
        x = x.float()
    
        # Get backbone features
        features = self.backbone(x)
        c1, c2, c3, c4 = features["c1"], features["c2"], features["c3"], features["c4"]

        # FPN top-down pathway
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="nearest")

        # Upsample to original size and predict
        out = self.output_conv(F.interpolate(p1, size=x.shape[2:], mode="bilinear"))
        return out


class MultiLoss(nn.Module):
    """
    Custom multi-task loss using uncertainty weighting (as in the notebook).
    L_total = L_bce / (2*sigma1^2) + L_aff / (2*sigma2^2) + log(sigma1) + log(sigma2)
    """

    def __init__(self):
        super().__init__()
        # Learnable log-variance parameters
        self.log_var_bce = nn.Parameter(torch.zeros(1))
        self.log_var_aff = nn.Parameter(torch.zeros(1))

    def forward(self, bce_loss, aff_loss):
        # Uncertainty weighting
        loss = (
            1.0 / (2 * torch.exp(self.log_var_bce))
        ) * bce_loss + self.log_var_bce * 0.5
        loss += (
            1.0 / (2 * torch.exp(self.log_var_aff))
        ) * aff_loss + self.log_var_aff * 0.5
        return loss


def affinity_loss(pred):
    """
    Calculates the affinity loss (pixel-to-pixel similarity)
    between adjacent pixels in the prediction.
    """
    # Horizontal affinity
    loss_h = F.binary_cross_entropy_with_logits(pred[:, :, :, 1:], pred[:, :, :, :-1])
    # Vertical affinity
    loss_v = F.binary_cross_entropy_with_logits(pred[:, :, 1:, :], pred[:, :, :-1, :])

    return loss_h + loss_v


def init_model(hub_cache_dir: str, device: torch.device):
    """
    Initializes the model, sets the torch hub directory,
    moves model to device, and wraps with DataParallel if >1 GPU.

    Args:
        hub_cache_dir (str): Path to cache torch hub models.
        device (torch.device): The target device (e.g., 'cuda', 'cpu').

    Returns:
        torch.nn.Module: The initialized model.
    """
    print(f"Setting torch hub cache to: {hub_cache_dir}")
    torch.hub.set_dir(hub_cache_dir)

    model = FPN()

    # Move to device first
    model.to(device)

    # Check for multi-GPU and wrap with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    return model
