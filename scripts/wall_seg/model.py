import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from torchvision.models._utils import IntermediateLayerGetter


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        backbone = resnet34(weights=None)
        return_layers = {"layer1": "c1", "layer2": "c2", "layer3": "c3", "layer4": "c4"}

        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        out_channels = 64

        # Lateral convs
        self.lateral4 = nn.Conv2d(512, out_channels, 1)
        self.lateral3 = nn.Conv2d(256, out_channels, 1)
        self.lateral2 = nn.Conv2d(128, out_channels, 1)
        self.lateral1 = nn.Conv2d(64, out_channels, 1)

        # Output conv
        self.output_conv = nn.Conv2d(out_channels, 1, 1)

    def forward(self, x: torch.Tensor):
        x = x.repeat(1, 3, 1, 1)

        features = self.backbone(x)
        c1, c2, c3, c4 = features["c1"], features["c2"], features["c3"], features["c4"]

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="nearest")

        out = self.output_conv(F.interpolate(p1, size=x.shape[2:], mode="bilinear"))
        return out


# Kendall style learnable parameters
# L_total = L1 + alpha * L2
# But instead, we go for L_total = L1 / 2*sigma1^2 + L2 / 2*sigma2*2 + log(sigma1) + log(sigma2)
# Here, the sigma's are learnable


class MultiLoss(nn.Module):
    def __init__(self):
        super().__init__()
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
    loss = F.binary_cross_entropy_with_logits(
        pred[:, :, 1:, :], pred[:, :, :-1, :]
    ) + F.binary_cross_entropy_with_logits(pred[:, :, :, 1:], pred[:, :, :, :-1])

    return loss
