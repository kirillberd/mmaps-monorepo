import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0Embed(nn.Module):

    def __init__(self, l2_normalize: bool = True):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        base.eval()

        self.features = base.features
        self.avgpool = base.avgpool
        self.l2_normalize = l2_normalize

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)  # [B, 1280, H', W']
        x = self.avgpool(x)  # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)  # [B, 1280]

        if self.l2_normalize:
            # Явная L2-нормировка (экспортится в простые ONNX-операции)
            # x := x / sqrt(sum(x^2) + eps)
            eps = 1e-12
            norm = torch.sqrt(torch.clamp((x * x).sum(dim=1, keepdim=True), min=eps))
            x = x / norm

        return x
