"""ResNet-18 feature encoders used by all image and wavelet branches."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Encoder(nn.Module):
    """A ResNet-18 without its classification layer.

    Non-RGB inputs reuse the pretrained first convolution by averaging its RGB
    weights and repeating them across the requested number of channels.
    """

    output_dim = 512

    def __init__(self, in_channels: int, pretrained: bool = True) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        network = resnet18(weights=weights)
        if in_channels != 3:
            network.conv1 = self._adapt_first_convolution(
                network.conv1,
                in_channels,
                reuse_source_weights=pretrained,
            )
        network.fc = nn.Identity()
        self.network = network
        self.in_channels = in_channels

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected BCHW input, received shape {tuple(inputs.shape)}")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"Encoder expects {self.in_channels} channels, received {inputs.shape[1]}"
            )
        return self.network(inputs)

    @staticmethod
    def _adapt_first_convolution(
        source: nn.Conv2d,
        in_channels: int,
        reuse_source_weights: bool,
    ) -> nn.Conv2d:
        target = nn.Conv2d(
            in_channels=in_channels,
            out_channels=source.out_channels,
            kernel_size=source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            bias=False,
        )
        if reuse_source_weights:
            with torch.no_grad():
                mean_kernel = source.weight.mean(dim=1, keepdim=True)
                repeated_kernel = mean_kernel.repeat(1, in_channels, 1, 1)
                target.weight.copy_(repeated_kernel * (3.0 / in_channels))
        else:
            nn.init.kaiming_normal_(target.weight, mode="fan_out", nonlinearity="relu")
        return target


def build_encoder(
    backbone: str,
    in_channels: int,
    pretrained: bool,
) -> ResNet18Encoder:
    """Build a supported feature encoder."""

    normalized_name = backbone.lower().replace("-", "")
    if normalized_name != "resnet18":
        raise ValueError(f"Unsupported backbone '{backbone}'. Only resnet18 is available.")
    return ResNet18Encoder(in_channels=in_channels, pretrained=pretrained)
