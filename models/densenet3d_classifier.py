"""MONAI DenseNet-121 (3-D) wrapped for binary / multiclass CT classification."""
from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121


class DenseNet3DClassifier(nn.Module):
    """
    3-D DenseNet-121 from MONAI for volumetric CT classification.

    MONAI's DenseNet121 already includes a global adaptive-average-pool and a
    final fully-connected layer, so this wrapper is intentionally thin.  The
    internal split between ``features`` (dense blocks) and ``class_layers``
    (relu + pool + flatten + linear) is exposed so that BaselineClassifierFinetuner
    can apply separate learning rates or freeze the encoder independently.

    Args:
        in_channels:  Number of input channels (1 for CT-only).
        num_classes:  Number of output classes.
        dropout_prob: Dropout probability applied inside the dense blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            dropout_prob=dropout_prob,
        )

    # ------------------------------------------------------------------
    # Encoder / head parameter helpers
    # ------------------------------------------------------------------
    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.model.features.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.model.class_layers.parameters())

    def freeze_encoder(self) -> None:
        for p in self.model.features.parameters():
            p.requires_grad = False
        self.model.features.eval()

    def unfreeze_encoder(self) -> None:
        for p in self.model.features.parameters():
            p.requires_grad = True
        self.model.features.train()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, D, H, W] → logits: [B, num_classes]"""
        return self.model(x)
