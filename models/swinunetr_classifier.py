"""SwinUNETR encoder + MLP head for 3D CT classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class SwinUNETRClassifier(nn.Module):
    """
    MONAI SwinUNETR encoder with a global-avg-pool → MLP classification head.

    The decoder portion of SwinUNETR is unused; only swinViT (the transformer
    encoder) is kept for feature extraction.  The deepest encoder feature map
    (index 4 from swinViT, shape [B, 16*feature_size, d/32, h/32, w/32]) is
    pooled and projected to class logits.

    Compatible with MONAI >= 1.5 (no img_size required; spatial dims are
    handled dynamically inside SwinTransformer).

    Args:
        in_channels:       Input channels (1 for CT-only).
        num_classes:       Number of output classes.
        feature_size:      SwinUNETR base feature size. Bottleneck dim =
                           16 × feature_size (768 for feature_size=48).
        pretrained_weights: Path to a pretrained .pth/.ckpt file.  Accepts the
                           MONAI SSL checkpoint (keys prefixed with "swinViT.")
                           or a full SwinUNETR checkpoint.  None = random init.
        head_hidden_dims:  Hidden layer widths in the MLP head.
        head_dropout:      Dropout probability inside the MLP head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        feature_size: int = 48,
        img_size: tuple = (96, 96, 96),
        pretrained_weights: str | None = None,
        head_hidden_dims: list = [256, 128],
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from monai.networks.nets import SwinUNETR

        # out_channels is required by MONAI but the decoder is unused.
        self.swin = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=2,
            feature_size=feature_size,
            use_checkpoint=True,
        )
        self._encoder_dim = 16 * feature_size  # e.g. 768 for feature_size=48

        if pretrained_weights is not None:
            self._load_pretrained(pretrained_weights)

        self.pool = nn.AdaptiveAvgPool3d(1)

        layers: list[nn.Module] = []
        in_dim = self._encoder_dim
        for h in head_hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(head_dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def _load_pretrained(self, weights_path: str) -> None:
        state = torch.load(weights_path, map_location="cpu")
        for key in ("state_dict", "model", "net"):
            if key in state:
                state = state[key]
                break

        # MONAI SSL checkpoint: keys are "swinViT.<rest>"
        swin_state = {
            k[len("swinViT."):]: v
            for k, v in state.items()
            if k.startswith("swinViT.")
        }
        if swin_state:
            missing, unexpected = self.swin.swinViT.load_state_dict(swin_state, strict=False)
            target = "swinViT"
        else:
            # Full SwinUNETR checkpoint (e.g. finetuned seg model)
            missing, unexpected = self.swin.load_state_dict(state, strict=False)
            target = "SwinUNETR"
        print(
            f"[SwinUNETRClassifier] pretrained {target} loaded "
            f"| missing={len(missing)} | unexpected={len(unexpected)}"
        )

    # ------------------------------------------------------------------
    # Encoder / head parameter helpers (used by BaselineClassifierFinetuner)
    # ------------------------------------------------------------------
    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.swin.swinViT.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.pool.parameters()) + list(self.head.parameters())

    def freeze_encoder(self) -> None:
        for p in self.swin.swinViT.parameters():
            p.requires_grad = False
        self.swin.swinViT.eval()

    def unfreeze_encoder(self) -> None:
        for p in self.swin.swinViT.parameters():
            p.requires_grad = True
        self.swin.swinViT.train()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, D, H, W] → logits: [B, num_classes]"""
        # swinViT returns a 5-tuple of progressively downsampled feature maps.
        # Index 4 is the deepest: [B, 16*feature_size, D/32, H/32, W/32].
        hidden = self.swin.swinViT(x, normalize=self.swin.normalize)
        feat = self.pool(hidden[-1]).flatten(1)   # [B, encoder_dim]
        return self.head(feat)
