"""
Facebook DINOv2 (2D ViT) classifier for individual CT slices.

Input:  [B, 1, H, W]  single-channel CT slice, values in [0, 1]
Output: [B, num_classes] logits

Gray→RGB conversion, resize to img_size (must be a multiple of 14), and
ImageNet normalisation all happen inside the model, so the external data
pipeline stays identical to every other baseline (single-channel CT patch).

Aggregation of per-slice scores into a 3D-level prediction is NOT done
here — it is the caller's responsibility at inference time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


class DINOv2SliceClassifier(nn.Module):
    """
    Args:
        model_name:       torch.hub name, e.g. 'dinov2_vitb14'.
        num_classes:      Number of output classes.
        img_size:         Spatial size fed to DINOv2.  Must be a multiple of 14.
                          168 (=12×14) is the nearest valid size to the CT native
                          160 px slice resolution.
        head_hidden_dims: MLP head hidden widths.  None → [embed_dim // 2].
        head_dropout:     Dropout inside the MLP head.
        local_weights:    Path to a locally-stored DINOv2 checkpoint (skips hub).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        num_classes: int = 2,
        img_size: int = 168,
        head_hidden_dims: list[int] | None = None,
        head_dropout: float = 0.1,
        local_weights: str | None = None,
        imagenet_norm: bool = True,
    ) -> None:
        super().__init__()
        assert img_size % 14 == 0, f"img_size must be a multiple of 14, got {img_size}"
        self.img_size = img_size

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=(local_weights is None),
        )
        if local_weights is not None:
            sd = torch.load(local_weights, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd.get("state_dict", sd))
            self.backbone.load_state_dict(sd, strict=False)
            print(f"[DINOv2SliceClassifier] loaded local weights from {local_weights}")

        embed_dim = self.backbone.embed_dim
        print(f"[DINOv2SliceClassifier] {model_name}  embed_dim={embed_dim}  img_size={img_size}")

        if head_hidden_dims is None:
            head_hidden_dims = [embed_dim // 2]

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for h in head_hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(head_dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*layers)

        self.imagenet_norm = imagenet_norm
        self.register_buffer("_mean", _IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("_std",  _IMAGENET_STD.view(1, 3, 1, 1))

    # ------------------------------------------------------------------
    # Interface expected by BaselineClassifierFinetuner
    # ------------------------------------------------------------------
    def encoder_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return list(self.head.parameters())

    def freeze_encoder(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_encoder(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()

    # ------------------------------------------------------------------
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1, H, W] → [B, 3, img_size, img_size] ImageNet-normalised"""
        x = x.float().expand(-1, 3, -1, -1).contiguous()   # grey → RGB
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.imagenet_norm:
            x = (x - self._mean.float()) / self._std.float()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, H, W] → logits: [B, num_classes]"""
        x = self._preprocess(x)
        feats = self.backbone.forward_features(x)["x_norm_clstoken"]  # [B, embed_dim]
        return self.head(feats)
