"""
2D MedDINOv3 classifier for colon cancer vs. diverticulitis.

Forward flow:
  volume [B, N, H, W]  (N grayscale axial slices from the colon crop)
  → expand to 3-ch, resize to img_size → [B*N, 3, img_size, img_size]
  → MedDINOv3 ViT-B/16 CLS token  → [B*N, 768]
  → reshape + mean over N slices   → [B, 768]
  → MLP head                        → [B, num_classes]

Backbone: MedDINOv3 ViT-B/16 pretrained on CT-3M
  checkpoint: /data/benchaaben/experiments/meddinov3_vitb16_ct3m.pth
  paper: https://arxiv.org/abs/2509.02379
"""

import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_DINOV3_PATH = os.environ.get(
    "MEDDINOV3_CODE_PATH",
    "/path/to/MedDINOv3/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3",
)
_MEDDINOV3_CKPT = os.environ.get(
    "MEDDINOV3_CKPT_PATH",
    "/path/to/meddinov3_vitb16_ct3m.pth",
)


def _load_meddinov3_backbone(checkpoint_path: str = _MEDDINOV3_CKPT) -> nn.Module:
    if _DINOV3_PATH not in sys.path:
        sys.path.insert(0, _DINOV3_PATH)
    from dinov3.models import vision_transformer as vits

    model = vits.vit_base(patch_size=16)
    if not Path(checkpoint_path).exists():
        print(f"[MedDINOv3] Checkpoint not found: {checkpoint_path} — random init")
        return model

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher_sd = ckpt.get("teacher", ckpt)
    prefix = "backbone."
    sd = {k[len(prefix):]: v for k, v in teacher_sd.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[MedDINOv3] Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"[MedDINOv3] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    print(f"[MedDINOv3] Loaded backbone from {checkpoint_path}")
    return model


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], num_classes: int, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim, *hidden_dims, num_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MedDINOv3_2D_Classifier(nn.Module):
    """
    2D MedDINOv3 slice-based classifier with mean pooling across slices.

    Args:
        num_classes:   Number of output classes (2 for binary).
        head_hidden:   Hidden dims for the MLP classification head.
        head_dropout:  Dropout in the MLP head.
        img_size:      Slices are resized to (img_size, img_size) before encoding.
        freeze_backbone: If True, backbone weights are frozen (linear probe mode).
        checkpoint_path: Path to meddinov3_vitb16_ct3m.pth.
    """

    EMBED_DIM = 768

    def __init__(
        self,
        num_classes: int = 2,
        head_hidden: Sequence[int] = (256, 128),
        head_dropout: float = 0.1,
        img_size: int = 224,
        freeze_backbone: bool = False,
        checkpoint_path: str = _MEDDINOV3_CKPT,
    ):
        super().__init__()
        self.img_size = img_size
        self.backbone = _load_meddinov3_backbone(checkpoint_path)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = MLPHead(self.EMBED_DIM, list(head_hidden), num_classes, head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, H, W]  — batch of volumes, N grayscale axial slices each
        Returns:
            logits: [B, num_classes]
        """
        B, N, H, W = x.shape

        # Resize slices to img_size and expand grayscale → 3 channels
        flat = x.view(B * N, 1, H, W)
        if H != self.img_size or W != self.img_size:
            flat = F.interpolate(flat, size=(self.img_size, self.img_size),
                                 mode="bilinear", align_corners=False)
        flat = flat.expand(-1, 3, -1, -1)  # [B*N, 3, img_size, img_size]

        feats = self.backbone(flat)         # [B*N, 768]
        feats = feats.view(B, N, -1)        # [B, N, 768]
        pooled = feats.mean(dim=1)          # [B, 768]
        return self.head(pooled)            # [B, num_classes]
