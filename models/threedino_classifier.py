"""
3DINO multitask model — classification head on a 3D DINOv2 ViT backbone.

Requires the 3dino_repo to be on the Python path.  Set the environment variable:
    export THREEDINO_REPO=/data/benchaaben/3dino_repo

Backbone choices:
    3dino     — ViT-Large pretrained by AICONSlab (3dino_vit_weights.pth)
    meddinov3 — ViT-Base  pretrained on CT-3M     (meddinov3_vitb16_ct3m.pth)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_THREEDINO_REPO = os.environ.get("THREEDINO_REPO", "/data/benchaaben/3dino_repo")


def _ensure_repo_on_path():
    if _THREEDINO_REPO not in sys.path:
        sys.path.insert(0, _THREEDINO_REPO)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a trainable low-rank delta."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / rank
        d_in, d_out = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def inject_lora(backbone, num_layers: int, rank: int, alpha: float):
    """Replace attn.qkv in the last `num_layers` transformer blocks with LoRALinear."""
    all_blocks = [blk for chunk in backbone.blocks for blk in chunk if hasattr(blk, "attn")]
    for blk in all_blocks[-num_layers:]:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, rank=rank, alpha=alpha)
    n_lora = sum(
        p.numel() for blk in all_blocks[-num_layers:]
        for p in blk.attn.qkv.parameters() if p.requires_grad
    )
    print(f"LoRA: last {num_layers} layers  rank={rank}  alpha={alpha}  "
          f"trainable params={n_lora:,}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ThreeDINOClassifier(nn.Module):
    """
    3DINO backbone with a linear classification head on the CLS token.

    Args:
        backbone:    3DINO ViT backbone (ViT-L or ViT-B).
        num_classes: Number of output classes.
    """

    def __init__(self, backbone: nn.Module, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.cls_head = nn.Sequential(
            nn.LayerNorm(backbone.embed_dim),
            nn.Linear(backbone.embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        return self.cls_head(feats["x_norm_clstoken"])

    def predict_volume(
        self,
        x_full: torch.Tensor,
        patch_size: int,
        sw_batch_size: int = 2,
        overlap: float = 0.5,
    ) -> torch.Tensor:
        """
        Tile overlapping patches over a full volume, pool CLS tokens, return logits.

        Args:
            x_full: [1, 1, D, H, W] on device
        Returns:
            cls_logits: [1, num_classes] on CPU
        """
        # Pad any dimension smaller than patch_size (mirrors infer_cls.py behaviour)
        D, H, W = x_full.shape[2:]
        pad_d = max(0, patch_size - D)
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        if pad_d or pad_h or pad_w:
            x_full = F.pad(x_full, (0, pad_w, 0, pad_h, 0, pad_d))

        stride = patch_size // 2
        cls_tokens: list[torch.Tensor] = []

        with torch.no_grad():
            for z0 in range(0, x_full.shape[2], stride):
                for y0 in range(0, x_full.shape[3], stride):
                    for x0 in range(0, x_full.shape[4], stride):
                        patch = x_full[
                            :, :,
                            z0:z0 + patch_size,
                            y0:y0 + patch_size,
                            x0:x0 + patch_size,
                        ]
                        if any(s < patch_size for s in patch.shape[2:]):
                            continue
                        feats = self.backbone.forward_features(patch)
                        cls_tokens.append(feats["x_norm_clstoken"].detach())

        pooled = torch.cat(cls_tokens, dim=0).mean(dim=0, keepdim=True)
        return self.cls_head(pooled).cpu()


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def _load_meddinov3_weights(backbone: nn.Module, weights_path: str) -> None:
    chkpt = torch.load(weights_path, map_location="cpu")
    if isinstance(chkpt, dict) and any(k.startswith("backbone.") for k in chkpt):
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in chkpt.items()
            if not any(tag in k for tag in ("ibot", "dino_head", "student"))
        }
    else:
        state_dict = chkpt
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    print(f"[3DINO] MedDINOv3 weights loaded  missing={len(missing)}  "
          f"unexpected={len(unexpected)}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_threedino_classifier(
    weights_path: str | None,
    backbone_name: str = "3dino",
    num_classes: int = 2,
    lora_layers: int = 0,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
) -> ThreeDINOClassifier:
    """
    Build a ThreeDINOClassifier and optionally inject LoRA.

    Args:
        weights_path:  Path to backbone .pth file (3dino_vit_weights.pth or
                       meddinov3_vitb16_ct3m.pth).  None = random init.
        backbone_name: "3dino" (ViT-L) or "meddinov3" (ViT-B).
        num_classes:   Number of output classes.
        lora_layers:   Inject LoRA into last N transformer layers (0 = off).
        lora_rank:     LoRA rank.
        lora_alpha:    LoRA alpha scaling.
    """
    _ensure_repo_on_path()
    from dinov2.models import build_model
    from dinov2.utils.utils import load_pretrained_weights

    if backbone_name == "meddinov3":
        class _Args:
            arch = "vit_base_3d"; patch_size = 16; layerscale = 1e-5
            ffn_layer = "mlp"; block_chunks = 1; qkv_bias = True
            proj_bias = True; ffn_bias = True; drop_path_rate = 0.0
            drop_path_uniform = True

        backbone, _ = build_model(_Args(), only_teacher=True, img_size=112)
        if weights_path:
            _load_meddinov3_weights(backbone, weights_path)
        else:
            print("[3DINO] No weights provided — random ViT-B init")

    else:  # 3dino (ViT-Large, default)
        class _Args:
            arch = "vit_large_3d"; patch_size = 16; layerscale = 1e-5
            ffn_layer = "mlp"; block_chunks = 4; qkv_bias = True
            proj_bias = True; ffn_bias = True; drop_path_rate = 0.0
            drop_path_uniform = True

        backbone, _ = build_model(_Args(), only_teacher=True, img_size=112)
        if weights_path:
            load_pretrained_weights(backbone, weights_path, "teacher")
        else:
            print("[3DINO] No weights provided — random ViT-L init")

    if lora_layers > 0:
        inject_lora(backbone, lora_layers, lora_rank, lora_alpha)

    return ThreeDINOClassifier(backbone, num_classes=num_classes)
