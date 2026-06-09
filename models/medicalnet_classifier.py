"""ResNet-18 with MedicalNet pretrained weights for 3-D CT classification.

Weight loading follows the same approach used in the BWT baseline
(Differential-Diagnosis-of-BWT/classifier/models/resnet.py):
strip the "module." prefix from the MedicalNet checkpoint, then load
directly into MONAI's ResNetFeatures whose key names are compatible.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import monai.networks.nets as nets


class _GetLast(nn.Module):
    """Extract the last element from a tuple/list (MONAI ResNetFeatures output)."""
    def forward(self, x):
        return x[-1] if isinstance(x, (list, tuple)) else x


_MEDICALNET_URLS = {
    18:  "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/resolve/main/resnet_18.pth",
    34:  "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet34/resolve/main/resnet_34.pth",
    50:  "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50/resolve/main/resnet_50.pth",
    101: "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet101/resolve/main/resnet_101.pth",
}

_WEIGHTS_CACHE = os.environ.get(
    "MEDICALNET_WEIGHTS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "model_weights"),
)


class MedicalNetR18Classifier(nn.Module):
    """
    ResNet-18 3-D backbone pretrained with MedicalNet + MLP classification head.

    Architecture mirrors the BWT baseline:
        ResNetFeatures → GetLast → AdaptiveAvgPool3d(1) → Flatten → MLP head

    Pretrained weights are downloaded once to ``_WEIGHTS_CACHE`` via wget
    (same pattern as the BWT repo) and reused on subsequent calls.

    Args:
        in_channels:      Input channels (1 for CT-only).
        num_classes:      Number of output classes.
        pretrained:       If True, load MedicalNet ResNet-18 weights.
        head_hidden_dims: Hidden layer widths in the MLP head.
        head_dropout:     Dropout probability inside the MLP head.
    """

    ENCODER_OUT_CH = 512   # ResNet-18 layer4 output channels

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        pretrained: bool = True,
        head_hidden_dims: list = [256],
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = nets.ResNetFeatures(
            model_name="resnet18",
            spatial_dims=3,
            in_channels=in_channels,
            pretrained=False,
        )

        if pretrained:
            self._load_medicalnet_weights(in_channels)

        self.pool = nn.Sequential(
            _GetLast(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
        )

        head_layers: list[nn.Module] = []
        in_dim = self.ENCODER_OUT_CH
        for h in head_hidden_dims:
            head_layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
            ]
            in_dim = h
        head_layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*head_layers)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def _load_medicalnet_weights(self, in_channels: int) -> None:
        os.makedirs(_WEIGHTS_CACHE, exist_ok=True)
        weights_path = os.path.join(_WEIGHTS_CACHE, "resnet18_3d_medicalnet.pth")
        url = _MEDICALNET_URLS[18]

        if not os.path.exists(weights_path):
            print(f"[MedicalNetR18] Downloading MedicalNet weights → {weights_path}")
            ret = os.system(f"wget -q -O {weights_path} {url}")
            if ret != 0 or not os.path.exists(weights_path):
                print("[MedicalNetR18] WARNING: download failed; skipping pretrained init.")
                return

        checkpoint = torch.load(weights_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)

        # Strip "module." prefix (MedicalNet was saved with DataParallel)
        new_state: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            new_state[k[7:] if k.startswith("module.") else k] = v

        # Adapt conv1 if in_channels != 1 (same logic as BWT baseline)
        if "conv1.weight" in new_state and new_state["conv1.weight"].shape[1] != in_channels:
            w = new_state["conv1.weight"]
            new_state["conv1.weight"] = w.repeat(1, in_channels, 1, 1, 1) / in_channels

        missing, unexpected = self.encoder.load_state_dict(new_state, strict=False)
        n_loaded = len(new_state) - len(unexpected)
        print(
            f"[MedicalNetR18] pretrained weights loaded "
            f"| {n_loaded}/{len(new_state)} keys matched "
            f"| missing={len(missing)} | unexpected={len(unexpected)}"
        )

    # ------------------------------------------------------------------
    # Encoder / head parameter helpers
    # ------------------------------------------------------------------
    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        # pool has no learnable params; head does
        return list(self.head.parameters())

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.encoder.train()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, D, H, W] → logits: [B, num_classes]"""
        feat = self.pool(self.encoder(x))   # [B, 512]
        return self.head(feat)
