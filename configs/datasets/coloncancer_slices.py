"""
ColonCancerSlices — returns one 2D axial slice per training sample.

Training flow (per __getitem__):
  1. Pick a random volume from the split.
  2. Take a random 3D crop of `patch_size` (same as 3D baselines).
  3. Extract one random axial slice → [1, H, W].
  4. Return {"source": [1, H, W], "target": volume_label}.

Loss is applied per-slice; all slices from the same volume share that
volume's label (no per-slice annotation needed).

Inference: call `predict_volume()` in inference/predict_volume_dinov2_2d.py,
which samples N slices and averages their softmax probabilities.
"""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from utils.functions_utils import load_volume, pad_to_shape

_IMAGE_CANDIDATES = [
    "{:03d}.npz", "{:03d}.b2nd", "{}.npz", "{}.b2nd",
    "{:03d}.nii.gz", "{}.nii.gz",
]


def _find_file(base: Path, uid, patterns):
    for pat in patterns:
        p = base / pat.format(int(uid))
        if p.exists():
            return p
    return None


class ColonCancerSlices(data.Dataset):
    """
    Args:
        dataset_name:          Dataset folder name (e.g. "Dataset115_CC").
        split:                 "train" | "val" | "test".
        patch_size:            [D, H, W] 3D crop drawn before slicing.
                               None = use the full volume depth, full H/W.
        num_patches_per_epoch: Number of slices drawn per epoch (controls
                               epoch length).  None = one pass over subjects.
        augment:               Random H/V flips during training.
        seed:                  Base RNG seed.
    """

    def __init__(
        self,
        dataset_name: str = "Dataset109_CC",
        split: str = "train",
        patch_size: list[int] | None = None,
        num_patches_per_epoch: int | None = None,
        num_slices_per_volume: int = 8,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.patch_size = patch_size
        self.num_patches_per_epoch = num_patches_per_epoch
        self.num_slices_per_volume = num_slices_per_volume
        self.augment = augment and (split == "train")
        self.seed = seed
        self.epoch = 0

        root = Path(os.environ.get("root_path", "/data/colon_cancer/CC_Detection"))
        scaled_dir = "rescaledTs" if split == "test" else "rescaledTr"
        self.images_path = root / "pp_data" / dataset_name / scaled_dir
        splits_file = root / "raw_data" / dataset_name / "splits.csv"

        df = pd.read_csv(splits_file)
        if split is not None:
            df = df[df["Split"] == split]

        self.samples: list[tuple] = []
        for _, row in df.iterrows():
            uid    = row["UID"]
            target = int(row["target"])
            path   = _find_file(self.images_path, uid, _IMAGE_CANDIDATES)
            if path is None:
                print(f"[ColonCancerSlices] Warning: image not found for UID={uid}, skipping.")
                continue
            self.samples.append((uid, path, target))

        print(f"[ColonCancerSlices] split={split}  subjects={len(self.samples)}"
              f"  epoch_len={num_patches_per_epoch or len(self.samples)}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_patches_per_epoch if self.num_patches_per_epoch else len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id if worker_info else 0
        rng = np.random.default_rng(self.seed + self.epoch * 100_000 + worker_id * 1_000 + idx)

        # Random volume
        uid, img_path, target = self.samples[int(rng.integers(0, len(self.samples)))]

        vol = load_volume(img_path)
        vol = np.transpose(vol, (2, 1, 0))  # → [D, H, W]
        D, H, W = vol.shape

        # Random 3D crop (same logic as BaseDataset in coloncancer.py)
        if self.patch_size is not None:
            ps = list(self.patch_size)

            def start(dim, pdim):
                return int(rng.integers(0, max(1, dim - pdim + 1))) if dim > pdim else 0

            z, x, y = start(D, ps[0]), start(H, ps[1]), start(W, ps[2])
            crop = vol[z:z+ps[0], x:x+ps[1], y:y+ps[2]]

            if crop.shape != tuple(ps):
                crop, _ = pad_to_shape(crop, ps)
        else:
            crop = vol  # use full volume

        # Sample K random axial slices from the crop (with replacement if crop is shallow)
        K = self.num_slices_per_volume
        depth = crop.shape[0]
        indices = rng.integers(0, depth, size=K)  # K random indices, with replacement
        slices = crop[indices]  # [K, H, W]

        img_t = torch.from_numpy(slices.copy()).unsqueeze(1).float()  # [K, 1, H, W]

        if self.augment:
            if rng.random() > 0.5:
                img_t = img_t.flip(-1)   # horizontal flip (same flip for all K slices)
            if rng.random() > 0.5:
                img_t = img_t.flip(-2)   # vertical flip

        return {
            "source": img_t,                                    # [K, 1, H, W]
            "target": torch.tensor(target, dtype=torch.long),  # scalar — same for all K
            "uid":    str(uid),
        }
