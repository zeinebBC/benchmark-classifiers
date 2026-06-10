"""
Unified inference script for all classifiers on Dataset109_CC.

Supported model types:
    densenet3d   – 3-D DenseNet-121          (BaselineClassifierFinetuner)
    medicalnet   – MedicalNet ResNet-18       (BaselineClassifierFinetuner)
    swinunetr    – SwinUNETR encoder + MLP   (BaselineClassifierFinetuner)
    dinov2_3d    – 3-D DINOv2 Primus        (DINOv2ClassifierFinetuner)
    meddinov3_2d – 2-D MedDINOv3 slice stack (MedDINOv3ClassifierFinetuner)

Inference strategies
--------------------
densenet3d / medicalnet
    Fully convolutional + AdaptiveAvgPool3d: the full colon-cropped volume is
    fed in a single forward pass.  No spatial manipulation needed.

swinunetr / dinov2_3d
    Fixed-size window required (96³ and 160³ respectively).  Overlapping
    patches are extracted with configurable overlap, the model runs on each
    patch independently, and the logits are averaged across all patches.
    Volumes smaller than the window are zero-padded symmetrically.

meddinov3_2d
    Slice-based 2-D model; single forward pass on the pre-sampled slice stack.

Usage examples
--------------
    # DenseNet / MedicalNet — full volume
    python inference_unified.py --model_type densenet3d \\
        --checkpoint /path/to/best.ckpt --split val

    # SwinUNETR — sliding window
    python inference_unified.py --model_type swinunetr \\
        --checkpoint /path/to/best.ckpt --split val \\
        --patch_size 96 96 96 --overlap 0.5 --sw_batch_size 4

    # DINOv2-3D — sliding window
    python inference_unified.py --model_type dinov2_3d \\
        --checkpoint /path/to/best.ckpt --split val \\
        --patch_size 160 160 160 --overlap 0.5 --sw_batch_size 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("root_path", "/data/colon_cancer/CC_Detection")
os.environ.setdefault("auto_seg", "/data/colon_cancer/totalseg/total")

from utils.functions_utils import load_volume

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_data(args) -> None:
    """
    Run the three-step preprocessing pipeline on raw NIfTI data.

    Steps
    -----
    1. Crop to ROI (colon or lesion mask defines the bounding box)
    2. Resample to isotropic spacing
    3. CT windowing [-100, 500] + normalise → .npz

    Crop modes
    ----------
    colon   Use TotalSegmentator colon masks (--seg_dir must point to the
            directory containing {uid}.nii.gz colon segmentations).
    lesion  Use tumour / lesion masks (--seg_dir points to the directory
            containing {uid}.nii.gz lesion segmentations).

    In both cases the same mask file drives the bounding-box crop; the only
    difference is which directory --seg_dir points to.
    """
    from utils.cropping    import batch_crop_and_save
    from utils.resampling  import batch_resample_and_save
    from utils.normalizing import process_and_window_dataset

    # auto_seg env var is read by batch_crop_and_save to locate the crop mask
    os.environ["auto_seg"] = str(args.seg_dir)

    split_suffix = "Ts" if args.split == "test" else "Tr"
    raw   = Path(args.raw_dir)
    pp    = Path(os.environ.get("root_path", "/data/colon_cancer/CC_Detection")) \
            / "pp_data" / args.dataset

    images_dir = raw / f"images{split_suffix}"

    if not images_dir.exists():
        raise FileNotFoundError(f"Raw images not found: {images_dir}")

    cropped_dir   = pp / f"raw_cropped{split_suffix}"
    resampled_dir = pp / f"resampled{split_suffix}"
    rescaled_dir  = pp / f"rescaled{split_suffix}"

    print(f"\n[preprocess] crop_mode={args.crop_mode}  seg_dir={args.seg_dir}")
    print(f"[preprocess] Step 1/3 — cropping  →  {cropped_dir}")
    batch_crop_and_save(
        images_dir = images_dir,
        output_dir = cropped_dir,
        margin_min = 20,
        split      = split_suffix,
    )

    print(f"[preprocess] Step 2/3 — resampling {args.resample_spacing}mm  →  {resampled_dir}")
    batch_resample_and_save(
        root_dir       = cropped_dir,
        output_dir     = resampled_dir,
        target_spacing = tuple(args.resample_spacing),
        split          = split_suffix,
    )

    print(f"[preprocess] Step 3/3 — windowing [-100, 500]  →  {rescaled_dir}")
    process_and_window_dataset(
        images_dir = resampled_dir / "images_resampled",
        output_dir = rescaled_dir,
        window_min = -100,
        window_max =  500,
        split      = split_suffix,
    )
    print(f"[preprocess] Done. Preprocessed volumes at {rescaled_dir}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset  (loads full volumes — no cropping)
# ──────────────────────────────────────────────────────────────────────────────

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


class InferenceVolumeDataset(data.Dataset):
    """
    Loads full preprocessed 3-D volumes for inference.

    No cropping is applied here — spatial handling is done per-model in
    run_inference().  Every item has a potentially different spatial size,
    so the DataLoader must use batch_size=1.

    Returns dicts:
        "source"  – [1, D, H, W] float32 in [0, 1]
        "target"  – scalar long tensor
        "uid"     – subject UID string
    """

    def __init__(self, dataset_name: str, split: str,
                 root_path: str | None = None,
                 images_dir: str | None = None):
        root = Path(root_path or os.environ.get(
            "root_path", "/data/colon_cancer/CC_Detection"))
        if images_dir is not None:
            images_dir = Path(images_dir)
        else:
            images_dir = (
                root / "pp_data" / dataset_name / "rescaledTs"
                if split == "test"
                else root / "pp_data" / dataset_name / "rescaledTr"
            )
        splits_file = root / "raw_data" / dataset_name / "splits.csv"
        df = pd.read_csv(splits_file)
        df = df[df["Split"] == split]

        self.samples: list[tuple] = []
        for _, row in df.iterrows():
            uid    = row["UID"]
            target = int(row["target"])
            path   = _find_file(images_dir, uid, _IMAGE_CANDIDATES)
            if path is None:
                print(f"[InferenceVolumeDataset] WARNING: not found UID={uid}")
                continue
            self.samples.append((str(uid), path, target))

        print(f"[InferenceVolumeDataset] split={split}: "
              f"{len(self.samples)} subjects")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        uid, img_path, target = self.samples[idx]
        vol = load_volume(img_path)
        vol = np.transpose(vol, (2, 1, 0))            # match ColonCancer axis order
        vol_t = torch.from_numpy(vol).float().unsqueeze(0)  # [1, D, H, W]
        return {
            "source": vol_t,
            "target": torch.tensor(target, dtype=torch.long),
            "uid":    uid,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Sliding-window classification
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sliding_window_classify(
    model:        nn.Module,
    volume:       torch.Tensor,   # [1, C, D, H, W]  on device
    patch_size:   list[int],      # [pD, pH, pW]
    overlap:      float = 0.5,
    sw_batch_size: int  = 4,
) -> torch.Tensor:
    """
    Tile overlapping patches over a 3-D volume, run the classifier on each
    patch, and return the mean logits [1, num_classes] on CPU.

    Volumes smaller than patch_size in any dimension are symmetrically
    zero-padded before tiling.
    """
    pD, pH, pW = patch_size
    D, H, W    = volume.shape[-3:]

    # Symmetric zero-pad so every dim is >= patch size
    pad_d = max(0, pD - D)
    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)
    if pad_d or pad_h or pad_w:
        volume = F.pad(volume, (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2,
        ))
    D, H, W = volume.shape[-3:]

    sD = max(1, int(pD * (1.0 - overlap)))
    sH = max(1, int(pH * (1.0 - overlap)))
    sW = max(1, int(pW * (1.0 - overlap)))

    def _starts(size: int, patch: int, stride: int) -> list[int]:
        starts = list(range(0, size - patch + 1, stride))
        if not starts or starts[-1] + patch < size:
            starts.append(size - patch)
        return starts

    patches = [
        volume[..., z:z + pD, y:y + pH, x:x + pW]
        for z in _starts(D, pD, sD)
        for y in _starts(H, pH, sH)
        for x in _starts(W, pW, sW)
    ]
    n_patches = len(patches)

    all_logits: list[torch.Tensor] = []
    for i in range(0, n_patches, sw_batch_size):
        mb = torch.cat(patches[i:i + sw_batch_size], dim=0)  # [mb, C, pD, pH, pW]
        all_logits.append(model(mb).cpu())

    stacked = torch.cat(all_logits, dim=0)   # [n_patches, num_classes]
    print(f"    vol {tuple(volume.shape[-3:])} → {n_patches} patches, "
          f"mean logits over {stacked.shape[0]}")
    return stacked.mean(dim=0, keepdim=True)  # [1, num_classes]


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_weights(model: nn.Module, ckpt_path: str) -> nn.Module:
    """Load 'model.*' keys from a Lightning checkpoint into model."""
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model_state = {k[len("model."):]: v
                   for k, v in state.items() if k.startswith("model.")}
    if not model_state:
        model_state = state
    missing, unexpected = model.load_state_dict(model_state, strict=True)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys")
    return model


def build_model(args) -> tuple[nn.Module, str]:
    """Instantiate model and load checkpoint. Returns (model, dataset_type)."""
    mt = args.model_type
    print(f"[build_model] type={mt}  checkpoint={args.checkpoint}")

    if mt == "densenet3d":
        from models.densenet3d_classifier import DenseNet3DClassifier
        model        = DenseNet3DClassifier(in_channels=1, num_classes=2)
        dataset_type = "3d"

    elif mt == "medicalnet":
        from models.medicalnet_classifier import MedicalNetR18Classifier
        model = MedicalNetR18Classifier(
            in_channels=1, num_classes=2,
            pretrained=False,
            head_hidden_dims=[256], head_dropout=0.1,
        )
        dataset_type = "3d"

    elif mt == "swinunetr":
        from models.swinunetr_classifier import SwinUNETRClassifier
        model = SwinUNETRClassifier(
            in_channels=1, num_classes=2,
            feature_size=args.feature_size,
            img_size=tuple(args.patch_size),
            head_hidden_dims=[256, 128], head_dropout=0.1,
        )
        dataset_type = "3d"

    elif mt == "dinov2_3d":
        from utils.load_pretrained_backbone import create_backbone_with_checkpoint
        from models.downstream_meta_arch import DINOv2_3D_Classifier
        backbone = create_backbone_with_checkpoint(
            checkpoint_path=None,
            backbone_kwargs=dict(
                embed_dim=864, eva_depth=16, eva_numheads=12,
                input_channels=1, num_classes=1,
                input_shape=args.patch_size,
                patch_embed_size=[8, 8, 8],
                patch_drop_rate=0.0, classification=True,
                init_values=0.01, scale_attn_inner=True,
            ),
            strict=False, device="cpu", prefix=None,
        )
        model = DINOv2_3D_Classifier(
            backbone=backbone, num_classes=2,
            head_hidden_dims=[432, 216, 108], head_dropout=0.1,
            pooling="cls", input_channels=1,
        )
        dataset_type = "3d"

    elif mt == "meddinov3_2d":
        from models.meddinov3_2d_classifier import MedDINOv3_2D_Classifier
        model = MedDINOv3_2D_Classifier(
            num_classes=2, head_hidden=args.head_hidden,
            head_dropout=0.1, img_size=args.img_size,
            freeze_backbone=False,
            checkpoint_path=args.backbone_checkpoint,
        )
        dataset_type = "2d"

    elif mt == "threedino":
        from models.threedino_classifier import build_threedino_classifier
        model = build_threedino_classifier(
            weights_path=args.backbone_weights,
            backbone_name=args.backbone,
            num_classes=2,
            lora_layers=args.lora_layers,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )
        # Load fine-tuned classifier weights on top of backbone
        ckpt  = torch.load(args.checkpoint, map_location="cpu")
        # latest_checkpoint.pth wraps state_dict under "model" key
        state = ckpt["model"] if ("model" in ckpt and "iter" in ckpt) else ckpt
        model.load_state_dict(state)
        model.eval()
        return model, "threedino"

    else:
        raise ValueError(f"Unknown model_type: {mt}")

    model = _load_model_weights(model, args.checkpoint)
    model.eval()
    return model, dataset_type


# ──────────────────────────────────────────────────────────────────────────────
# Inference loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:         nn.Module,
    dataloader:    data.DataLoader,
    device:        torch.device,
    model_type:    str,
    patch_size:    list[int],
    overlap:       float = 0.5,
    sw_batch_size: int   = 4,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Per-model inference strategies:

    densenet3d / medicalnet
        Full volume → single forward pass.  AdaptiveAvgPool3d collapses any
        spatial size to a fixed-dim feature vector automatically.

    swinunetr / dinov2_3d
        Sliding window: extract overlapping patches of patch_size, run the
        model on each, return mean logits across all patches.

    meddinov3_2d
        Slice stack → single forward pass.
    """
    all_logits:  list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_uids:    list[str]          = []

    uses_sw      = model_type in ("swinunetr", "dinov2_3d")
    uses_threedino = model_type == "threedino"

    for batch_idx, batch in enumerate(dataloader):
        images  = batch["source"].to(device)   # [B, 1, D, H, W] or [B, N, H, W]
        targets = batch["target"]
        uids    = batch["uid"]

        if uses_threedino:
            # 3DINO: CLS-token pooling via predict_volume (one volume at a time)
            batch_logits: list[torch.Tensor] = []
            for i in range(images.shape[0]):
                vol = images[i : i + 1]   # [1, 1, D, H, W]
                logits_i = model.predict_volume(
                    vol, patch_size=patch_size[0],
                    sw_batch_size=sw_batch_size, overlap=overlap,
                )  # [1, num_classes] on CPU
                batch_logits.append(logits_i)
            logits = torch.cat(batch_logits, dim=0)

        elif uses_sw:
            # One sample at a time — volumes may have different spatial sizes
            batch_logits: list[torch.Tensor] = []
            for i in range(images.shape[0]):
                vol      = images[i : i + 1]   # [1, C, D, H, W]
                logits_i = sliding_window_classify(
                    model, vol, patch_size, overlap, sw_batch_size,
                )  # [1, num_classes] on CPU
                batch_logits.append(logits_i)
            logits = torch.cat(batch_logits, dim=0)   # [B, num_classes] on CPU

        else:
            # densenet3d / medicalnet: full volume, AdaptiveAvgPool handles size
            # meddinov3_2d: slice stack, single pass
            logits = model(images).cpu()

        all_logits.append(logits)
        all_targets.append(targets.cpu())
        all_uids.extend(uids if isinstance(uids, list) else list(uids))

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx + 1}/{len(dataloader)}]")

    return (
        torch.cat(all_logits,  dim=0),
        torch.cat(all_targets, dim=0),
        all_uids,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    preds = logits.argmax(dim=1).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    y     = targets.numpy()

    acc  = accuracy_score(y, preds)
    f1   = f1_score(y, preds, average="macro")
    f1pc = f1_score(y, preds, average=None)
    cm   = confusion_matrix(y, preds)
    try:
        auroc = roc_auc_score(y, probs[:, 1])
    except ValueError:
        auroc = float("nan")

    return {
        "accuracy":         acc,
        "f1_macro":         f1,
        "auroc":            auroc,
        "f1_class_0":       float(f1pc[0]) if len(f1pc) > 0 else float("nan"),
        "f1_class_1":       float(f1pc[1]) if len(f1pc) > 1 else float("nan"),
        "confusion_matrix": cm,
    }


def print_metrics(metrics: dict, model_type: str, split: str) -> None:
    cm = metrics.pop("confusion_matrix")
    print(f"\n{'='*55}")
    print(f"  Model : {model_type}   Split : {split}")
    print(f"{'='*55}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro) : {metrics['f1_macro']:.4f}")
    print(f"  AUROC      : {metrics['auroc']:.4f}")
    print(f"  F1 class-0 : {metrics['f1_class_0']:.4f}  (diverticulitis)")
    print(f"  F1 class-1 : {metrics['f1_class_1']:.4f}  (colon cancer)")
    print(f"  Confusion matrix:\n{cm}")
    print(f"{'='*55}\n")
    metrics["confusion_matrix"] = cm


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified inference for all BWT classifiers on Dataset109_CC.")

    p.add_argument("--model_type", required=True,
                   choices=["densenet3d", "medicalnet", "swinunetr",
                            "dinov2_3d", "meddinov3_2d", "threedino"])
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint file.  Mutually exclusive with --checkpoint_dir.")
    p.add_argument("--checkpoint_dir", default=None,
                   help="Folder containing checkpoints named <model_type>.ckpt "
                        "(or <model_type>.pth for threedino).  "
                        "The script picks the right file from --model_type automatically.")

    # Dataset
    p.add_argument("--dataset",    default="Dataset109_CC")
    p.add_argument("--split",      default="val",
                   choices=["train", "val", "test"])
    p.add_argument("--output_dir", default=None,
                   help="Where to write predictions CSV (defaults to ckpt dir)")

    # Spatial — used by swinunetr and dinov2_3d sliding window
    p.add_argument("--patch_size", nargs=3, type=int, default=[160, 160, 160],
                   help="Sliding-window patch size. "
                        "Use 96 96 96 for SwinUNETR, 160 160 160 for DINOv2-3D.")
    p.add_argument("--overlap",       type=float, default=0.5,
                   help="Overlap fraction between adjacent patches (swinunetr/dinov2_3d)")
    p.add_argument("--sw_batch_size", type=int,   default=4,
                   help="Patches processed in parallel during sliding window")
    p.add_argument("--feature_size",  type=int,   default=48,
                   help="SwinUNETR feature_size (must match training)")

    # 2-D MedDINOv3
    p.add_argument("--num_slices",  type=int,  default=32)
    p.add_argument("--img_size",    type=int,  default=224)
    p.add_argument("--head_hidden", nargs="+", type=int, default=[256, 128])
    p.add_argument("--backbone_checkpoint",
                   default=os.environ.get("MEDDINOV3_CKPT_PATH", None),
                   help="Path to MedDINOv3 backbone checkpoint (required for meddinov3_2d)")

    # 3DINO-specific
    p.add_argument("--backbone_weights",
                   default=os.environ.get("THREEDINO_WEIGHTS_PATH", None),
                   help="Path to 3DINO/MedDINOv3 backbone .pth (required for threedino). "
                        "Auto-detected from sibling of --checkpoint dir if not set.")
    p.add_argument("--backbone",   default="3dino", choices=["3dino", "meddinov3"],
                   help="Backbone variant for threedino (default: 3dino)")
    p.add_argument("--lora_layers", type=int,   default=0,
                   help="LoRA layers injected at training (0 = no LoRA). Must match checkpoint.")
    p.add_argument("--lora_rank",   type=int,   default=16)
    p.add_argument("--lora_alpha",  type=float, default=32.0)
    p.add_argument("--pp_dir",      default=None,
                   help="Override image directory for threedino "
                        "(e.g. .../nnUNetPlans_3d_fullres). "
                        "Bypasses the default root/pp_data/dataset/rescaledTr layout.")

    # Preprocessing (optional — runs before inference when --preprocess is set)
    p.add_argument("--preprocess",  action="store_true",
                   help="Run preprocessing (crop → resample → window) before inference.")
    p.add_argument("--raw_dir",     default=None,
                   help="Path to raw dataset folder containing images{Tr,Ts}/ and "
                        "labels{Tr,Ts}/ subdirs. Required when --preprocess is set.")
    p.add_argument("--crop_mode",   default="colon", choices=["colon", "lesion"],
                   help="'colon' = crop to TotalSegmentator colon mask; "
                        "'lesion' = crop to tumour/lesion mask. (default: colon)")
    p.add_argument("--seg_dir",     default=os.environ.get("auto_seg",
                                    "/data/colon_cancer/totalseg/total"),
                   help="Directory containing {uid}.nii.gz segmentation masks used "
                        "for cropping.  For colon mode: TotalSegmentator output.  "
                        "For lesion mode: tumour mask directory.")
    p.add_argument("--resample_spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                   help="Target voxel spacing after resampling (default: 1 1 1)")

    # Runtime
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    # ---- Resolve checkpoint path ----
    if args.checkpoint is None and args.checkpoint_dir is None:
        raise ValueError("Provide either --checkpoint or --checkpoint_dir.")
    if args.checkpoint is not None and args.checkpoint_dir is not None:
        raise ValueError("--checkpoint and --checkpoint_dir are mutually exclusive.")
    if args.checkpoint_dir is not None:
        ext = ".pth" if args.model_type == "threedino" else ".ckpt"
        resolved = Path(args.checkpoint_dir) / f"{args.model_type}{ext}"
        if not resolved.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {resolved}. "
                f"Expected a file named '{args.model_type}{ext}' inside --checkpoint_dir."
            )
        args.checkpoint = str(resolved)
        print(f"[inference] Resolved checkpoint: {args.checkpoint}")

    # ---- Optional preprocessing ----
    if args.preprocess:
        if args.raw_dir is None:
            raise ValueError("--raw_dir is required when --preprocess is set.")
        preprocess_data(args)

    # Auto-detect 3DINO backbone weights from the experiment directory
    if args.model_type == "threedino" and args.backbone_weights is None:
        exp_dir = Path(args.checkpoint).resolve().parent.parent
        fname   = ("meddinov3_vitb16_ct3m.pth" if args.backbone == "meddinov3"
                   else "3dino_vit_weights.pth")
        candidate = exp_dir / fname
        if candidate.exists():
            args.backbone_weights = str(candidate)
            print(f"[inference] Auto-detected backbone weights: {args.backbone_weights}")
        else:
            raise FileNotFoundError(
                f"Backbone weights not found at {candidate}. "
                "Pass --backbone_weights <path> explicitly."
            )

    model, dataset_type = build_model(args)
    model = model.to(device)
    model.eval()

    # ---- Dataset & loader ----
    if dataset_type == "2d":
        from configs.datasets.coloncancer_slices import ColonCancerSlices
        dataset = ColonCancerSlices(
            dataset_name=args.dataset, split=args.split,
            num_slices=args.num_slices, augment=False,
        )
    else:
        dataset = InferenceVolumeDataset(
            dataset_name=args.dataset, split=args.split,
            images_dir=args.pp_dir,
        )

    # batch_size=1 is required: volumes have different spatial sizes and
    # cannot be stacked into a batch tensor.
    loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # ---- Inference ----
    if args.model_type == "threedino":
        strategy = "cls-token-pooling"
    elif args.model_type in ("swinunetr", "dinov2_3d"):
        strategy = "sliding-window"
    else:
        strategy = "full-volume"
    print(f"[inference] {args.model_type} | strategy={strategy} | "
          f"split={args.split} | {len(dataset)} subjects")
    if strategy in ("sliding-window", "cls-token-pooling"):
        print(f"            patch={args.patch_size}  overlap={args.overlap}  "
              f"sw_batch={args.sw_batch_size}")

    logits, targets, uids = run_inference(
        model=model, dataloader=loader, device=device,
        model_type=args.model_type, patch_size=args.patch_size,
        overlap=args.overlap, sw_batch_size=args.sw_batch_size,
    )

    # ---- Metrics & output ----
    metrics = compute_metrics(logits, targets)
    print_metrics(metrics, args.model_type, args.split)

    preds = logits.argmax(dim=1).numpy()
    probs = torch.softmax(logits, dim=1).numpy()

    out_dir = Path(args.output_dir or Path(args.checkpoint).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"predictions_{args.model_type}_{args.dataset}_{args.split}.csv"
    pd.DataFrame({
        "uid":         uids,
        "true_label":  targets.numpy(),
        "pred_label":  preds,
        "prob_class0": probs[:, 0],
        "prob_class1": probs[:, 1],
    }).to_csv(csv_path, index=False)
    print(f"[inference] Predictions → {csv_path}")

    summary = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    summary.update({"model": args.model_type, "split": args.split,
                    "dataset": args.dataset, "n_samples": len(uids)})
    metrics_path = out_dir / f"metrics_{args.model_type}_{args.dataset}_{args.split}.csv"
    pd.DataFrame([summary]).to_csv(metrics_path, index=False)
    print(f"[inference] Metrics     → {metrics_path}")


if __name__ == "__main__":
    main()
