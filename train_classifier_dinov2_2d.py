"""
Training script: Facebook DINOv2 (2D) per-slice classifier.

Data flow
---------
  Training  : ColonCancerSlices returns one 2D axial slice [1, H, W] per
              sample.  The volume-level label is used for every slice.
              Loss is per-slice (one cross-entropy per [B, 1, H, W] batch).

  Model     : DINOv2SliceClassifier
              [B, 1, H, W] → gray→RGB → resize to img_size → ImageNet norm
              → DINOv2 ViT → CLS token [B, 768] → MLP head → [B, 2] logits.

  Inference : predict_volume() in inference/predict_volume_dinov2_2d.py
              samples N slices from the full volume and averages softmax
              probabilities to get a single 3D-level prediction.

Usage
-----
    python train_classifier_dinov2_2d.py --dataset Dataset115_CC
    python train_classifier_dinov2_2d.py --model_name dinov2_vits14
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("root_path", "/data/colon_cancer/CC_Detection")
os.environ.setdefault("auto_seg", "/data/colon_cancer/totalseg/total")

import torch

from configs.datasets.coloncancer_slices import ColonCancerSlices
from models.dinov2_2d_classifier import DINOv2SliceClassifier
from training.class_finetune_baselines import BaselineClassifierFinetuner
from training.data_module_finetune import DataModuleCC
from utils.mlflow_utils import load_run_id, save_run_info


class DINOv2SliceFinetuner(BaselineClassifierFinetuner):
    """
    Extends BaselineClassifierFinetuner to handle [B, K, 1, H, W] batches.

    ColonCancerSlices returns K slices per volume, so the batch shape from the
    DataLoader is [B, K, 1, H, W] with labels [B].  This class flattens to
    [B*K, 1, H, W] and expands labels to [B*K] before calling the parent step,
    so each slice gets an independent gradient signal with the volume's label.
    """

    def _shared_step(self, batch, stage: str):
        images = batch["source"]   # [B, K, 1, H, W]
        labels = batch["target"]   # [B]
        B, K = images.shape[:2]
        # Flatten K slices into the batch dimension
        batch["source"] = images.reshape(B * K, *images.shape[2:])   # [B*K, 1, H, W]
        batch["target"] = labels.repeat_interleave(K)                 # [B*K]
        return super()._shared_step(batch, stage)

MLRUNS_DIR = "/data/colon_cancer/dinov2_experiments/mlruns"
OUTPUT_DIR  = "/data/colon_cancer/dinov2_experiments/dinov2_pretrain"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",            default="Dataset109_CC")
    p.add_argument("--model_name",         default="dinov2_vitb14",
                   choices=[
                       "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
                       "dinov2_vits14_reg", "dinov2_vitb14_reg",
                       "dinov2_vitl14_reg", "dinov2_vitg14_reg",
                   ],
                   help="DINOv2 backbone variant to load from torch.hub")
    p.add_argument("--local_weights",      default=None,
                   help="Path to a local DINOv2 checkpoint (skips torch.hub download)")
    p.add_argument("--patch_size",         nargs=3, type=int, default=[160, 160, 160],
                   help="3D crop drawn from the volume before slicing. "
                        "Axial slices are patch_size[1] × patch_size[2].")
    p.add_argument("--img_size",           type=int, default=168,
                   help="Spatial size fed to DINOv2 (must be a multiple of 14). "
                        "168=12×14 is nearest to CT native 160 px. "
                        "Other options: 154 (11×14), 196 (14×14), 224 (16×14).")
    p.add_argument("--num_slices_per_volume", type=int, default=8,
                   help="Axial slices sampled per volume per iteration. "
                        "Effective batch sent to the model = batch_size × num_slices_per_volume.")
    p.add_argument("--batch_size",         type=int,   default=4,
                   help="Volumes per batch. Model sees batch_size × num_slices_per_volume slices.")
    p.add_argument("--num_patches_train",  type=int,   default=2500,
                   help="Slices drawn per training epoch.")
    p.add_argument("--num_patches_val",    type=int,   default=500,
                   help="Slices drawn per validation epoch.")
    p.add_argument("--max_epochs",         type=int,   default=150)
    p.add_argument("--lr",                 type=float, default=1e-3,
                   help="Learning rate for the classification head")
    p.add_argument("--lr_encoder",         type=float, default=1e-5,
                   help="Peak LR for the DINOv2 encoder after unfreeze")
    p.add_argument("--weight_decay",       type=float, default=1e-4)
    p.add_argument("--warmup_epochs",      type=int,   default=5,
                   help="Linear warm-up epochs for the head LR")
    p.add_argument("--encoder_warmup_epochs", type=int, default=5,
                   help="Epochs to ramp encoder LR from 0 after unfreeze")
    p.add_argument("--label_smoothing",    type=float, default=0.0)
    p.add_argument("--class_weights",      nargs=2, type=float, default=[1.195, 0.860],
                   metavar=("W0", "W1"),
                   help="Per-class loss weights. Default balanced for 58%% cancer Dataset115_CC")
    p.add_argument("--freeze_strategy",    default="first_epochs",
                   choices=["always", "first_epochs", "never"])
    p.add_argument("--freeze_epochs",      type=int,   default=5)
    p.add_argument("--num_workers",        type=int,   default=4)
    p.add_argument("--no_imagenet_norm",   action="store_true",
                   help="Skip ImageNet mean/std normalisation. "
                        "WARNING: DINOv2 was pretrained with this norm; disabling it "
                        "shifts inputs far outside the model's expected range.")
    p.add_argument("--resume",             default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    if args.resume is not None:
        run_dir  = Path(args.resume).parent
        run_name = run_dir.name
        print(f"[resume] Continuing from {args.resume}")
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        run_name  = f"dinov2_2d_{args.model_name}_{args.dataset}_{timestamp}"
        run_dir   = Path(OUTPUT_DIR) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    existing_run_id = load_run_id(args.resume)

    # ---- Datasets: each item is one 2D axial slice ----
    ds_train = ColonCancerSlices(
        dataset_name=args.dataset,
        split="train",
        patch_size=args.patch_size,
        num_patches_per_epoch=args.num_patches_train,
        num_slices_per_volume=args.num_slices_per_volume,
        augment=True,
    )
    ds_val = ColonCancerSlices(
        dataset_name=args.dataset,
        split="val",
        patch_size=args.patch_size,
        num_patches_per_epoch=args.num_patches_val,
        num_slices_per_volume=args.num_slices_per_volume,
        augment=False,
    )

    dm = DataModuleCC(
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Model: pure 2D classifier ----
    backbone = DINOv2SliceClassifier(
        model_name=args.model_name,
        num_classes=2,
        img_size=args.img_size,
        head_hidden_dims=None,   # default: [embed_dim // 2]
        head_dropout=0.1,
        local_weights=args.local_weights,
        imagenet_norm=not args.no_imagenet_norm,
    )
    model = DINOv2SliceFinetuner(
        model=backbone,
        model_name=f"dinov2_2d_{args.model_name}",
        num_classes=2,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        freeze_strategy=args.freeze_strategy,
        freeze_epochs=args.freeze_epochs,
        label_smoothing=args.label_smoothing,
        class_weights=args.class_weights,
        warmup_epochs=args.warmup_epochs,
        encoder_warmup_epochs=args.encoder_warmup_epochs,
        scheduler_t_max=args.max_epochs,
        scheduler_type="cosine",
    )

    # ---- Logging & callbacks ----
    logger = MLFlowLogger(
        experiment_name="baseline_classifiers",
        tracking_uri=f"file:{MLRUNS_DIR}",
        run_name=run_name,
        run_id=existing_run_id,
        tags={"model": f"dinov2_2d_{args.model_name}", "dataset": args.dataset},
    )
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_auroc", min_delta=0.001, patience=40, mode="max"),
        ModelCheckpoint(
            dirpath=str(run_dir),
            filename="dinov2_2d_best_{epoch:03d}-{val_auroc:.3f}",
            monitor="val_auroc",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
    ]

    # ---- Trainer ----
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        max_epochs=args.max_epochs,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        default_root_dir=str(run_dir),
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    save_run_info(run_dir, run_name, logger.run_id)
    print(f"[done] Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
