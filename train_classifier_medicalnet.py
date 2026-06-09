"""
Training script: MedicalNet ResNet-18 classifier for colon cancer vs. diverticulitis.

Input: 1-channel CT-only colon-cropped volumes from rescaledTr/.
Backbone: ResNet-18 pretrained on 8 diverse medical 3-D datasets (MedicalNet,
          TencentMedicalNet/MedicalNet-Resnet18 on HuggingFace).

Usage:
    python train_classifier_medicalnet.py
    python train_classifier_medicalnet.py --no_pretrained --max_epochs 200
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

from configs.datasets.coloncancer import ColonCancer
from models.medicalnet_classifier import MedicalNetR18Classifier
from training.class_finetune_baselines import BaselineClassifierFinetuner
from training.data_module_finetune import DataModuleCC
from utils.mlflow_utils import load_run_id, save_run_info

MLRUNS_DIR = "/data/colon_cancer/dinov2_experiments/mlruns"
OUTPUT_DIR = "/data/colon_cancer/dinov2_experiments/dinov2_pretrain"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",           default="Dataset109_CC")
    p.add_argument("--patch_size",        nargs=3, type=int, default=[160, 160, 160])
    p.add_argument("--no_pretrained",     action="store_true",
                   help="Disable MedicalNet initialisation (random weights)")
    p.add_argument("--batch_size",        type=int,   default=8)
    p.add_argument("--num_patches_train", type=int,   default=2500)
    p.add_argument("--num_patches_val",   type=int,   default=250)
    p.add_argument("--max_epochs",        type=int,   default=200)
    p.add_argument("--lr",                type=float, default=1e-2,
                   help="Learning rate for the classification head (SGD default 1e-2)")
    p.add_argument("--lr_encoder",        type=float, default=1e-3,
                   help="Learning rate for the ResNet encoder (SGD default 1e-3)")
    p.add_argument("--weight_decay",      type=float, default=1e-3,
                   help="Weight decay (MedicalNet original: 1e-3)")
    p.add_argument("--optimizer_type",    default="sgd", choices=["sgd", "adamw"],
                   help="sgd = TencentMedicalNet original; adamw = modern alternative")
    p.add_argument("--scheduler_type",    default="poly", choices=["poly", "cosine"],
                   help="poly = MedicalNet original (power=0.9); cosine = cosine annealing")
    p.add_argument("--label_smoothing",   type=float, default=0.0,
                   help="CrossEntropy label smoothing (0 = off)")
    p.add_argument("--class_weights",     nargs=2,    type=float, default=[1.195, 0.860],
                   metavar=("W0", "W1"),
                   help="Per-class loss weights. Default balanced for 58%% cancer Dataset115_CC")
    p.add_argument("--freeze_strategy",      default="first_epochs",
                   choices=["always", "first_epochs", "never"])
    p.add_argument("--freeze_epochs",        type=int,   default=10)
    p.add_argument("--encoder_warmup_epochs", type=int,  default=20,
                   help="Epochs to linearly ramp encoder LR from 0 after unfreeze "
                        "(prevents forgetting pretrained weights; default 20 for SGD)")
    p.add_argument("--num_workers",       type=int,   default=4)
    p.add_argument("--resume",            default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    patch_size = args.patch_size

    if args.resume is not None:
        run_dir  = Path(args.resume).parent
        run_name = run_dir.name
        print(f"[resume] Continuing from {args.resume}")
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        run_name  = f"medicalnet_r18_classifier_{args.dataset}_{timestamp}"
        run_dir   = Path(OUTPUT_DIR) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    existing_run_id = load_run_id(args.resume)

    # ---- Datasets ----
    ds_kwargs = dict(
        dataset_name=args.dataset,
        patch_size=patch_size,
        head="classification_colon",
        use_labels=False,
    )
    ds_train = ColonCancer(**ds_kwargs, split="train",
                           num_patches_per_epoch=args.num_patches_train)
    ds_val   = ColonCancer(**ds_kwargs, split="val",
                           num_patches_per_epoch=args.num_patches_val)

    dm = DataModuleCC(
        ds_train=ds_train, ds_val=ds_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Model ----
    backbone = MedicalNetR18Classifier(
        in_channels=1,
        num_classes=2,
        pretrained=not args.no_pretrained,
        head_hidden_dims=[256],
        head_dropout=0.1,
    )
    model = BaselineClassifierFinetuner(
        model=backbone,
        model_name="medicalnet_r18",
        num_classes=2,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer_type,
        freeze_strategy=args.freeze_strategy,
        freeze_epochs=args.freeze_epochs,
        label_smoothing=args.label_smoothing,
        class_weights=args.class_weights,
        scheduler_type=args.scheduler_type,
        encoder_warmup_epochs=args.encoder_warmup_epochs,
        scheduler_t_max=args.max_epochs,
    )

    # ---- Logging & callbacks ----
    logger = MLFlowLogger(
        experiment_name="baseline_classifiers",
        tracking_uri=f"file:{MLRUNS_DIR}",
        run_name=run_name,
        run_id=existing_run_id,
        tags={"model": "medicalnet_r18", "dataset": args.dataset,
              "pretrained": str(not args.no_pretrained)},
    )
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_auroc", min_delta=0.001, patience=40, mode="max"),
        ModelCheckpoint(
            dirpath=str(run_dir),
            filename="medicalnet_r18_best_{epoch:03d}-{val_auroc:.3f}",
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
