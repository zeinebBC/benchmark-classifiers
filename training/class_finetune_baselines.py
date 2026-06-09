"""Generic PyTorch Lightning module for baseline 3-D CT classifiers.

Works with any model that exposes:
    model.encoder_parameters()  → iterable of encoder params
    model.head_parameters()     → iterable of head params
    model.freeze_encoder()      → disable encoder grads / set eval mode
    model.unfreeze_encoder()    → enable encoder grads / set train mode
    model(x)                    → logits [B, num_classes]

All three baseline models (SwinUNETRClassifier, DenseNet3DClassifier,
MedicalNetR18Classifier) satisfy this interface.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning import LightningModule
import torchmetrics


def _extract_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["source"], batch["target"]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported batch type: {type(batch)}")


class BaselineClassifierFinetuner(LightningModule):
    """
    Fine-tunes any encoder+head model on the colon-cancer / diverticulitis task.

    Metrics logged (train_ / val_ prefix):
        loss, acc, f1 (macro), auroc, f1_class_0, f1_class_1

    Args:
        model:           nn.Module with the interface described above.
        model_name:      String label used in MLFlow tags / checkpoint names.
        num_classes:     Number of output classes (default 2).
        lr:              Learning rate for the classification head.
        lr_encoder:      Learning rate for the encoder.  Set equal to ``lr``
                         to use a single param group (e.g. for DenseNet3D which
                         has no pretrained encoder weights).
        weight_decay:    Optimizer weight decay.
        optimizer_type:  "adamw" (default) or "sgd" (MedicalNet original).
        freeze_strategy: "always" | "first_epochs" | "never".
        freeze_epochs:   Number of warm-up epochs during which the encoder is
                         frozen (only used when freeze_strategy="first_epochs").
        label_smoothing: CrossEntropyLoss label smoothing.
        class_weights:   Per-class weights for the loss (list of floats).
        scheduler_type:        "cosine" | "poly" (PolyLR power=0.9, MedicalNet original).
        warmup_epochs:         Linear warm-up epochs for the HEAD from epoch 0.
                               Ignored for "poly".
        encoder_warmup_epochs: After the encoder unfreezes, ramp its LR from 0
                               to the full encoder LR over this many epochs.
                               Prevents catastrophic forgetting of pretrained weights.
        scheduler_t_max:       Total epochs for the scheduler.  None = no scheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "baseline",
        num_classes: int = 2,
        lr: float = 1e-4,
        lr_encoder: float = 1e-5,
        weight_decay: float = 1e-5,
        optimizer_type: str = "adamw",
        freeze_strategy: str = "first_epochs",
        freeze_epochs: int = 10,
        label_smoothing: float = 0.0,
        class_weights: Optional[Iterable[float]] = None,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 0,
        encoder_warmup_epochs: int = 10,
        scheduler_t_max: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.model_name = model_name

        weight_tensor = (
            torch.tensor(list(class_weights), dtype=torch.float32)
            if class_weights else None
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=weight_tensor, label_smoothing=label_smoothing
        )

        task_kw = {"task": "multiclass", "num_classes": num_classes}
        self.train_acc   = torchmetrics.Accuracy(**task_kw)
        self.val_acc     = torchmetrics.Accuracy(**task_kw)
        self.train_f1    = torchmetrics.F1Score(**task_kw)
        self.val_f1      = torchmetrics.F1Score(**task_kw)
        self.train_auroc = torchmetrics.AUROC(**task_kw)
        self.val_auroc   = torchmetrics.AUROC(**task_kw)
        self.train_f1_pc = torchmetrics.F1Score(**task_kw, average="none")
        self.val_f1_pc   = torchmetrics.F1Score(**task_kw, average="none")

        self.lr              = lr
        self.lr_encoder      = lr_encoder
        self.weight_decay    = weight_decay
        self.optimizer_type        = optimizer_type
        self.freeze_strategy       = freeze_strategy
        self.freeze_epochs         = freeze_epochs
        self.scheduler_type        = scheduler_type
        self.warmup_epochs         = warmup_epochs
        self.encoder_warmup_epochs = encoder_warmup_epochs
        self.scheduler_t_max       = scheduler_t_max
        self.num_classes           = num_classes

    # ------------------------------------------------------------------
    # Backbone freeze / unfreeze
    # ------------------------------------------------------------------
    def _freeze_encoder(self) -> None:
        if hasattr(self.model, "freeze_encoder"):
            self.model.freeze_encoder()

    def _unfreeze_encoder(self) -> None:
        if hasattr(self.model, "unfreeze_encoder"):
            self.model.unfreeze_encoder()

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        ep = self.current_epoch
        if self.freeze_strategy == "always":
            if ep == 0:
                self._freeze_encoder()
        elif self.freeze_strategy == "first_epochs":
            if ep < self.freeze_epochs:
                self._freeze_encoder()
            elif ep == self.freeze_epochs:
                self._unfreeze_encoder()
        elif self.freeze_strategy == "never":
            if ep == 0:
                self._unfreeze_encoder()

        # Propagate epoch to dataset for deterministic augmentation seeding
        if hasattr(self, "trainer") and hasattr(self.trainer, "datamodule"):
            for attr in ("ds_train", "ds_val"):
                ds = getattr(self.trainer.datamodule, attr, None)
                if ds and hasattr(ds, "set_epoch"):
                    ds.set_epoch(ep)

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------
    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, labels = _extract_batch(batch)
        logits = self.model(images)
        loss   = self.criterion(logits, labels)
        preds  = logits.argmax(dim=1)
        probs  = torch.softmax(logits, dim=1)

        acc    = getattr(self, f"{stage}_acc")
        f1     = getattr(self, f"{stage}_f1")
        auroc  = getattr(self, f"{stage}_auroc")
        f1_pc  = getattr(self, f"{stage}_f1_pc")
        acc(preds, labels)
        f1(preds, labels)
        auroc(probs, labels)
        f1_pc(preds, labels)   # needed so epoch-end compute() returns real values

        bs = labels.size(0)
        self.log(f"{stage}_loss",  loss,  prog_bar=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}_acc",   acc,   prog_bar=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}_f1",    f1,    prog_bar=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}_auroc", auroc, prog_bar=True, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    # ------------------------------------------------------------------
    # Per-class F1 at epoch end
    # ------------------------------------------------------------------
    def on_train_epoch_end(self) -> None:
        for i, v in enumerate(self.train_f1_pc.compute()):
            self.log(f"train_f1_class_{i}", v, on_epoch=True)
        self.train_f1_pc.reset()

    def on_validation_epoch_end(self) -> None:
        for i, v in enumerate(self.val_f1_pc.compute()):
            self.log(f"val_f1_class_{i}", v, on_epoch=True)
        self.val_f1_pc.reset()
        # Log per-group learning rates once per validation epoch
        if hasattr(self, "trainer") and self.trainer.optimizers:
            for i, pg in enumerate(self.trainer.optimizers[0].param_groups):
                self.log(f"lr_group_{i}", pg["lr"], on_epoch=True)

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        has_split = (
            hasattr(self.model, "encoder_parameters")
            and hasattr(self.model, "head_parameters")
            and self.lr_encoder != self.lr
        )
        if has_split:
            param_groups = [
                {"params": list(self.model.head_parameters()),    "lr": self.lr},
                {"params": list(self.model.encoder_parameters()), "lr": self.lr_encoder},
            ]
        else:
            param_groups = [{"params": list(self.model.parameters()), "lr": self.lr}]

        if self.optimizer_type == "sgd":
            optimizer = SGD(
                param_groups, momentum=0.9, nesterov=True, weight_decay=self.weight_decay,
            )
        else:
            optimizer = AdamW(param_groups, weight_decay=self.weight_decay)

        if self.scheduler_t_max is None:
            return optimizer

        T    = self.scheduler_t_max
        fe   = self.freeze_epochs if self.freeze_strategy == "first_epochs" else 0
        ew   = self.encoder_warmup_epochs   # warmup epochs for encoder after unfreeze
        hw   = self.warmup_epochs           # warmup epochs for head from epoch 0

        # ---- helpers ----
        def _poly(epoch):
            return max(1e-7, (1.0 - epoch / T) ** 0.9)

        def _cosine(epoch, start, warm, total):
            """Cosine schedule with linear warm-up, offset to start at `start`."""
            ep = epoch - start
            if ep < 0:
                return 0.0          # before this group starts (encoder during freeze)
            if warm > 0 and ep < warm:
                return 1e-3 + (1.0 - 1e-3) * ep / warm   # linear warmup
            progress = (ep - warm) / max(1, total - warm)
            return max(1e-7, 0.5 * (1.0 + math.cos(math.pi * progress)))

        # ---- per-group lambdas ----
        if not has_split:
            # Single group (DenseNet3D): standard schedule, no encoder concern
            if self.scheduler_type == "poly":
                lr_lambda = [_poly]
            else:
                lr_lambda = [lambda e: _cosine(e, start=0, warm=hw, total=T)]
        else:
            # Head: normal schedule from epoch 0
            if self.scheduler_type == "poly":
                head_fn = lambda e: _poly(e)
            else:
                head_fn = lambda e: _cosine(e, start=0, warm=hw, total=T)

            # Encoder: zero during frozen phase, then linear warmup, then main schedule.
            # This prevents catastrophic forgetting when pretrained weights are first exposed.
            if self.scheduler_type == "poly":
                def enc_fn(epoch):
                    if epoch < fe:
                        return 0.0                          # frozen
                    ep_after = epoch - fe
                    if ep_after < ew:
                        return (ep_after / max(1, ew)) * _poly(epoch)   # warmup
                    return _poly(epoch)                     # full poly decay
            else:
                def enc_fn(epoch):
                    if epoch < fe:
                        return 0.0                          # frozen
                    return _cosine(epoch, start=fe, warm=ew, total=T - fe)

            lr_lambda = [head_fn, enc_fn]

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
