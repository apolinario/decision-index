"""
Trainer module for D-JEPA fine-tuning and calibration with AMP and gradient accumulation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from djepa.model.djepa import DJEPA
from djepa.training.loss import DJEPACombinedLoss


class DJEPATrainer:
    """
    Manages D-JEPA training with mixed precision, gradient accumulation, and calibration monitoring.
    """

    def __init__(
        self,
        model: DJEPA,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        lr: float = 3e-5,
        lr_backbone: Optional[float] = None,
        lr_head: Optional[float] = None,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.1,
        epochs: int = 5,
        accumulate_grad_batches: int = 4,
        use_amp: bool = True,
        device: Optional[torch.device] = None,
        loss_fn: Optional[DJEPACombinedLoss] = None,
        output_dir: Union[str, Path] = "checkpoints",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.accumulate_grad_batches = accumulate_grad_batches
        self.use_amp = use_amp
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

        self.loss_fn = loss_fn if loss_fn is not None else DJEPACombinedLoss()

        effective_lr_backbone = lr_backbone if lr_backbone is not None else lr
        effective_lr_head = lr_head if lr_head is not None else (lr * 3.0)

        # Optimizer: separate backbone from new predictor/scorer heads with differential learning rates
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "backbone" in n and not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "lr": effective_lr_backbone,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "backbone" in n and any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "lr": effective_lr_backbone,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "backbone" not in n and not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "lr": effective_lr_head,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "backbone" not in n and any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "lr": effective_lr_head,
                "weight_decay": 0.0,
            },
        ]
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        total_steps = (len(train_dataloader) // accumulate_grad_batches + 1) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_nll = 0.0
        total_brier = 0.0
        total_anti_lure = 0.0
        correct = 0
        total_samples = 0

        self.optimizer.zero_grad()

        amp_context = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if (self.use_amp and self.device.type == "cuda")
            else torch.no_grad()
        )

        for step, batch in enumerate(self.train_dataloader):
            state_input_ids = batch["state_input_ids"].to(self.device)
            state_attention_mask = batch["state_attention_mask"].to(self.device)
            option_input_ids = batch["option_input_ids"].to(self.device)
            option_attention_mask = batch["option_attention_mask"].to(self.device)
            options_valid_mask = batch["options_valid_mask"].to(self.device)
            targets = batch["targets"].to(self.device)
            trap_indices = batch["trap_indices"].to(self.device)

            with amp_context:
                outputs = self.model(
                    state_input_ids=state_input_ids,
                    state_attention_mask=state_attention_mask,
                    option_input_ids=option_input_ids,
                    option_attention_mask=option_attention_mask,
                    options_valid_mask=options_valid_mask,
                )

                loss_dict = self.loss_fn(
                    logits=outputs["logits"],
                    probabilities=outputs["probabilities"],
                    scores=outputs["scores"],
                    targets=targets,
                    trap_indices=trap_indices,
                    options_valid_mask=options_valid_mask,
                )

                loss = loss_dict["loss"] / self.accumulate_grad_batches

            loss.backward()

            if (step + 1) % self.accumulate_grad_batches == 0 or (step + 1) == len(self.train_dataloader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Tracking metrics
            bs = targets.size(0)
            total_loss += loss_dict["loss"].item() * bs
            total_nll += loss_dict["loss_nll"].item() * bs
            total_brier += loss_dict["loss_brier"].item() * bs
            total_anti_lure += loss_dict["loss_anti_lure"].item() * bs

            preds = torch.argmax(outputs["probabilities"], dim=-1)
            correct += (preds == targets).sum().item()
            total_samples += bs

        return {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_samples),
            "train_nll": total_nll / max(1, total_samples),
            "train_brier": total_brier / max(1, total_samples),
            "train_anti_lure": total_anti_lure / max(1, total_samples),
            "train_acc": correct / max(1, total_samples),
            "temperature": float(self.model.scorer.temperature.item()),
        }

    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        eval_dl = dataloader or self.val_dataloader
        if eval_dl is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        all_probs = []
        all_targets = []

        amp_context = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if (self.use_amp and self.device.type == "cuda")
            else torch.no_grad()
        )

        for batch in eval_dl:
            state_input_ids = batch["state_input_ids"].to(self.device)
            state_attention_mask = batch["state_attention_mask"].to(self.device)
            option_input_ids = batch["option_input_ids"].to(self.device)
            option_attention_mask = batch["option_attention_mask"].to(self.device)
            options_valid_mask = batch["options_valid_mask"].to(self.device)
            targets = batch["targets"].to(self.device)
            trap_indices = batch["trap_indices"].to(self.device)

            with amp_context:
                outputs = self.model(
                    state_input_ids=state_input_ids,
                    state_attention_mask=state_attention_mask,
                    option_input_ids=option_input_ids,
                    option_attention_mask=option_attention_mask,
                    options_valid_mask=options_valid_mask,
                )

                loss_dict = self.loss_fn(
                    logits=outputs["logits"],
                    probabilities=outputs["probabilities"],
                    scores=outputs["scores"],
                    targets=targets,
                    trap_indices=trap_indices,
                    options_valid_mask=options_valid_mask,
                )

            bs = targets.size(0)
            total_loss += loss_dict["loss"].item() * bs
            preds = torch.argmax(outputs["probabilities"], dim=-1)
            correct += (preds == targets).sum().item()
            total_samples += bs

            all_probs.append(outputs["probabilities"].cpu())
            all_targets.append(targets.cpu())

        acc = correct / max(1, total_samples)

        probs_cat = torch.cat(all_probs, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        confidences, predictions = torch.max(probs_cat, dim=-1)
        accuracies = (predictions == targets_cat).float()

        # Compute 10-bin ECE
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean().item()
            if prop_in_bin > 0:
                acc_in_bin = accuracies[in_bin].mean().item()
                conf_in_bin = confidences[in_bin].mean().item()
                ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin

        return {
            "val_loss": total_loss / max(1, total_samples),
            "val_acc": acc,
            "val_ece": ece,
            "total_samples": total_samples,
        }

    def save_checkpoint(self, path: Union[str, Path]):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "temperature": float(self.model.scorer.temperature.item()),
            },
            save_path,
        )

    def load_checkpoint(self, path: Union[str, Path], load_optimizer: bool = False):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_optimizer and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "temperature" in checkpoint and hasattr(self.model, "scorer"):
            with torch.no_grad():
                self.model.scorer.log_temperature.copy_(
                    torch.tensor(float(torch.log(torch.tensor(checkpoint["temperature"]))), device=self.device)
                )
