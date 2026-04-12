"""
[LAYER_START] Session 7: Training Pipeline
Full training loop with validation, checkpointing, early stopping, and logging.

[TRAINING PATH] Train → validate → checkpoint → early stop → export for inference.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, Optional

from torch.utils.data import DataLoader

from src.training.losses import WeightedMSELoss, CombinedLoss
from src.training.metrics import compute_all_metrics
from src.training.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training pipeline for DepressionModel.

    Handles:
        - Training loop with gradient clipping
        - Per-epoch validation with CCC/RMSE/MAE
        - Early stopping on validation CCC
        - Best model checkpointing
        - LR scheduling (ReduceLROnPlateau)
        - Training curve logging
        - Reproducibility via seed fixing
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        config: Dict,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: DepressionModel instance
            train_loader: Training DataLoader from SequenceBuilder
            dev_loader: Validation DataLoader from SequenceBuilder
            config: Parsed training_config.yaml dict
            device: torch device (auto-detects if None)
        """
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader

        # --- Reproducibility ---
        seed = config.get('training', {}).get('seed', 42)
        self._set_seed(seed)

        # --- Loss ---
        loss_cfg = config.get('loss', {})
        loss_type = loss_cfg.get('type', 'combined')

        if loss_type == 'combined':
            self.criterion = CombinedLoss(
                phq_threshold=loss_cfg.get('phq_threshold', 10.0),
                high_weight=loss_cfg.get('high_weight', 2.0),
                low_weight=loss_cfg.get('low_weight', 1.0),
                ccc_weight=loss_cfg.get('ccc_weight', 0.5),
                n_bins=loss_cfg.get('n_bins', 5),
                floor_weight=loss_cfg.get('floor_weight', 0.5),
                ceil_weight=loss_cfg.get('ceil_weight', 5.0),
            )
        else:
            self.criterion = WeightedMSELoss(
                phq_threshold=loss_cfg.get('phq_threshold', 10.0),
                high_weight=loss_cfg.get('high_weight', 2.0),
                low_weight=loss_cfg.get('low_weight', 1.0),
            )

        # --- Fit continuous loss weights from training labels ---
        if hasattr(self.criterion, 'fit'):
            train_labels = [
                s['label'] if isinstance(s['label'], (int, float))
                else s['label'].item()
                for s in train_loader.dataset.samples
            ]
            self.criterion.fit(train_labels)

        # Move criterion to device (registered buffers like bin_weights follow)
        self.criterion = self.criterion.to(self.device)

        # --- Regularization ---
        reg_cfg = config.get('regularization', {})
        self.entropy_lambda = reg_cfg.get('entropy_lambda', 0.1)
        self.gate_reg_lambda = reg_cfg.get('gate_reg_lambda', 0.01)

        # --- Optimizer ---
        opt_cfg = config.get('optimizer', {})
        opt_type = opt_cfg.get('type', 'adamw').lower()
        opt_params = dict(
            lr=opt_cfg.get('lr', 1e-3),
            weight_decay=opt_cfg.get('weight_decay', 1e-3),
        )

        if opt_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), **opt_params
            )
        elif opt_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), **opt_params
            )
        else:
            logger.warning(
                f"[VALIDATION_CHECK] Unknown optimizer '{opt_type}', defaulting to AdamW"
            )
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), **opt_params
            )
        logger.info(f"[LAYER_START] Optimizer: {opt_type}, lr={opt_params['lr']}, wd={opt_params['weight_decay']}")

        # --- Scheduler ---
        sched_cfg = config.get('scheduler', {})
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # CCC: higher is better
            factor=sched_cfg.get('factor', 0.5),
            patience=sched_cfg.get('patience', 3),
            min_lr=sched_cfg.get('min_lr', 1e-6),
        )

        # --- Early Stopping ---
        es_cfg = config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get('patience', 5),
            min_delta=es_cfg.get('min_delta', 0.001),
            mode='max',
        )

        # --- Training params ---
        train_cfg = config.get('training', {})
        self.max_epochs = train_cfg.get('epochs', 100)
        self.gradient_clip_norm = train_cfg.get('gradient_clip_norm', 1.0)

        # --- Checkpointing ---
        ckpt_cfg = config.get('checkpointing', {})
        self.save_dir = Path(ckpt_cfg.get('save_dir', 'checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_filename = ckpt_cfg.get('model_filename', 'best_model.pt')

        # --- Logging ---
        log_cfg = config.get('logging', {})
        self.log_dir = Path(log_cfg.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_steps = log_cfg.get('log_every_n_steps', 5)
        self.save_curves = log_cfg.get('save_curves', True)

        # --- History ---
        self.history = {
            'train_loss': [],
            'train_objective_loss': [],
            'val_loss': [],
            'val_ccc': [],
            'val_rmse': [],
            'val_mae': [],
            'lr': [],
        }

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"[LAYER_START] Trainer initialized: device={self.device}, "
            f"params={total_params:,}, epochs={self.max_epochs}, "
            f"lr={opt_cfg.get('lr', 1e-3)}, grad_clip={self.gradient_clip_norm}"
        )

    # =================================================================
    # TRAINING PATH: Main training loop
    # =================================================================
    def train(self) -> Dict[str, float]:
        """
        [TRAINING PATH] Run full training pipeline.

        Returns:
            Dict with best validation metrics
        """
        logger.info(
            f"[TRAINING_PATH] Starting training: {self.max_epochs} max epochs, "
            f"train={len(self.train_loader.dataset)} subjects, "
            f"val={len(self.dev_loader.dataset)} subjects"
        )

        best_metrics = {'ccc': -1.0, 'rmse': float('inf'), 'mae': float('inf')}
        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            # --- Train one epoch ---
            train_loss, train_objective_loss = self._train_epoch(epoch)

            # --- Validate ---
            val_loss, val_metrics = self._validate_epoch(epoch)

            # --- Record history ---
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_objective_loss'].append(train_objective_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_ccc'].append(val_metrics['ccc'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['lr'].append(current_lr)

            # --- Log epoch ---
            logger.info(
                f"[EPOCH {epoch}/{self.max_epochs}] "
                f"train_loss={train_loss:.4f} | "
                f"train_objective={train_objective_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_ccc={val_metrics['ccc']:.4f} | "
                f"val_rmse={val_metrics['rmse']:.2f} | "
                f"val_mae={val_metrics['mae']:.2f} | "
                f"lr={current_lr:.6f}"
            )

            # --- Scheduler step (monitors val CCC) ---
            self.scheduler.step(val_metrics['ccc'])

            # --- Checkpoint if best ---
            if val_metrics['ccc'] > best_metrics['ccc']:
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = epoch
                self._save_checkpoint(epoch, val_metrics)

            # --- Early stopping ---
            if self.early_stopping(val_metrics['ccc'], epoch):
                logger.info(
                    f"[TRAINING_PATH] Stopped at epoch {epoch}. "
                    f"Best CCC={best_metrics['ccc']:.4f} at epoch {best_metrics.get('epoch', '?')}"
                )
                break

        elapsed = time.time() - start_time

        # --- Save training curves ---
        if self.save_curves:
            self._save_training_curves()

        logger.info(
            f"[TRAINING_PATH] Training complete in {elapsed:.1f}s. "
            f"Best: CCC={best_metrics['ccc']:.4f}, "
            f"RMSE={best_metrics['rmse']:.2f}, MAE={best_metrics['mae']:.2f} "
            f"at epoch {best_metrics.get('epoch', '?')}"
        )

        return best_metrics

    # =================================================================
    # Internal: Train one epoch
    # =================================================================
    def _train_epoch(self, epoch: int) -> tuple:
        """Run one training epoch, return average base and optimized losses."""
        self.model.train()
        total_base_loss = 0.0
        total_objective_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(self.train_loader, 1):
            features = batch['features'].to(self.device, non_blocking=True)     # (B, T, D)
            labels = batch['labels'].to(self.device, non_blocking=True)          # (B,)
            mask = batch['mask'].to(self.device, non_blocking=True)              # (B, T)
            lengths = batch['lengths']                                            # (B,) stays CPU for pack_padded_sequence

            # Forward
            self.optimizer.zero_grad()
            predictions = self.model(features, mask, lengths)  # (B,)

            # Loss + entropy regularization
            base_loss = self.criterion(predictions, labels)
            loss = base_loss

            # Maximize attention entropy to prevent collapse to single timestep
            if self.entropy_lambda > 0 and hasattr(self.model, 'get_attention_entropy'):
                attn_entropy = self.model.get_attention_entropy()
                if attn_entropy is not None:
                    loss = loss - self.entropy_lambda * attn_entropy

            # Gate regularization: penalize deviation from 0.5 to prevent collapse
            if self.gate_reg_lambda > 0 and hasattr(self.model, 'last_gate_value'):
                gate_val = self.model.last_gate_value
                gate_penalty = (gate_val - 0.5) ** 2
                loss = loss + self.gate_reg_lambda * gate_penalty

            # Backward + clip + step
            loss.backward()
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )
            self.optimizer.step()

            total_base_loss += base_loss.item()
            total_objective_loss += loss.item()
            num_batches += 1

            # Step-level logging
            if step % self.log_every_n_steps == 0:
                logger.debug(
                    f"  [step {step}] loss={loss.item():.4f}"
                )

        avg_base_loss = total_base_loss / max(num_batches, 1)
        avg_objective_loss = total_objective_loss / max(num_batches, 1)
        return avg_base_loss, avg_objective_loss

    # =================================================================
    # Internal: Validate one epoch
    # =================================================================
    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> tuple:
        """
        Run validation, return (avg_loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []

        for batch in self.dev_loader:
            features = batch['features'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)
            lengths = batch['lengths']                                     # stays CPU for pack_padded_sequence

            predictions = self.model(features, mask, lengths)
            loss = self.criterion(predictions, labels)

            total_loss += loss.item()
            num_batches += 1

            # Clamp to valid PHQ-8 range for metric computation (MED-5: match inference)
            all_predictions.append(predictions.clamp(0.0, 24.0).cpu().numpy())
            all_targets.append(labels.cpu().numpy())

        avg_loss = total_loss / max(num_batches, 1)

        if num_batches == 0:
            logger.warning("Validation set produced zero batches — skipping metrics")
            return avg_loss, {'ccc': 0.0, 'rmse': float('inf'), 'mae': float('inf')}

        # Compute metrics
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        metrics = compute_all_metrics(all_predictions, all_targets)

        return avg_loss, metrics

    # =================================================================
    # Internal: Checkpointing
    # =================================================================
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> Path:
        """Save best model checkpoint with model_config for inference."""
        # Extract model architecture config for Predictor compatibility
        model_cfg = self.config.get('model', {})
        if not model_cfg:
            # Infer from model attributes
            model_cfg = {
                'input_dim': self.model.input_dim,
                'mlp': {'hidden_dim': self.model.mlp.hidden_dim,
                        'dropout': 0.5},
                'bigru': {'hidden_size': self.model.bigru.gru.hidden_size,
                          'num_layers': self.model.bigru.gru.num_layers},
                'head': {'dropout': 0.5},
            }

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model_config': model_cfg,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'early_stopping_state': self.early_stopping.state_dict(),
            'metrics': metrics,
            'best_metric': metrics.get('ccc', 0.0),
            'config': self.config,
        }

        path = self.save_dir / self.model_filename
        torch.save(checkpoint, path)
        logger.info(
            f"[CHECKPOINT] Best model saved: {path} | "
            f"epoch={epoch} | ccc={metrics['ccc']:.4f} | "
            f"rmse={metrics.get('rmse', 0):.2f} | mae={metrics.get('mae', 0):.2f}"
        )
        return path

    def _save_training_curves(self) -> Path:
        """Save training history as JSON for plotting."""
        path = self.log_dir / "training_curves.json"
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"[CHECKPOINT] Training curves saved: {path}")
        return path

    # =================================================================
    # Utility: Load checkpoint
    # =================================================================
    @staticmethod
    def load_checkpoint(
        path: str,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> Dict:
        """
        Load model weights from checkpoint.

        Args:
            path: Path to checkpoint .pt file
            model: Model instance to load weights into
            device: Target device

        Returns:
            Checkpoint dict (with metrics, epoch, config)
        """
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        logger.info(
            f"[INFERENCE_PATH] Loaded checkpoint: {path}, "
            f"epoch={checkpoint.get('epoch', '?')}, "
            f"ccc={checkpoint.get('metrics', {}).get('ccc', '?')}"
        )
        return checkpoint

    # =================================================================
    # Utility: Seed fixing
    # =================================================================
    @staticmethod
    def _set_seed(seed: int) -> None:
        """Fix all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"[TRAINING_PATH] Seed set to {seed}")
