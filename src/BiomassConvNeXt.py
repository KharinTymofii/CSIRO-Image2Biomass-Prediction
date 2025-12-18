import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Callable
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class BiomassConvNeXt(pl.LightningModule):
    """
    Dual-head model with two approaches for biomass prediction:
    1. GLOBAL HEAD: Merge features from both patches -> predict total mass
    2. LOCAL HEAD: Predict mass for each patch separately -> sum them

    Input: Left and Right image patches (no tabular features)
    Output: ['Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    """

    def __init__(
        self,
        kaggle_score: Callable,
        backbone_name: str = 'seresnextaa101d_32x8d.sw_in12k_ft_in1k_288',
        backbone_hidden_dim: int = 1536,
        backbone_local_file: str = '',
        num_targets: int = 3,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        hidden_ratio: float = 0.5,
        dropout: float = 0.2,
        use_log_target: bool = True,
        freeze_backbone: bool = True,
    ):
        """
        Args:
            backbone_name: Backbone model name (e.g., 'seresnextaa101d_32x8d.sw_in12k_ft_in1k_288')
            num_targets: Number of regression targets (3)
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            hidden_ratio: Ratio for hidden layer size
            dropout: Dropout probability
            use_log_target: If True, predict log1p transformed targets
            freeze_backbone: If True, freeze backbone initially
            unfreeze_last_n_layers: Number of last encoder layers to unfreeze
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_targets = num_targets
        self.use_log_target = use_log_target

        # Load backbone
        if backbone_local_file:
            self.backbone = timm.create_model(
                backbone_name, pretrained=False)
            state_dict = torch.load(backbone_local_file)
            self.backbone.load_state_dict(state_dict)
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=True)
            print("Loaded pretrained model from timm:", backbone_name)

        self.backbone_hidden_dim = backbone_hidden_dim

        # Freeze/unfreeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = max(32, int(self.backbone_hidden_dim * hidden_ratio))

        # =================================================================
        # HEAD 1: GLOBAL APPROACH
        # Concatenate averaged features from left and right patches
        # Then predict total mass for the whole image
        # =================================================================
        self.head_global = nn.Sequential(
            nn.Linear(self.backbone_hidden_dim * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_targets)
        )

        # =================================================================
        # HEAD 2: LOCAL APPROACH (Sum of Parts)
        # Predict mass for left and right patches separately
        # Then sum the predictions (mass_total = mass_left + mass_right)
        # =================================================================
        self.head_local = nn.Sequential(
            nn.Linear(self.backbone_hidden_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_targets)
        )

        # Learnable ensemble weights for combining both heads
        self.ensemble_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        self.lr = lr
        self.weight_decay = weight_decay

        self.validation_step_outputs = []
        self.kaggle_score = kaggle_score

    def forward(self, batch: dict) -> dict:
        """
        Forward pass with dual-head processing.

        Args:
            batch: Dict with 'left_image' and 'right_image' tensors

        Returns:
            Dict with 'global' and 'local' predictions
        """
        # Extract features from both patches using DINOv2
        left_outputs = self.backbone(batch['left_image'])
        left_features = left_outputs.last_hidden_state  # [B, N_patches+1, 768]

        right_outputs = self.backbone(batch['right_image'])
        right_features = right_outputs.last_hidden_state
        right_patches = right_features[:, 1:, :]

        # =================================================================
        # HEAD 1: GLOBAL APPROACH
        # Average pool each side -> concatenate -> predict
        # =================================================================
        left_pooled = left_patches.mean(dim=1)  # [B, 768]
        right_pooled = right_patches.mean(dim=1)  # [B, 768]

        global_context = torch.cat(
            [left_pooled, right_pooled], dim=1)  # [B, 1536]
        pred_global = self.head_global(global_context)  # [B, 3]

        # =================================================================
        # HEAD 2: LOCAL APPROACH (Sum of Parts)
        # Predict for each side separately, then sum
        # =================================================================
        pred_left = self.head_local(left_pooled)  # [B, 3]
        pred_right = self.head_local(right_pooled)  # [B, 3]
        pred_local = pred_left + pred_right  # [B, 3] - sum of masses

        return {
            'global': pred_global,
            'local': pred_local,
            'left': pred_left,
            'right': pred_right
        }

    def compute_loss(self, preds: dict, targets: torch.Tensor) -> dict:
        """
        Compute losses for both heads.

        Args:
            preds: Dict with predictions from forward pass
            targets: Ground truth targets [B, 3]

        Returns:
            Dict with individual and total losses
        """
        # Loss for global head
        loss_global = F.huber_loss(preds['global'], targets, delta=1.0)

        # Loss for local head (sum of parts)
        loss_local = F.huber_loss(preds['local'], targets, delta=1.0)

        # Total loss (equal weight for both approaches)
        loss_total = 0.5 * loss_global + 0.5 * loss_local

        return {
            'loss_global': loss_global,
            'loss_local': loss_local,
            'loss_total': loss_total
        }

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        targets = batch['targets']  # [B, 3]
        losses = self.compute_loss(preds, targets)

        # Log losses
        self.log('train/loss_total', losses['loss_total'],
                 on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=targets.size(0))
        self.log('train/loss_global', losses['loss_global'],
                 on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/loss_local', losses['loss_local'],
                 on_step=False, on_epoch=True, prog_bar=False)

        return losses['loss_total']

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        targets = batch['targets']
        losses = self.compute_loss(preds, targets)

        # Log losses
        self.log('val/loss_total', losses['loss_total'],
                 on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=targets.size(0))
        self.log('val/loss_global', losses['loss_global'],
                 on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/loss_local', losses['loss_local'],
                 on_step=False, on_epoch=True, prog_bar=False)

        # Ensemble prediction for metrics
        w = F.softmax(self.ensemble_weights, dim=0)
        pred_ensemble = w[0] * preds['global'] + w[1] * preds['local']

        # Convert back from log space if needed
        if self.use_log_target:
            pred_ensemble = torch.expm1(pred_ensemble)
            pred_global = torch.expm1(preds['global'])
            pred_local = torch.expm1(preds['local'])
            targets_original = torch.expm1(targets)
        else:
            pred_global = preds['global']
            pred_local = preds['local']
            targets_original = targets

        # Clamp to non-negative
        pred_ensemble = torch.clamp(pred_ensemble, min=0.0)
        pred_global = torch.clamp(pred_global, min=0.0)
        pred_local = torch.clamp(pred_local, min=0.0)

        # Store for epoch-end metrics
        self.validation_step_outputs.append({
            'pred_ensemble': pred_ensemble.detach().cpu(),
            'pred_global': pred_global.detach().cpu(),
            'pred_local': pred_local.detach().cpu(),
            'targets': targets_original.detach().cpu()
        })

        return losses['loss_total']

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return

        # Concatenate all predictions
        pred_ensemble = torch.cat([x['pred_ensemble']
                                   for x in self.validation_step_outputs], dim=0).numpy()
        pred_global = torch.cat([x['pred_global']
                                for x in self.validation_step_outputs], dim=0).numpy()
        pred_local = torch.cat([x['pred_local']
                               for x in self.validation_step_outputs], dim=0).numpy()
        targets = torch.cat([x['targets']
                            for x in self.validation_step_outputs], dim=0).numpy()

        # Compute R2 scores for each approach
        r2_ensemble = self.kaggle_score(targets, pred_ensemble)
        r2_global = self.kaggle_score(targets, pred_global)
        r2_local = self.kaggle_score(targets, pred_local)

        self.log('val/r2_ensemble', r2_ensemble, on_epoch=True, prog_bar=True)
        self.log('val/r2_global', r2_global, on_epoch=True, prog_bar=False)
        self.log('val/r2_local', r2_local, on_epoch=True, prog_bar=False)

        # Log ensemble weights
        w = F.softmax(self.ensemble_weights, dim=0)
        self.log('val/weight_global', w[0].item(), on_epoch=True)
        self.log('val/weight_local', w[1].item(), on_epoch=True)

        self.validation_step_outputs.clear()

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Inference: return ensemble prediction.

        Args:
            batch: Input batch
            batch_idx: Batch index

        Returns:
            Predictions [B, 3]
        """
        preds = self(batch)

        # Ensemble both heads
        w = F.softmax(self.ensemble_weights, dim=0)
        final_pred = w[0] * preds['global'] + w[1] * preds['local']

        # Convert from log space if needed
        if self.use_log_target:
            final_pred = torch.expm1(final_pred)

        return torch.clamp(final_pred, min=0.0)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs or 20,
            eta_min=self.lr * 0.01
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
