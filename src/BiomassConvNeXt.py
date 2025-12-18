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
        backbone_input_size: int,
        backbone_name: str = 'seresnextaa101d_32x8d.sw_in12k_ft_in1k_288',
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

        backbone_kwargs = {
            'model_name': backbone_name,
            'num_classes': 0,  # No classification head
            'global_pool': '',
        }

        # Load backbone
        if backbone_local_file:
            self.backbone = timm.create_model(
                **backbone_kwargs, pretrained=False)
            state_dict = torch.load(backbone_local_file)
            self.backbone.load_state_dict(state_dict)
        else:
            self.backbone = timm.create_model(
                **backbone_kwargs, pretrained=True)
            print("Loaded pretrained model from timm:", backbone_name)

        self.backbone_hidden_dim = self._get_backbone_output_dim(
            self.backbone, input_size=backbone_input_size)

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
            nn.BatchNorm1d(hidden_size * 2),
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
            nn.Conv2d(self.backbone_hidden_dim, hidden_size, kernel_size=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_size, self.num_targets, kernel_size=1),
        )

        # Learnable ensemble weights for combining both heads
        self.ensemble_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        self.lr = lr
        self.weight_decay = weight_decay

        self.validation_step_outputs = []
        self.kaggle_score = kaggle_score

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass with dual-head processing.

        Args:
            left_img: Left image tensor
            right_img: Right image tensor

        Returns:
            Tensor with predictions for biomass
        """
        # Extract features from the backbone
        left_features = self.backbone(left_img)
        right_features = self.backbone(right_img)

        # Adaptive pooling to get fixed-size feature maps
        l_avg = F.adaptive_avg_pool2d(
            left_features, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
        l_max = F.adaptive_max_pool2d(
            left_features, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
        r_avg = F.adaptive_avg_pool2d(
            right_features, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
        r_max = F.adaptive_max_pool2d(
            right_features, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]

        # [B, C*2] (concatenate avg and max from both sides)
        global_feat = torch.cat([l_avg, r_avg], dim=1)  # [B, C*2]

        # Pass through a linear layer to predict biomass
        pred_global = self.head_global(global_feat)  # [B, 3]

        # [B, 3, H, W] -> [B, 3] (spatial pooling)
        pred_left = self.head_local(left_features)  # [B, 3, H, W]
        pred_right = self.head_local(right_features)  # [B, 3, H, W]

        # Global average pooling to get [B, 3]
        pred_left = F.adaptive_avg_pool2d(
            pred_left, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 3]
        pred_right = F.adaptive_avg_pool2d(
            pred_right, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 3]

        pred_local = pred_left + pred_right  # [B, 3]

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

    @staticmethod
    def _get_backbone_output_dim(backbone: nn.Module, input_size: int) -> int:
        """Utility to get backbone output dimension given input size."""
        dummy = torch.randn(1, 3, input_size, input_size)
        backbone_output = backbone.forward_features(dummy)
        if hasattr(backbone_output, 'shape'):
            backbone_out_dim = backbone_output.shape[1]
        elif hasattr(backbone_output, 'last_hidden_state'):
            backbone_out_dim = backbone_output.last_hidden_state.size(-1)
        else:
            raise ValueError("Cannot determine backbone output dimension.")
        return backbone_out_dim

    @staticmethod
    def _expand_targets_for_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand 3-target tensors (green, total, gdm) to 5 targets for metric.

        Mapping (same as notebook logic):
            clover = gdm - green
            dead   = total - green - clover
            order  = [clover, dead, green, total, gdm]
        """
        green_true, green_pred = y_true[:, 0], y_pred[:, 0]
        total_true, total_pred = y_true[:, 1], y_pred[:, 1]
        gdm_true, gdm_pred = y_true[:, 2], y_pred[:, 2]

        clover_true = torch.clamp(gdm_true - green_true, min=0.0)
        clover_pred = torch.clamp(gdm_pred - green_pred, min=0.0)

        dead_true = torch.clamp(total_true - green_true - clover_true, min=0.0)
        dead_pred = torch.clamp(total_pred - green_pred - clover_pred, min=0.0)

        y_true_5 = torch.stack(
            [clover_true, dead_true, green_true, total_true, gdm_true], dim=1)
        y_pred_5 = torch.stack(
            [clover_pred, dead_pred, green_pred, total_pred, gdm_pred], dim=1)

        return y_true_5, y_pred_5

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        preds = self(batch['left_image'], batch['right_image'])
        targets = batch['targets']  # [B, 3] (green, total, gdm)
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
        preds = self(batch['left_image'], batch['right_image'])
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

        # remove NaNs and infs
        pred_ensemble = torch.nan_to_num(pred_ensemble, nan=0.0, posinf=250.0, neginf=0.0)
        pred_global = torch.nan_to_num(pred_global, nan=0.0, posinf=250.0, neginf=0.0)
        pred_local = torch.nan_to_num(pred_local, nan=0.0, posinf=250.0, neginf=0.0)

        # Clamp to non-negative
        pred_ensemble = torch.clamp(pred_ensemble, min=0.0, max=250.0)
        pred_global = torch.clamp(pred_global, min=0.0, max=250.0)
        pred_local = torch.clamp(pred_local, min=0.0, max=250.0)

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
        # Expand 3 targets -> 5 for metric
        targets_t = torch.tensor(targets)
        ens_t = torch.tensor(pred_ensemble)
        glob_t = torch.tensor(pred_global)
        loc_t = torch.tensor(pred_local)

        y_true_5, y_pred_ens_5 = self._expand_targets_for_metric(
            targets_t, ens_t)
        _, y_pred_glob_5 = self._expand_targets_for_metric(targets_t, glob_t)
        _, y_pred_loc_5 = self._expand_targets_for_metric(targets_t, loc_t)

        r2_ensemble = self.kaggle_score(y_true_5.numpy(), y_pred_ens_5.numpy())
        r2_global = self.kaggle_score(y_true_5.numpy(), y_pred_glob_5.numpy())
        r2_local = self.kaggle_score(y_true_5.numpy(), y_pred_loc_5.numpy())

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
        preds = self(batch['left_image'], batch['right_image'])

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
