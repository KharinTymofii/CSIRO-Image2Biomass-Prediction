import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Callable
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .CrossViewAttention import CrossViewAttention
from .RegressionHead import RegressionHead


class BiomassTransformer(pl.LightningModule):
    """
    Vision Transformer model for biomass prediction with:
    - Swin Transformer / DINOv2 backbone
    - Cross-view attention between left/right images
    - Simple global pooling (no spatial operations)
    - Multiple fusion strategies (concat, gated, separate)

    Optimized for Vision Transformers that output [B, H, W, C] or [B, N, C] formats.
    """

    def __init__(
        self,
        kaggle_score: Callable,
        backbone_input_size: int,
        backbone_name: str = 'swinv2_base_window8_256',
        num_targets: int = 3,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        hidden_ratio: float = 0.5,
        dropout: float = 0.2,
        use_log_target: bool = True,
        freeze_backbone: bool = True,
        use_spatial_attention: bool = False,
        use_cross_attention: bool = True,
        fusion_method: str = 'gated',  # 'concat', 'gated', 'separate'
        scheduler: str = 'cosine',  # 'cosine', 'plateau'
    ):
        """
        Args:
            kaggle_score: Competition metric function
            backbone_input_size: Input image size
            backbone_name: Transformer backbone from timm (swin, dinov2, etc.)
            num_targets: Number of regression targets (3: green, total, gdm)
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            hidden_ratio: Ratio for hidden layer size
            dropout: Dropout probability
            use_log_target: If True, predict log1p transformed targets
            freeze_backbone: If True, freeze backbone initially
            use_cross_attention: Whether to use cross-view attention
            fusion_method: How to fuse left/right features
            scheduler: Learning rate scheduler type
        """
        super().__init__()
        self.save_hyperparameters()

        self.kaggle_score = kaggle_score
        self.num_targets = num_targets
        self.use_log_target = use_log_target
        self.fusion_method = fusion_method
        self.use_cross_attention = use_cross_attention

        # Load backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool=''  # We'll do custom pooling
        )

        self._freeze_backbone = freeze_backbone

        # Get backbone output dimension
        self.backbone_hidden_dim = self._get_backbone_output_dim(
            self.backbone, input_size=backbone_input_size
        )

        print(f"Backbone output dimension: {self.backbone_hidden_dim}")

        # Freeze/unfreeze backbone
        if self._freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Feature dimension after global pooling
        feature_dim = self.backbone_hidden_dim

        # Optional: Add projection layer to reduce dimensions
        if self.backbone_hidden_dim > 2048:
            # For large models (e.g., Swin-L with 1536 dims), add projection
            feature_dim = 1024
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_hidden_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.projection = nn.Identity()

        # Cross-view attention
        if use_cross_attention:
            self.cross_attn = CrossViewAttention(
                feature_dim,
                heads=8,
                dropout=dropout
            )

        # Determine fused dimension based on fusion method
        if fusion_method == 'concat':
            fused_dim = feature_dim * 2
        elif fusion_method == 'separate':
            fused_dim = feature_dim
        elif fusion_method == 'gated':
            fused_dim = feature_dim * 2
            # Gating mechanism
            self.gate_left = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid()
            )
            self.gate_right = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        hidden_size = max(128, int(fused_dim * hidden_ratio))

        # Regression heads
        if fusion_method == 'separate':
            self.head_left = RegressionHead(
                feature_dim, hidden_size, num_targets, dropout
            )
            self.head_right = RegressionHead(
                feature_dim, hidden_size, num_targets, dropout
            )
            self.head_fused = RegressionHead(
                feature_dim * 2, hidden_size * 2, num_targets, dropout
            )
        else:
            self.head = RegressionHead(
                fused_dim, hidden_size, num_targets, dropout
            )

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler

        self.validation_step_outputs = []

    @staticmethod
    def _get_backbone_output_dim(backbone: nn.Module, input_size: int) -> int:
        """Get backbone output dimension for Transformers."""
        dummy = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            output = backbone.forward_features(dummy)  # type: ignore

        print(f"DEBUG: Backbone output shape = {output.shape}")

        if len(output.shape) == 4:
            # Swin format: [B, H, W, C] or [B, C, H, W]
            # Heuristic: last dimension is usually channels for Swin
            if output.shape[3] > output.shape[1]:
                # NHWC format (Swin): [B, H, W, C]
                dim = output.shape[3]
                print(f"DEBUG: Detected NHWC format, C = {dim}")
            else:
                # NCHW format
                dim = output.shape[1]
                print(f"DEBUG: Detected NCHW format, C = {dim}")
        elif len(output.shape) == 3:
            # [B, N, C] format (standard ViT/DINOv2)
            dim = output.shape[2]
            print(f"DEBUG: Detected [B, N, C] format, C = {dim}")
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")

        return dim
    
    def train(self, mode: bool = True):
        """Override train() to keep backbone in eval mode if frozen."""
        super().train(mode)
        if self._freeze_backbone:
            self.backbone.eval()
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and pool features from Transformer backbone."""
        # Use torch.no_grad() for frozen backbone during training
        if self._freeze_backbone and self.training:
            with torch.no_grad():
                features = self.backbone.forward_features(x)  # type: ignore
        else:
            features = self.backbone.forward_features(x)  # type: ignore

        # Handle different output formats and apply global pooling
        if len(features.shape) == 4:
            # Swin format: [B, H, W, C] or [B, C, H, W]
            if features.shape[3] > features.shape[1]:
                # NHWC: [B, H, W, C] -> pool over H, W
                pooled = features.mean(dim=[1, 2])  # [B, C]
            else:
                # NCHW: [B, C, H, W] -> pool over H, W
                pooled = features.mean(dim=[2, 3])  # [B, C]

        elif len(features.shape) == 3:
            # [B, N, C] format - pool over tokens
            # Option 1: Use CLS token if present (token 0)
            # Option 2: Average all tokens
            pooled = features.mean(dim=1)  # [B, C]

        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")

        # Apply projection if needed
        pooled = self.projection(pooled)

        return pooled

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass with dual-image processing.

        Args:
            left_img: Left image tensor [B, 3, H, W]
            right_img: Right image tensor [B, 3, H, W]

        Returns:
            Dict with predictions
        """
        # Extract and pool features
        left_feat = self.forward_features(left_img)  # [B, feature_dim]
        right_feat = self.forward_features(right_img)  # [B, feature_dim]

        # Cross-view attention
        if self.use_cross_attention:
            left_enhanced, right_enhanced = self.cross_attn(
                left_feat, right_feat)
            left_feat = left_feat + left_enhanced
            right_feat = right_feat + right_enhanced

        # Fusion strategy
        if self.fusion_method == 'concat':
            fused = torch.cat([left_feat, right_feat], dim=1)
            pred = self.head(fused)

        elif self.fusion_method == 'gated':
            gate_l = self.gate_left(right_feat)
            gate_r = self.gate_right(left_feat)
            left_gated = left_feat * gate_l
            right_gated = right_feat * gate_r
            fused = torch.cat([left_gated, right_gated], dim=1)
            pred = self.head(fused)

        elif self.fusion_method == 'separate':
            pred_left = self.head_left(left_feat)
            pred_right = self.head_right(right_feat)
            fused = torch.cat([left_feat, right_feat], dim=1)
            pred_fused = self.head_fused(fused)
            # Weighted ensemble
            pred = (pred_left + pred_right + pred_fused * 2) / 4.0

        return {'prediction': pred}

    def compute_loss(self, preds: dict, targets: torch.Tensor) -> dict:
        """Compute Huber loss."""
        loss = F.huber_loss(preds['prediction'], targets, delta=1.0)
        return {'loss': loss}

    @staticmethod
    def _expand_targets_for_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand 3-target tensors to 5 targets for competition metric."""
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
        targets = batch['targets']
        losses = self.compute_loss(preds, targets)

        self.log('train_loss', losses['loss'],
                 on_step=True, on_epoch=True, prog_bar=True)
        return losses['loss']

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        preds = self(batch['left_image'], batch['right_image'])
        targets = batch['targets']
        losses = self.compute_loss(preds, targets)

        self.log('val_loss', losses['loss'],
                 on_step=False, on_epoch=True, prog_bar=True)

        # Convert back from log space if needed
        if self.use_log_target:
            pred = torch.expm1(preds['prediction'])
            targets_original = torch.expm1(targets)
        else:
            pred = preds['prediction']
            targets_original = targets

        # Clean predictions
        pred = torch.nan_to_num(pred, nan=0.0, posinf=250.0, neginf=0.0)
        pred = torch.clamp(pred, min=0.0, max=250.0)

        # Store for epoch-end metrics
        self.validation_step_outputs.append({
            'pred': pred.detach().cpu(),
            'targets': targets_original.detach().cpu()
        })

        return losses['loss']

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return

        # Concatenate all predictions
        preds = torch.cat([x['pred']
                          for x in self.validation_step_outputs], dim=0).numpy()
        targets = torch.cat([x['targets']
                            for x in self.validation_step_outputs], dim=0).numpy()

        # Expand to 5 targets for metric
        targets_t = torch.tensor(targets)
        preds_t = torch.tensor(preds)
        y_true_5, y_pred_5 = self._expand_targets_for_metric(
            targets_t, preds_t)

        # Compute competition metric
        r2_score = self.kaggle_score(y_true_5.numpy(), y_pred_5.numpy())
        self.log('val_r2_score', r2_score, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.clear()

    @torch.no_grad()
    @torch.inference_mode()
    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Inference."""
        preds = self(batch['left_image'], batch['right_image'])

        if self.use_log_target:
            final_pred = torch.expm1(preds['prediction'])
        else:
            final_pred = preds['prediction']

        final_pred = torch.clamp(final_pred, min=0.0, max=250.0)
        return final_pred

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.scheduler_type == 'cosine':
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
        elif self.scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=3,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_r2_score',
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer
