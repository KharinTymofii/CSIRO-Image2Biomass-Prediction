import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Callable
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on important regions."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        return x * attention


class CrossViewAttention(nn.Module):
    """Cross-attention between left and right image features."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.q_left = nn.Linear(dim, dim)
        self.kv_right = nn.Linear(dim, dim * 2)

        self.q_right = nn.Linear(dim, dim)
        self.kv_left = nn.Linear(dim, dim * 2)

        self.proj_left = nn.Linear(dim, dim)
        self.proj_right = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, left_feat: torch.Tensor, right_feat: torch.Tensor):
        """
        Args:
            left_feat: [B, dim]
            right_feat: [B, dim]
        Returns:
            left_enhanced, right_enhanced
        """
        B, dim = left_feat.shape

        # Left queries, Right keys/values
        q_l = self.q_left(left_feat).view(B, 1, self.heads,
                                          dim // self.heads).transpose(1, 2)
        kv_r = self.kv_right(right_feat)
        k_r, v_r = kv_r.chunk(2, dim=-1)
        k_r = k_r.view(B, 1, self.heads, dim // self.heads).transpose(1, 2)
        v_r = v_r.view(B, 1, self.heads, dim // self.heads).transpose(1, 2)

        attn_l = (q_l @ k_r.transpose(-2, -1)) * self.scale
        attn_l = attn_l.softmax(dim=-1)
        attn_l = self.dropout(attn_l)

        left_enhanced = (attn_l @ v_r).transpose(1, 2).reshape(B, dim)
        left_enhanced = self.proj_left(left_enhanced)

        # Right queries, Left keys/values
        q_r = self.q_right(right_feat).view(
            B, 1, self.heads, dim // self.heads).transpose(1, 2)
        kv_l = self.kv_left(left_feat)
        k_l, v_l = kv_l.chunk(2, dim=-1)
        k_l = k_l.view(B, 1, self.heads, dim // self.heads).transpose(1, 2)
        v_l = v_l.view(B, 1, self.heads, dim // self.heads).transpose(1, 2)

        attn_r = (q_r @ k_l.transpose(-2, -1)) * self.scale
        attn_r = attn_r.softmax(dim=-1)
        attn_r = self.dropout(attn_r)

        right_enhanced = (attn_r @ v_l).transpose(1, 2).reshape(B, dim)
        right_enhanced = self.proj_right(right_enhanced)

        return left_enhanced, right_enhanced


class MultiScalePooling(nn.Module):
    """Extract features at multiple scales using different pooling strategies."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.AdaptiveAvgPool2d(2),
            nn.AdaptiveMaxPool2d(2),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C * 6] - concatenated multi-scale features
        """
        features = []
        for pool in self.pools:
            pooled = pool(x).flatten(1)
            features.append(pooled)
        return torch.cat(features, dim=1)


class RegressionHead(nn.Module):
    """Advanced regression head with residual connections."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_targets: int = 3,
        dropout: float = 0.2,
        num_layers: int = 3
    ):
        super().__init__()

        layers = []
        current_dim = in_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                # nn.BatchNorm1d(hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        # Final projection
        layers.append(nn.Linear(current_dim, num_targets))

        self.net = nn.Sequential(*layers)

        # Residual projection if needed
        self.residual = nn.Linear(
            in_dim, num_targets) if in_dim != num_targets else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x) * 0.1


class BiomassImprovedCNN(pl.LightningModule):
    """
    Improved CNN model for biomass prediction with:
    - Multi-scale feature extraction
    - Cross-view attention between left/right images
    - Spatial attention
    - Advanced regression head
    - Multiple prediction strategies (separate, fused, gated)
    """

    def __init__(
        self,
        kaggle_score: Callable,
        backbone_input_size: int,
        backbone_name: str = 'tf_efficientnetv2_s.in21k_ft_in1k',
        num_targets: int = 3,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        hidden_ratio: float = 0.5,
        dropout: float = 0.2,
        use_log_target: bool = True,
        freeze_backbone: bool = True,
        use_spatial_attention: bool = True,
        use_cross_attention: bool = True,
        fusion_method: str = 'gated',  # 'concat', 'gated', 'separate'
        scheduler: str = 'cosine',  # 'cosine', 'plateau'
    ):
        """
        Args:
            backbone_name: Backbone model name from timm
            num_targets: Number of regression targets (3: green, total, gdm)
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            hidden_ratio: Ratio for hidden layer size
            dropout: Dropout probability
            use_log_target: If True, predict log1p transformed targets
            freeze_backbone: If True, freeze backbone initially
            use_spatial_attention: Whether to use spatial attention
            use_cross_attention: Whether to use cross-view attention
            fusion_method: How to fuse left/right features
            scheduler: Learning rate scheduler type
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_targets = num_targets
        self.use_log_target = use_log_target
        self.fusion_method = fusion_method
        self.use_spatial_attention = use_spatial_attention
        self.use_cross_attention = use_cross_attention

        # Load backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )

        # Get backbone output dimension
        self.backbone_hidden_dim = self._get_backbone_output_dim(
            self.backbone, input_size=backbone_input_size
        )

        # Freeze/unfreeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Spatial attention
        if use_spatial_attention:
            self.spatial_attn = SpatialAttention(self.backbone_hidden_dim)

        # Multi-scale pooling
        self.multi_scale_pool = MultiScalePooling(self.backbone_hidden_dim)
        # 6x because: avg(1) + max(1) + avg(2x2) + max(2x2) = 1 + 1 + 4 + 4 = 10
        # Actually: 1 + 1 + 4 + 4 = 10 features per channel, so C * 10
        raw_pooled_dim = self.backbone_hidden_dim * 10

        # Projection to reduce dimensionality (critical for memory!)
        # pooled_dim = 12800
        pooled_dim = 512  # Much smaller dimension
        self.projection = nn.Sequential(
            nn.Linear(raw_pooled_dim, pooled_dim),
            nn.LayerNorm(pooled_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Cross-view attention
        if use_cross_attention:
            self.cross_attn = CrossViewAttention(
                pooled_dim,
                heads=8,
                dropout=dropout
            )

        # Determine feature dimension based on fusion method
        if fusion_method == 'concat':
            fused_dim = pooled_dim * 2
        elif fusion_method == 'separate':
            fused_dim = pooled_dim  # Each branch processes separately
        elif fusion_method == 'gated':
            fused_dim = pooled_dim * 2
            # Gating mechanism
            self.gate_left = nn.Sequential(
                nn.Linear(pooled_dim, pooled_dim),
                nn.Sigmoid()
            )
            self.gate_right = nn.Sequential(
                nn.Linear(pooled_dim, pooled_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        hidden_size = max(64, int(fused_dim * hidden_ratio))

        # Regression heads
        if fusion_method == 'separate':
            # Separate predictions for left and right, then average
            self.head_left = RegressionHead(
                pooled_dim, hidden_size, num_targets, dropout
            )
            self.head_right = RegressionHead(
                pooled_dim, hidden_size, num_targets, dropout
            )
            self.head_fused = RegressionHead(
                pooled_dim * 2, hidden_size * 2, num_targets, dropout
            )
        else:
            self.head = RegressionHead(
                fused_dim, hidden_size, num_targets, dropout
            )

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler

        self.validation_step_outputs = []
        self.kaggle_score = kaggle_score

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
        features = self.backbone.forward_features(x)

        # Apply spatial attention if enabled
        if self.use_spatial_attention:
            features = self.spatial_attn(features)

        # Multi-scale pooling
        pooled = self.multi_scale_pool(features)

        # Project to smaller dimension
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
        # Extract features
        left_feat = self.forward_features(left_img)  # [B, pooled_dim]
        right_feat = self.forward_features(right_img)  # [B, pooled_dim]

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
            # Gated fusion: each side modulates the other
            gate_l = self.gate_left(right_feat)
            gate_r = self.gate_right(left_feat)
            left_gated = left_feat * gate_l
            right_gated = right_feat * gate_r
            fused = torch.cat([left_gated, right_gated], dim=1)
            pred = self.head(fused)

        elif self.fusion_method == 'separate':
            # Three predictions: left, right, and fused
            pred_left = self.head_left(left_feat)
            pred_right = self.head_right(right_feat)
            fused = torch.cat([left_feat, right_feat], dim=1)
            pred_fused = self.head_fused(fused)
            # Ensemble: weighted average
            pred = (pred_left + pred_right + pred_fused * 2) / 4.0

        return {'prediction': pred}

    def compute_loss(self, preds: dict, targets: torch.Tensor) -> dict:
        """Compute Huber loss."""
        loss = F.huber_loss(preds['prediction'], targets, delta=1.0)
        return {'loss': loss}

    @staticmethod
    def _get_backbone_output_dim(backbone: nn.Module, input_size: int) -> int:
        """Utility to get backbone output dimension."""
        dummy = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            backbone_output = backbone.forward_features(dummy)

        if hasattr(backbone_output, 'shape'):
            # Assuming output is [B, C, H, W]
            backbone_out_dim = backbone_output.shape[1]
        else:
            raise ValueError("Cannot determine backbone output dimension.")

        return backbone_out_dim

    @staticmethod
    def _expand_targets_for_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand 3-target tensors (green, total, gdm) to 5 targets for metric.

        Mapping:
            green, total, gdm -> green, dead, clover, total, gdm
            clover = gdm - green
            dead = total - green - clover
            order = [clover, dead, green, total, gdm]
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
        targets = batch['targets']
        losses = self.compute_loss(preds, targets)

        self.log('train/loss', losses['loss'],
                 on_step=True, on_epoch=True, prog_bar=True)

        return losses['loss']

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        preds = self(batch['left_image'], batch['right_image'])
        targets = batch['targets']
        losses = self.compute_loss(preds, targets)

        self.log('val/loss', losses['loss'],
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

        # Expand 3 targets -> 5 for metric
        targets_t = torch.tensor(targets)
        preds_t = torch.tensor(preds)

        y_true_5, y_pred_5 = self._expand_targets_for_metric(
            targets_t, preds_t)

        # Compute competition metric (weighted R2)
        r2_score = self.kaggle_score(y_true_5.numpy(), y_pred_5.numpy())

        self.log('val/r2_score', r2_score, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.clear()

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Inference."""
        preds = self(batch['left_image'], batch['right_image'])

        # Convert from log space if needed
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
                    'monitor': 'val/r2_score',
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer
