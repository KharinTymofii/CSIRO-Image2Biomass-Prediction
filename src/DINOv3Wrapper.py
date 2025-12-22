import torch
import torch.nn as nn

class DINOv3InferenceWrapper(nn.Module):
    """Wrapper for DINOv3 models for TorchScript/ONNX export."""

    def __init__(self, lightning_model, img_size: int, mean: list, std: list):
        super().__init__()
        self.backbone = lightning_model.backbone
        self.projection = lightning_model.projection
        self.cross_attn = lightning_model.cross_attn if hasattr(
            lightning_model, 'cross_attn') else None

        # Handle different fusion methods
        self.fusion_method = lightning_model.hparams.fusion_method
        if self.fusion_method == 'gated':
            self.gate_left = lightning_model.gate_left
            self.gate_right = lightning_model.gate_right
            self.head = lightning_model.head
        elif self.fusion_method == 'separate':
            self.head_left = lightning_model.head_left
            self.head_right = lightning_model.head_right
            self.head_fused = lightning_model.head_fused
        else:  # concat
            self.head = lightning_model.head

        self.use_log_target = lightning_model.use_log_target
        self.use_cross_attention = lightning_model.hparams.use_cross_attention

        # Register config as constants for TorchScript
        self.register_buffer('_img_size', torch.tensor(
            img_size, dtype=torch.int32))
        self.register_buffer('_mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('_std', torch.tensor(std, dtype=torch.float32))

    @property
    def img_size(self) -> int:
        """Get image size."""
        return int(self._img_size.item())

    @property
    def mean(self) -> list:
        """Get normalization mean."""
        return self._mean.tolist()

    @property
    def std(self) -> list:
        """Get normalization std."""
        return self._std.tolist()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and pool features from DINOv3 backbone."""
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        features = outputs.last_hidden_state  # [B, N, C]
        pooled = features.mean(dim=1)  # [B, C]
        pooled = self.projection(pooled)
        return pooled

    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only predictions."""
        # Extract and pool features
        left_feat = self.forward_features(left_image)
        right_feat = self.forward_features(right_image)

        # Cross-view attention
        if self.use_cross_attention and self.cross_attn is not None:
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
            pred = (pred_left + pred_right + pred_fused * 2) / 4.0

        # Convert from log space if needed
        if self.use_log_target:
            pred = torch.expm1(pred)

        return pred
