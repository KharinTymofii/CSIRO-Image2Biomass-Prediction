import torch
import torch.nn as nn


class InferenceWrapper(nn.Module):
    """Wrapper for CNN models (BiomassImprovedCNN) for TorchScript/ONNX export."""

    def __init__(self, lightning_model, img_size: int, mean: list, std: list):
        super().__init__()
        self.backbone = lightning_model.backbone
        self.spatial_attn = lightning_model.spatial_attn
        self.multi_scale_pool = lightning_model.multi_scale_pool
        self.projection = lightning_model.projection
        self.cross_attn = lightning_model.cross_attn
        self.gate_left = lightning_model.gate_left
        self.gate_right = lightning_model.gate_right
        self.head = lightning_model.head
        self.use_log_target = lightning_model.use_log_target
        self.use_spatial_attention = lightning_model.hparams.use_spatial_attention
        self.use_cross_attention = lightning_model.hparams.use_cross_attention
        self.fusion_method = lightning_model.hparams.fusion_method
        
        # Register config as constants for TorchScript
        self.register_buffer('_img_size', torch.tensor(img_size, dtype=torch.int32))
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

    def _process_features(self, features: torch.Tensor) -> torch.Tensor:
        """Process backbone features (handles both CNN and ViT outputs)."""
        # Handle ViT/Swin output format
        if len(features.shape) == 3:
            B, N, C = features.shape

            # Remove CLS token if present
            has_cls = (N - 1) == int((N - 1) ** 0.5) ** 2
            if has_cls:
                features = features[:, 1:, :]
                N = N - 1

            H = W = int(N ** 0.5)

            # Reshape to spatial format
            if H * W == N:
                features = features.transpose(1, 2).reshape(B, C, H, W)
            else:
                # Fallback: global pool
                features = features.mean(dim=1).unsqueeze(-1).unsqueeze(-1)

        # Apply spatial attention if enabled and valid
        if self.use_spatial_attention and self.spatial_attn is not None and features.shape[2] > 1:
            features = self.spatial_attn(features)

        return features

    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only predictions."""
        # Extract features from backbone
        left_feat = self.backbone.forward_features(left_image)
        right_feat = self.backbone.forward_features(right_image)

        # Process features (handles both CNN and ViT)
        left_feat = self._process_features(left_feat)
        right_feat = self._process_features(right_feat)

        # Multi-scale pooling
        left_pooled = self.multi_scale_pool(left_feat)
        right_pooled = self.multi_scale_pool(right_feat)

        # Project to lower dimension
        left_proj = self.projection(left_pooled)
        right_proj = self.projection(right_pooled)

        # Apply cross-attention if enabled
        if self.use_cross_attention and self.cross_attn is not None:
            left_enhanced, right_enhanced = self.cross_attn(
                left_proj, right_proj)
            left_proj = left_proj + left_enhanced
            right_proj = right_proj + right_enhanced

        # Fusion strategy
        if self.fusion_method == 'concat':
            fused = torch.cat([left_proj, right_proj], dim=1)
            pred = self.head(fused)

        elif self.fusion_method == 'gated':
            gate_l = self.gate_left(right_proj)
            gate_r = self.gate_right(left_proj)
            left_gated = left_proj * gate_l
            right_gated = right_proj * gate_r
            fused = torch.cat([left_gated, right_gated], dim=1)
            pred = self.head(fused)

        else:  # 'separate' - fallback to average
            fused = (left_proj + right_proj) / 2
            pred = self.head(fused)

        # Convert from log space if needed
        if self.use_log_target:
            pred = torch.expm1(pred)

        return pred