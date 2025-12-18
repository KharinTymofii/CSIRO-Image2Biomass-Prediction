import torch
import torch.nn as nn

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