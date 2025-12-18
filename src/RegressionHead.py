import torch
import torch.nn as nn

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