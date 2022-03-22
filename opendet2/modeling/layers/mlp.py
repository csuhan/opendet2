import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if not hidden_dim:
            hidden_dim = in_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        feat_norm = F.normalize(feat, dim=1)
        return feat_norm


class ConvMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if not hidden_dim:
            hidden_dim = in_dim
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=3, stride=1, padding=1),
        )
        # Initialization
        for layer in self.head:
            if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        feat = self.head(x)
        feat_norm = F.normalize(feat, dim=1)
        return feat_norm