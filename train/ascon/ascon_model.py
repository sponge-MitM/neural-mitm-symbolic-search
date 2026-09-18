'Module object.'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================

# ============================================================

class AsconMLP(nn.Module):
    'Class AsconMLP.'
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2,
                 n_aux_tasks=0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        self.encoding_dim = prev_dim


        self.reg_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )


        self.rank_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )


        if n_aux_tasks > 0:
            self.aux_head = nn.Linear(prev_dim, n_aux_tasks)
        else:
            self.aux_head = None

    def forward(self, tensor_x, scalar_x=None):
        if scalar_x is None:
            x = tensor_x
        else:
            x = scalar_x

        h = self.encoder(x)
        reg_out = self.reg_head(h).squeeze(-1)
        rank_out = self.rank_head(h).squeeze(-1)
        aux_out = self.aux_head(h) if self.aux_head is not None else None
        return {'reg': reg_out, 'rank': rank_out, 'aux': aux_out, 'encoding': h}


# ============================================================

# ============================================================

class CyclicPad2d(nn.Module):
    'Class CyclicPad2d.'
    def __init__(self, pad_z, pad_x):
        super().__init__()
        self.pad_z = pad_z
        self.pad_x = pad_x

    def forward(self, x):
        # x: (B, C, Z, X)
        B, C, Z, X = x.shape

        # Z: cyclic pad
        if self.pad_z > 0:
            prefix = x[:, :, -self.pad_z:, :]
            suffix = x[:, :, :self.pad_z, :]
            x = torch.cat([prefix, x, suffix], dim=2)

        # X: zero pad
        if self.pad_x > 0:
            pad = torch.zeros(B, C, Z + 2*self.pad_z, self.pad_x,
                             device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x, pad], dim=3)

        return x


class CyclicConv2d(nn.Module):
    'Class CyclicConv2d.'
    def __init__(self, in_ch, out_ch, kernel_z=3, kernel_x=3, stride=1):
        super().__init__()
        self.pad_z = kernel_z // 2
        self.pad_x = kernel_x // 2
        self.padder = CyclicPad2d(self.pad_z, self.pad_x)
        self.conv = nn.Conv2d(in_ch, out_ch,
                              kernel_size=(kernel_z, kernel_x),
                              stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.padder(x)
        return self.conv(x)


class AsconCNN(nn.Module):
    'Class AsconCNN.'
    def __init__(self, scalar_dim, n_aux_tasks=0, tensor_channels=1):
        super().__init__()
        in_ch = tensor_channels


        self.conv1 = CyclicConv2d(in_ch, 16, kernel_z=3, kernel_x=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = CyclicConv2d(16, 32, kernel_z=3, kernel_x=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = CyclicConv2d(32, 64, kernel_z=3, kernel_x=3)
        self.bn3 = nn.BatchNorm2d(64)


        self.pool = nn.AdaptiveAvgPool2d((8, 2))  # (Z,X) -> (8,2)

        cnn_out_dim = 64 * 8 * 2  # 1024


        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        combined_dim = cnn_out_dim + 32


        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )


        self.reg_head = nn.Linear(64, 1)
        self.rank_head = nn.Linear(64, 1)
        if n_aux_tasks > 0:
            self.aux_head = nn.Linear(64, n_aux_tasks)
        else:
            self.aux_head = None

    def forward(self, tensor, scalar_features):
        'Function forward.'
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)  # (B, 1, Z, X)

        x = F.relu(self.bn1(self.conv1(tensor)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.flatten(1)

        s = self.scalar_encoder(scalar_features)
        combined = torch.cat([x, s], dim=1)

        h = self.fc(combined)

        reg_out = self.reg_head(h).squeeze(-1)
        rank_out = self.rank_head(h).squeeze(-1)
        aux_out = self.aux_head(h) if self.aux_head is not None else None

        return {'reg': reg_out, 'rank': rank_out, 'aux': aux_out, 'encoding': h}


# ============================================================

# ============================================================

def pairwise_ranking_loss(rank_pred, y_true, margin=2.0):
    'Function pairwise ranking loss.'
    n = rank_pred.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=rank_pred.device)

    idx_i = torch.randperm(n, device=rank_pred.device)[:n // 2]
    idx_j = torch.randperm(n, device=rank_pred.device)[:n // 2]

    pred_diff = rank_pred[idx_i] - rank_pred[idx_j]
    true_diff = y_true[idx_i] - y_true[idx_j]

    loss = F.relu(margin - pred_diff * true_diff.sign())
    return loss.mean()


def build_model(model_type, scalar_dim, n_aux_tasks=0):
    'Function build model.'
    if model_type == 'mlp':
        return AsconMLP(scalar_dim, n_aux_tasks=n_aux_tasks)
    elif model_type == 'cnn':
        return AsconCNN(scalar_dim, n_aux_tasks=n_aux_tasks)
    else:
        raise ValueError(f"unknown model type: {model_type}")
