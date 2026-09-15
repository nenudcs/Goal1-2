import torch
import torch.nn as nn
import torch.nn.functional as F

class Small3DEncoder(nn.Module):
    """V1 通用 3D encoder，便于先跑通流程。"""
    def __init__(self, in_channels=1, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.proj(x)

class VolumeClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim=128):
        super().__init__()
        self.encoder = Small3DEncoder(1, feature_dim)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)

class MultiModalClassifier(nn.Module):
    """
    T1CE/T2/FLAIR 三个模态共享 encoder。
    输入:
      x: [B,3,1,D,H,W]
      modality_mask: [B,3]
    缺失模态不会参与融合。
    """
    def __init__(self, feature_dim=128, heads=None):
        super().__init__()
        self.encoder = Small3DEncoder(1, feature_dim)
        self.modality_embedding = nn.Parameter(torch.randn(3, feature_dim) * 0.02)
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.heads = nn.ModuleDict()
        for name, out_dim in (heads or {}).items():
            self.heads[name] = nn.Linear(feature_dim, out_dim)

    def forward(self, x, modality_mask):
        B = x.shape[0]
        feats = []
        for i in range(3):
            fi = self.encoder(x[:, i])
            fi = fi + self.modality_embedding[i]
            feats.append(fi)
        feats = torch.stack(feats, dim=1)  # [B,3,F]

        mask = modality_mask.unsqueeze(-1).float()
        denom = mask.sum(1).clamp_min(1.0)
        fused = (feats * mask).sum(1) / denom
        fused = self.fusion(fused)

        return {name: head(fused) for name, head in self.heads.items()}
