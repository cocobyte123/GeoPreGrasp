from __future__ import annotations

import torch
from torch import nn


class PointNetScorer(nn.Module):
    """Small baseline scorer for P(success | object point cloud, angle, pose)."""

    def __init__(
        self,
        angle_dim: int = 6,
        pose_dim: int = 7,
        point_dim: int = 3,
        hidden_dim: int = 256,
        emb_dim: int = 256,
    ):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, emb_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(emb_dim + angle_dim + pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        point_cloud: torch.Tensor,
        angle: torch.Tensor,
        object_pose: torch.Tensor,
    ) -> torch.Tensor:
        point_features = self.point_encoder(point_cloud)
        global_features = point_features.max(dim=1).values
        features = torch.cat([global_features, angle, object_pose], dim=-1)
        return self.head(features)
