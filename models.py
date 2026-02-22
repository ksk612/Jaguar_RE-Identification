"""
ReID backbone (embedding only) + ArcFace head for Jaguar identification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ReIDModel(nn.Module):
    """
    Backbone that outputs a single embedding tensor (no classification head).
    Use with ArcFace head for training; use embedding only for retrieval.
    """

    def __init__(self, embedding_dim=512, num_classes=None, backbone="resnet50", pretrained=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        # num_classes ignored here; used only for compatibility if someone passes it

        if backbone == "resnet50":
            base = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone == "resnet101":
            base = models.resnet101(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.fc(feat)
        return emb  # single tensor, no tuple


class ArcFace(nn.Module):
    """ArcFace margin head. Use embeddings + labels in forward for training."""

    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)
        logits = F.linear(embeddings, W)

        if labels is None:
            return logits

        # ArcFace margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = F.one_hot(labels, self.num_classes).float()
        target_logits = torch.cos(theta + self.m * one_hot)
        if self.easy_margin:
            target_logits = torch.where(
                one_hot.bool(),
                target_logits,
                torch.cos(theta)
            )
        logits = logits + one_hot * (target_logits - torch.cos(theta))
        return self.s * logits
