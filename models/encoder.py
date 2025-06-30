# models/encoder.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  # Remove FC
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):  # images: (B, 5, 3, H, W)
        B, S, C, H, W = images.size()
        images = images.view(B * S, C, H, W)
        features = self.resnet(images)
        features = features.view(B * S, -1)
        features = self.bn(self.linear(features))
        return features.view(B, S, -1)  # (B, 5, embed_size)
