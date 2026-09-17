from pathlib import Path

import torch
import torch.nn as nn

from src.common.train_utils import _torch_load


def _conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size=3, stride=stride,
        padding=1, bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class MedicalNetEncoder(nn.Module):
    """MedicalNet-compatible 3D ResNet-10 encoder for one-channel volumes."""

    output_dim = 512

    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(
            in_channels, 64, kernel_size=7, stride=(1, 2, 2),
            padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=1)
        self.layer2 = self._make_layer(128, blocks=1, stride=2)
        self.layer3 = self._make_layer(256, blocks=1, stride=2)
        self.layer4 = self._make_layer(512, blocks=1, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        layers.extend(BasicBlock(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)


class MedicalNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = MedicalNetEncoder(in_channels=1)
        self.classifier = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))


def load_medicalnet_pretrained(encoder, checkpoint_path, device="cpu"):
    """Load matching encoder tensors from an extracted MedicalNet checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MedicalNet pretrained checkpoint not found: {checkpoint_path}")
    checkpoint = _torch_load(checkpoint_path, device)
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    own_state = encoder.state_dict()
    matched = {}
    for key, value in state.items():
        normalized = key
        for prefix in ("module.", "encoder.", "backbone."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        if normalized in own_state and own_state[normalized].shape == value.shape:
            matched[normalized] = value
    if not matched:
        raise ValueError(f"No compatible MedicalNet encoder tensors in {checkpoint_path}")
    encoder.load_state_dict(matched, strict=False)
    return len(matched), len(own_state)
