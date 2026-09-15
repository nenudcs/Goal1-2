"""Offline official backbones with sequence MIL and a symmetric pair head."""
import importlib.util
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import load_torch


def _strict_load(model, state):
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a state_dict")
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    expected = model.state_dict()
    mismatch = [key for key in expected.keys() & state.keys()
                if not isinstance(state[key], torch.Tensor) or expected[key].shape != state[key].shape]
    if expected.keys() != state.keys() or mismatch:
        raise ValueError(f"Checkpoint mismatch: missing={sorted(expected.keys()-state.keys())[:10]}, "
                         f"extra={sorted(state.keys()-expected.keys())[:10]}, shapes={mismatch[:10]}")
    model.load_state_dict(state, strict=True)


def _convnext(cfg, pretrained):
    settings = cfg["detection"]["models"]
    path = Path(settings["convnext_repo"]) / "models" / "convnext.py"
    if not path.is_file():
        raise FileNotFoundError(f"Official ConvNeXt source not found: {path}")
    spec = importlib.util.spec_from_file_location("competition_convnext", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.convnext_tiny(pretrained=False)
    if pretrained:
        checkpoint = load_torch(settings["convnext_weights"])
        _strict_load(model, checkpoint.get("model", checkpoint))
    model.head = nn.Identity()  # Original 1000-class head was validated before replacement.
    return model


def amp_context(cfg, device):
    # FP32 is the fallback; avoid unscaled FP16 gradients on older environments.
    enabled = bool(cfg["project"].get("amp", True) and device.type == "cuda"
                   and torch.cuda.is_bf16_supported())
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def normalize_volume(volume):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3 or min(volume.shape) < 1 or not np.isfinite(volume).all():
        raise ValueError("Expected finite nonempty DHW volume")
    nonzero = volume[np.abs(volume) > 1e-8]
    if nonzero.size == 0:
        raise ValueError("Image is entirely empty; cannot produce a valid prediction")
    lo, hi = np.percentile(nonzero, [.5, 99.5])
    if hi <= lo:
        lo, hi = float(volume.min()), float(volume.max())
    if hi <= lo:
        raise ValueError("Constant image cannot be normalized")
    return np.ascontiguousarray(np.clip((volume - lo) / (hi - lo), 0, 1))


def tile_starts(length, size, stride):
    if size < 1 or stride < 1 or stride > size:
        raise ValueError("Tile stride must be positive and no larger than tile size")
    if length <= size:
        return [0]
    return sorted(set(range(0, length - size + 1, stride)) | {length - size})


def view_locations(shape, size, stride):
    depth, height, width = shape
    for z in range(depth):
        yield z, None, None  # Entire frame, preserving aspect ratio with padding.
        if height > size or width > size:
            for y in tile_starts(height, size, stride):
                for x in tile_starts(width, size, stride):
                    yield z, y, x


def image_tensor(volume, location, size, adjacent=True):
    z, y, x = location
    indices = [max(0, z-1), z, min(len(volume)-1, z+1)] if adjacent else [z, z, z]
    tensor = torch.from_numpy(volume[indices].copy())
    if y is not None:
        tensor = tensor[:, y:y+size, x:x+size]
    height, width = tensor.shape[-2:]
    scale = size / max(height, width)
    shape = (max(1, round(height*scale)), max(1, round(width*scale)))
    tensor = F.interpolate(tensor[None], size=shape, mode="bilinear", align_corners=False)[0]
    dh, dw = size-shape[0], size-shape[1]
    tensor = F.pad(tensor, (dw//2, dw-dw//2, dh//2, dh-dh//2))
    mean = tensor.new_tensor([.485, .456, .406])[:, None, None]
    std = tensor.new_tensor([.229, .224, .225])[:, None, None]
    return (tensor - mean) / std


class AbnormalModel(nn.Module):
    def __init__(self, backbone, feature_dim=768, attention_dim=128):
        super().__init__()
        self.backbone = backbone
        self.attention = nn.Sequential(nn.Linear(feature_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))
        self.head = nn.Linear(feature_dim, 3)
        self.backbone_trainable = True

    def set_backbone_trainable(self, value):
        self.backbone_trainable = bool(value)
        self.backbone.requires_grad_(value)
        self.backbone.train(self.training and value)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.train(mode and self.backbone_trainable)
        return self

    def sequence_logits(self, volume, cfg, device, training=False, rng=None):
        settings = cfg["detection"]["abnormal"]
        size = int(settings["view_size"])
        volume = normalize_volume(volume)
        locations = list(view_locations(volume.shape, size, int(settings["tile_stride"])))
        if training:
            count = min(len(locations), int(settings["bag_views"]))
            if count < 1:
                raise ValueError("bag_views must be positive")
            locations = (rng or random).sample(locations, count)
        microbatch = int(settings["microbatch"])
        if microbatch < 1:
            raise ValueError("microbatch must be positive")
        maximum = denominator = numerator = None
        with nullcontext() if training else torch.no_grad():
            for start in range(0, len(locations), microbatch):
                images = torch.stack([image_tensor(volume, v, size) for v in locations[start:start+microbatch]]).to(device)
                with amp_context(cfg, device):
                    features = self.backbone.forward_features(images)
                features = features.float()
                scores = self.attention(features).squeeze(1)
                block_max = scores.max()
                new_max = block_max if maximum is None else torch.maximum(maximum, block_max)
                weights = (scores - new_max).exp()
                old_scale = 0.0 if maximum is None else (maximum - new_max).exp()
                numerator = (0 if numerator is None else numerator*old_scale) + (features*weights[:, None]).sum(0)
                denominator = (0 if denominator is None else denominator*old_scale) + weights.sum()
                maximum = new_max
            logits = self.head(numerator / denominator)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite sequence logits")
        return logits


def build_abnormal(cfg, device, pretrained=True):
    return AbnormalModel(_convnext(cfg, pretrained), attention_dim=cfg["detection"]["abnormal"]["attention_dim"]).to(device)


def build_duplicate_encoder(cfg, device):
    settings = cfg["detection"]["models"]
    repo = Path(settings["dino_repo"])
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"Official DINOv2 hubconf.py missing in {repo}")
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    encoder = torch.hub.load(str(repo), "dinov2_vits14", source="local", pretrained=False)
    state = load_torch(settings["dino_weights"])
    _strict_load(encoder, state.get("model", state))
    return encoder.requires_grad_(False).eval().to(device)


@torch.no_grad()
def encode_study(encoder, records, cfg, device):
    from .data import load_sequence
    embeddings = []
    microbatch = int(cfg["detection"]["duplicate"]["microbatch"])
    if microbatch < 1:
        raise ValueError("duplicate.microbatch must be positive")
    for record in records:
        volume, _ = load_sequence(record)
        volume = normalize_volume(volume)
        total = None
        for start in range(0, len(volume), microbatch):
            batch = torch.stack([image_tensor(volume, (z, None, None), 224, adjacent=False)
                                 for z in range(start, min(start+microbatch, len(volume)))]).to(device)
            with amp_context(cfg, device):
                output = encoder(batch)
            if output.ndim != 2 or output.shape[1] != 384 or not torch.isfinite(output).all():
                raise ValueError("DINOv2 must return finite [B,384] features")
            value = F.normalize(output.float(), dim=1).sum(0)
            total = value if total is None else total + value
        if total is None or total.norm() <= 1e-12:
            raise ValueError("Invalid sequence embedding")
        embeddings.append(F.normalize(total / len(volume), dim=0).cpu())
    if not embeddings:
        raise ValueError("Empty study")
    series = torch.stack(embeddings)
    case = series.mean(0)
    if case.norm() <= 1e-12:
        raise ValueError("Invalid study embedding")
    return {"case": F.normalize(case, dim=0), "series": series}


def pair_features(a, b):
    ea, eb = a["case"].float(), b["case"].float()
    similarities = a["series"].float() @ b["series"].float().T
    statistics = torch.stack([similarities.max(),
                              (similarities.max(0).values.mean()+similarities.max(1).values.mean())/2,
                              ea.new_tensor(min(len(a["series"]), len(b["series"]))/max(len(a["series"]), len(b["series"]))),
                              (ea*eb).sum()])
    result = torch.cat([(ea-eb).abs(), ea*eb, statistics])
    if not torch.isfinite(result).all():
        raise ValueError("Invalid pair features")
    return result


class DuplicateHead(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2*dim+4, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)
