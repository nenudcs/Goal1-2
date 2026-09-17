import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.medicalnet3d import MedicalNetEncoder


class DuplicateModel(nn.Module):
    """Siamese MedicalNet ResNet-10 with a shared case encoder."""

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = MedicalNetEncoder(in_channels=1)
        self.projection = nn.Linear(self.encoder.output_dim, embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(3.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode_series(self, series_tensor, series_batch_size=4):
        chunks = []
        batch_size = max(2, int(series_batch_size))
        split_sizes = []
        remaining = len(series_tensor)
        while remaining:
            size = min(batch_size, remaining)
            if remaining - size == 1 and size > 2:
                size -= 1
            split_sizes.append(size)
            remaining -= size
        for chunk in series_tensor.split(split_sizes):
            chunks.append(self.projection(self.encoder(chunk)))
        return torch.cat(chunks)

    def encode_case(self, series_tensor, series_batch_size=4):
        embedding = self.encode_series(series_tensor, series_batch_size).mean(0, keepdim=True)
        return F.normalize(embedding, dim=-1)

    def score_embeddings(self, a, b):
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (a * b).sum(dim=-1) + self.bias

    def similarity(self, xa, xb, series_batch_size=4):
        count_a = len(xa)
        embeddings = self.encode_series(torch.cat((xa, xb)), series_batch_size)
        a = F.normalize(embeddings[:count_a].mean(0, keepdim=True), dim=-1)
        b = F.normalize(embeddings[count_a:].mean(0, keepdim=True), dim=-1)
        return self.score_embeddings(a, b)
