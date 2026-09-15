import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.basic3d import Small3DEncoder

class DuplicateModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = Small3DEncoder(1, embedding_dim)

    def encode_case(self, series_tensor):
        """
        series_tensor: [S,1,D,H,W]
        对 case 内所有 Series embedding 求平均，再 L2 normalize。
        """
        emb = self.encoder(series_tensor)
        emb = emb.mean(dim=0, keepdim=True)
        return F.normalize(emb, dim=-1)

    def similarity(self, xa, xb):
        ea = self.encode_case(xa)
        eb = self.encode_case(xb)
        return (ea * eb).sum(dim=-1)
