from pathlib import Path
import torch

from src.models.basic3d import VolumeClassifier, MultiModalClassifier
from src.tasks.segmentation.model import build_segmentation_model
from src.tasks.duplicate.model import DuplicateModel
from src.tasks.characteristics.schema import BINARY_FIELDS, CATEGORICAL_FIELDS, LOCATION_CLASSES

HEADS = {
    **{k: 2 for k in BINARY_FIELDS},
    **{k: len(v) for k, v in CATEGORICAL_FIELDS.items()},
    "Location": len(LOCATION_CLASSES),
}

def _load_state(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

def load_all_models(cfg, device):
    ck = Path(cfg["paths"]["checkpoints_dir"])

    abnormal = _load_state(VolumeClassifier(3).to(device), ck/"abnormal"/"best.pth", device)
    sequence = _load_state(VolumeClassifier(3).to(device), ck/"sequence"/"best.pth", device)
    seg_core = _load_state(build_segmentation_model().to(device), ck/"segmentation"/"core"/"best.pth", device)
    seg_abnormal = _load_state(build_segmentation_model().to(device), ck/"segmentation"/"abnormal"/"best.pth", device)
    characteristics = _load_state(
        MultiModalClassifier(cfg["characteristics"]["feature_dim"], HEADS).to(device),
        ck/"characteristics"/"best.pth", device
    )
    duplicate = _load_state(
        DuplicateModel(cfg["duplicate"]["embedding_dim"]).to(device),
        ck/"duplicate"/"best.pth", device
    )

    return {
        "abnormal": abnormal,
        "sequence": sequence,
        "seg_core": seg_core,
        "seg_abnormal": seg_abnormal,
        "characteristics": characteristics,
        "duplicate": duplicate,
    }
