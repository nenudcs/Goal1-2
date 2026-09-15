from pathlib import Path
import torch

def get_device(cfg):
    requested = cfg["project"].get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_checkpoint(model, optimizer, epoch, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
