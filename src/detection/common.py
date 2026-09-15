"""Small shared utilities for reproducible, container-only detection runs."""
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def source_hash(directory):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(root.rglob("*.py"))
    if not files:
        raise ValueError(f"No Python model source under {root}")
    return fingerprint({str(p.relative_to(root)): file_hash(p) for p in files
                        if ".git" not in p.parts and "__pycache__" not in p.parts})


def require_container():
    if os.name != "posix" or not Path("/2026aicompetition/workspace").is_dir():
        raise RuntimeError("本系统的运行与验收必须在赛事容器 /2026aicompetition/workspace 内进行")


def cuda_device(cfg):
    import torch
    require_container()
    if Path(cfg["paths"]["competition_log_dir"]).resolve() != Path("/2026aicompetition/workspace/logs"):
        raise ValueError("赛事运行的日志目录必须为 /2026aicompetition/workspace/logs")
    if cfg["project"].get("device") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("赛事配置要求 CUDA；不会静默退回 CPU")
    return torch.device("cuda")


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    import numpy as np
    import torch
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_torch(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def model_identity(cfg, branch=None):
    paths = cfg["detection"]["models"]
    if branch is not None:
        prefix = "convnext" if branch == "abnormal" else "dino"
        paths = {key: value for key, value in paths.items() if key.startswith(prefix)}
    return {key: source_hash(value) if key.endswith("_repo") else file_hash(value)
            for key, value in paths.items()}


def experiment_identity(cfg, manifest, branch, identities):
    return fingerprint({"format": 1, "manifest": manifest_identity(manifest, branch),
                        "model_sources": identities, "branch": branch,
                        "settings": branch_settings(cfg, branch), "seed": cfg["project"]["seed"],
                        "implementation": source_hash(Path(__file__).parent)})


def branch_settings(cfg, branch):
    return {k: v for k, v in cfg["detection"][branch].items() if k not in {"max_epochs", "patience"}}


def manifest_identity(manifest, branch=None):
    value = {"records": manifest["records"], "splits": manifest["splits"]}
    if branch == "duplicate":
        value.update(pairs=manifest["pairs"], negative_policy=manifest["negative_policy"])
    return fingerprint(value)
