from pathlib import Path
import yaml

def load_config(path: str):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 将相对路径统一解析为 project 根目录相对路径
    project_root = path.resolve().parents[1]
    cfg["_project_root"] = str(project_root)

    path_keys = cfg["paths"] if cfg.get("detection", {}).get("enabled") else (
        "labels_dir", "checkpoints_dir", "output_dir")
    for key in path_keys:
        if cfg["paths"][key] is None:
            continue
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            cfg["paths"][key] = str((project_root / p).resolve())

    if cfg.get("detection", {}).get("enabled"):
        for section in ("models", "data"):
            for key, value in cfg["detection"][section].items():
                if value and (key.endswith("_file") or section == "models"):
                    p = Path(value)
                    if not p.is_absolute():
                        cfg["detection"][section][key] = str((project_root / p).resolve())

    return cfg
