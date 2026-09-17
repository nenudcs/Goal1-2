from pathlib import Path
import yaml


def _is_absolute(value):
    text = str(value)
    return Path(text).is_absolute() or text.startswith("/")


def load_config(path: str):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 将相对路径统一解析为 project 根目录相对路径
    project_root = path.resolve().parents[1]
    cfg["_project_root"] = str(project_root)

    for key in (
        "annotation_root", "labels_dir", "checkpoints_dir", "output_dir",
        "competition_log_dir", "answer_root",
    ):
        value = cfg["paths"][key]
        p = Path(value)
        if not _is_absolute(value):
            cfg["paths"][key] = str((project_root / p).resolve())

    for section in ("abnormal", "duplicate"):
        for key in ("checkpoint_path", "pretrained_path", "resume_path"):
            value = cfg.get(section, {}).get(key, "")
            if value and not _is_absolute(value):
                cfg[section][key] = str((project_root / value).resolve())

    return cfg
