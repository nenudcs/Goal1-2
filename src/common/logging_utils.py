import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path

def build_logger(name: str):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(h)
    return logger

class CompetitionJSONLLogger:
    """
    比赛要求的 JSONL 训练日志。
    默认写入 /2026aicompetition/workspace/logs。
    如果当前环境无法创建该目录，则回退到项目 outputs/logs，方便本地调试。
    """
    def __init__(self, log_dir: str, filename: str, fallback_dir: str = "./outputs/logs"):
        self.requested_dir = Path(log_dir)
        try:
            self.requested_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.requested_dir / filename
        except Exception:
            fb = Path(fallback_dir)
            fb.mkdir(parents=True, exist_ok=True)
            self.path = fb / filename

    def write(
        self, epoch, step, phase, mode, data_source,
        loss=None, lr=None, checkpoint=None, pretrained_from=None, **metrics
    ):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "epoch": epoch,
            "step": step,
            "phase": phase,
            "mode": mode,
            "data_source": data_source,
            "loss": None if loss is None else float(loss),
            "lr": None if lr is None else float(lr),
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "pretrained_from": None if pretrained_from is None else str(pretrained_from),
        }
        for key, value in metrics.items():
            value = value.item() if hasattr(value, "item") else value
            record[key] = None if isinstance(value, float) and not math.isfinite(value) else value

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
