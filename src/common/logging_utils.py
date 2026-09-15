import json
import logging
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
    strict=True 时创建或写入失败直接报错；仅旧流程保留目录创建失败时的回退。
    """
    def __init__(self, log_dir: str, filename: str, fallback_dir: str = "./outputs/logs", strict=False):
        self.requested_dir = Path(log_dir)
        try:
            self.requested_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.requested_dir / filename
        except Exception:
            if strict:
                raise
            fb = Path(fallback_dir)
            fb.mkdir(parents=True, exist_ok=True)
            self.path = fb / filename

    def write(
        self, epoch, step, phase, mode, data_source,
        loss=None, lr=None, checkpoint=None, pretrained_from=None
    ):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "epoch": epoch,
            "step": step,
            "phase": phase,
            "mode": mode,
            "data_source": data_source,
        }
        if loss is not None:
            record["loss"] = float(loss)
        if lr is not None:
            record["lr"] = float(lr)
        if checkpoint is not None:
            record["checkpoint"] = str(checkpoint)
        if pretrained_from is not None:
            record["pretrained_from"] = str(pretrained_from)

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
