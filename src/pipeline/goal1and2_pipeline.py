import json
import time
from pathlib import Path

import numpy as np
import torch

from src.common.train_utils import load_checkpoint
from src.data.nifti import load_preprocessed_volume
from src.data.paths import list_case_series
from src.models.medicalnet3d import MedicalNetClassifier
from src.pipeline.duplicate_pipeline import run_duplicate_pipeline
from src.tasks.duplicate.model import DuplicateModel


SPECIAL_DIRS = {"fake", "composition", "compositing", "duplicate"}


def discover_case_dirs(input_path):
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    return [
        path for path in sorted(input_path.iterdir())
        if path.is_dir() and path.name.lower() not in SPECIAL_DIRS
        and list_case_series(path)
    ]


def load_quality_model(cfg, device):
    model = MedicalNetClassifier(num_classes=3).to(device)
    checkpoint = load_checkpoint(
        model, cfg["abnormal"]["checkpoint_path"], device, strict=True
    )
    expected = list(cfg["abnormal"]["class_names"])
    if checkpoint.get("class_names", expected) != expected:
        raise ValueError("Quality checkpoint class order does not match config.")
    expected_shape = list(cfg["data"]["target_shape"])
    if checkpoint.get("target_shape", expected_shape) != expected_shape:
        raise ValueError("Quality checkpoint input shape does not match config.")
    model.eval()
    return model


def load_duplicate_model(cfg, device):
    model = DuplicateModel(cfg["duplicate"]["embedding_dim"]).to(device)
    checkpoint = load_checkpoint(
        model, cfg["duplicate"]["checkpoint_path"], device, strict=True
    )
    expected_shape = list(cfg["data"]["target_shape"])
    if checkpoint.get("target_shape", expected_shape) != expected_shape:
        raise ValueError("Duplicate checkpoint input shape does not match config.")
    model.eval()
    return model


@torch.no_grad()
def predict_quality_case(case_dir, model, cfg, device):
    probabilities = []
    for _, path in list_case_series(case_dir):
        volume, _ = load_preprocessed_volume(
            path, cfg["data"]["target_shape"],
            cfg["data"]["intensity_clip_percentiles"],
        )
        logits = model(volume.unsqueeze(0).to(device))
        probabilities.append(logits.softmax(1).squeeze(0).cpu().numpy())
    if not probabilities:
        raise RuntimeError(f"No readable NIfTI series in {case_dir}")
    values = np.asarray(probabilities)
    if cfg["data"].get("case_aggregation", "max") == "mean":
        return values.mean(0)
    return values.max(0)


def run_goal_pipeline(task, input_path, output_path, cfg, device):
    if task not in {"goal1", "goal2"}:
        raise ValueError("task must be goal1 or goal2")
    case_dirs = discover_case_dirs(input_path)
    if not case_dirs:
        raise ValueError(f"No readable NIfTI cases found in {input_path}")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    quality_model = load_quality_model(cfg, device)
    for case_dir in case_dirs:
        started = time.perf_counter()
        probabilities = predict_quality_case(case_dir, quality_model, cfg, device)
        case_output = output_path / case_dir.name
        case_output.mkdir(parents=True, exist_ok=True)
        prediction_path = case_output / "prediction.json"
        record = {}
        if prediction_path.is_file():
            record = json.loads(prediction_path.read_text(encoding="utf-8"))
        record.update({
            "AccessionNumber": case_dir.name,
            "IsNotHumanBodyProb": float(probabilities[1]),
            "IsStitchedProb": float(probabilities[2]),
            "ProcessingTime_ms": round((time.perf_counter() - started) * 1000, 3),
        })
        prediction_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if task == "goal2":
        duplicate_model = load_duplicate_model(cfg, device)
        run_duplicate_pipeline(
            case_dirs, output_path / "duplicate_pairs.jsonl",
            duplicate_model, cfg, device,
        )
    return case_dirs
