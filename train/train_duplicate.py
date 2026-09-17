from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.common.logging_utils import CompetitionJSONLLogger, build_logger
from src.common.seed import seed_everything
from src.common.train_utils import get_device, load_checkpoint, save_checkpoint
from src.data.paths import PathResolver
from src.data.split import split_duplicate_components
from src.models.medicalnet3d import load_medicalnet_pretrained
from src.tasks.duplicate.dataset import DuplicatePairDataset, duplicate_collate, read_positive_pairs
from src.tasks.duplicate.model import DuplicateModel


def duplicate_metrics(labels, probabilities):
    result = {
        "auroc": float("nan"), "auprc": float("nan"),
        "recall_at_10pct_fpr": 0.0, "precision_at_15pct_recall": 0.0,
    }
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(set(labels.tolist())) < 2:
        return result
    result["auroc"] = roc_auc_score(labels, probabilities)
    result["auprc"] = average_precision_score(labels, probabilities)
    thresholds = np.r_[np.inf, np.unique(probabilities)[::-1], -np.inf]
    recalls, precisions = [], []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        tn = int(np.sum(~predicted & (labels == 0)))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        fpr = fp / max(fp + tn, 1)
        if fpr <= 0.10:
            recalls.append(recall)
        if recall >= 0.15:
            precisions.append(precision)
    result["recall_at_10pct_fpr"] = max(recalls, default=0.0)
    result["precision_at_15pct_recall"] = max(precisions, default=0.0)
    return result


def _batch_logits(model, xa_list, xb_list, device, series_batch_size):
    return torch.cat([
        model.similarity(xa.to(device), xb.to(device), series_batch_size)
        for xa, xb in zip(xa_list, xb_list)
    ])


def main(cfg):
    logger = build_logger("train_duplicate")
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)
    manifest = Path(cfg["paths"]["labels_dir"]) / cfg["duplicate"]["manifest"]
    positive_pairs = read_positive_pairs(manifest)
    resolver = PathResolver(
        cfg["paths"]["annotation_root"], cfg["data"].get("source_dirs")
    )
    all_ids = [
        identifier for identifier in resolver.duplicate_accessions()
        if resolver.list_case_series(resolver.duplicate_accession_dir(identifier))
    ]
    train_ids, val_ids = split_duplicate_components(
        all_ids, positive_pairs, cfg["split"]["val_ratio"],
        cfg["split"]["random_state"],
    )
    common = (
        manifest, cfg["paths"]["annotation_root"], cfg["data"]["target_shape"],
        cfg["data"]["intensity_clip_percentiles"], cfg["duplicate"]["negative_ratio"],
    )
    train_ds = DuplicatePairDataset(
        *common, cfg["project"]["seed"], train_ids, cfg["data"].get("source_dirs")
    )
    val_ds = DuplicatePairDataset(
        *common, cfg["project"]["seed"] + 1, val_ids, cfg["data"].get("source_dirs")
    )
    loader_options = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": cfg["project"]["num_workers"],
        "pin_memory": cfg["project"].get("pin_memory", True) and device.type == "cuda",
        "collate_fn": duplicate_collate,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)
    model = DuplicateModel(cfg["duplicate"]["embedding_dim"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    positives = sum(item[2] for item in train_ds.items)
    negatives = len(train_ds) - positives
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], device=device)
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg["project"].get("amp", True) and device.type == "cuda"
    )
    start_epoch = 1
    pretrained_from = cfg["duplicate"].get("pretrained_path", "")
    resume = cfg["duplicate"].get("resume_path") or cfg["training"].get("resume", "")
    if resume:
        state = load_checkpoint(model, resume, device, optimizer, strict=True)
        start_epoch = int(state.get("epoch", 0)) + 1
    elif pretrained_from:
        matched, total = load_medicalnet_pretrained(model.encoder, pretrained_from, device)
        logger.info("Loaded MedicalNet encoder tensors: %d/%d", matched, total)

    json_logger = CompetitionJSONLLogger(
        cfg["paths"]["competition_log_dir"], "duplicate_train.jsonl",
        fallback_dir=str(Path(cfg["paths"]["output_dir"]) / "logs"),
    )
    logger.info(
        "Goal 2 duplicate training | model=Siamese MedicalNet ResNet-10 "
        "device=%s train_cases=%d val_cases=%d train_pairs=%d val_pairs=%d",
        device, len(train_ids), len(val_ids), len(train_ds), len(val_ds),
    )
    best_score, global_step = -float("inf"), 0
    checkpoint_path = Path(cfg["duplicate"]["checkpoint_path"])
    last_path = checkpoint_path.with_name("last.pth")
    series_batch_size = cfg["duplicate"].get("series_batch_size", 4)
    metadata = {
        "model_name": "Siamese-MedicalNet-ResNet10",
        "embedding_dim": cfg["duplicate"]["embedding_dim"],
        "target_shape": list(cfg["data"]["target_shape"]),
        "case_pooling": "mean_then_l2",
        "pretrained_from": pretrained_from or None,
    }

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        model.train()
        train_losses = []
        for step, (xa, xb, labels, a, b) in enumerate(train_loader, 1):
            global_step += 1
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = _batch_logits(model, xa, xb, device, series_batch_size)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            if step == 1 or step % cfg["training"]["log_every"] == 0:
                logger.info(
                    "[TRAIN] epoch=%d step=%d/%d loss=%.6f first_pair=(%s,%s)",
                    epoch, step, len(train_loader), loss.item(), a[0], b[0],
                )

        model.eval()
        val_losses, labels_all, probabilities = [], [], []
        with torch.no_grad():
            for xa, xb, labels, _, _ in val_loader:
                labels = labels.to(device)
                logits = _batch_logits(model, xa, xb, device, series_batch_size)
                val_losses.append(criterion(logits, labels).item())
                labels_all.extend(labels.cpu().tolist())
                probabilities.extend(logits.sigmoid().cpu().tolist())
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        metrics = duplicate_metrics(labels_all, probabilities)
        logger.info("[VAL] epoch=%d train_loss=%.6f val_loss=%.6f metrics=%s", epoch, train_loss, val_loss, metrics)
        extra = dict(metadata, train_loss=train_loss, val_loss=val_loss, metrics=metrics)
        save_checkpoint(model, optimizer, epoch, last_path, extra)
        score = metrics["auprc"] if np.isfinite(metrics["auprc"]) else -val_loss
        if score > best_score:
            best_score = score
            save_checkpoint(model, optimizer, epoch, checkpoint_path, extra)
            logger.info("Saved best checkpoint: %s", checkpoint_path)
        json_logger.write(
            epoch, global_step, "val", "training", "internal/validation_v2",
            loss=val_loss, lr=optimizer.param_groups[0]["lr"],
            checkpoint=checkpoint_path, pretrained_from=pretrained_from or None,
            **metrics,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    arguments = parser.parse_args()
    main(load_config(arguments.config))
