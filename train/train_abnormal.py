from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.common.logging_utils import CompetitionJSONLLogger, build_logger
from src.common.seed import seed_everything
from src.common.train_utils import get_device, load_checkpoint, save_checkpoint
from src.data.paths import read_manifest
from src.data.split import split_accessions
from src.models.medicalnet3d import MedicalNetClassifier, load_medicalnet_pretrained
from src.tasks.abnormal.dataset import AbnormalDataset, normalize_abnormal_labels


def _binary_metrics(labels, scores):
    if len(set(labels)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def case_metrics(series_probs, series_labels, accessions, aggregation):
    grouped_probs, grouped_labels = defaultdict(list), defaultdict(list)
    for prob, label, accession in zip(series_probs, series_labels, accessions):
        grouped_probs[str(accession)].append(prob)
        grouped_labels[str(accession)].append(int(label))
    probs, labels = [], []
    for accession in sorted(grouped_probs):
        if len(set(grouped_labels[accession])) != 1:
            raise ValueError(f"Case {accession} has conflicting three-class labels.")
        values = np.asarray(grouped_probs[accession])
        probs.append(values.mean(0) if aggregation == "mean" else values.max(0))
        labels.append(grouped_labels[accession][0])
    if not labels:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    probs = np.asarray(probs)
    predicted = probs.argmax(1)
    fake_auc, fake_ap = _binary_metrics([x == 1 for x in labels], probs[:, 1])
    composition_auc, composition_ap = _binary_metrics(
        [x == 2 for x in labels], probs[:, 2]
    )
    return {
        "accuracy": accuracy_score(labels, predicted),
        "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
        "fake_auroc": fake_auc,
        "fake_auprc": fake_ap,
        "composition_auroc": composition_auc,
        "composition_auprc": composition_ap,
    }


def _selection_score(metrics, val_loss):
    values = [metrics.get("fake_auroc"), metrics.get("composition_auroc")]
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else -float(val_loss)


def main(cfg):
    logger = build_logger("train_abnormal")
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)
    manifest = Path(cfg["paths"]["labels_dir"]) / cfg["abnormal"]["manifest"]
    frame = normalize_abnormal_labels(
        read_manifest(manifest), cfg["abnormal"].get("label_aliases")
    )
    conflicts = frame.groupby("AccessionNumber")["Label"].nunique()
    if (conflicts > 1).any():
        raise ValueError(
            f"Cases with conflicting three-class labels: {conflicts[conflicts > 1].index.tolist()[:10]}"
        )
    case_labels = frame.groupby("AccessionNumber")["Label"].first().to_dict()
    train_ids, val_ids = split_accessions(
        frame["AccessionNumber"].unique(), cfg["split"]["val_ratio"],
        cfg["split"]["random_state"], labels=case_labels,
    )
    dataset_args = (
        manifest, cfg["paths"]["annotation_root"], None,
        cfg["data"]["target_shape"], cfg["data"]["intensity_clip_percentiles"],
        cfg["abnormal"].get("label_aliases"), cfg["data"].get("source_dirs"),
    )
    train_ds = AbnormalDataset(*dataset_args[:2], train_ids, *dataset_args[3:])
    val_ds = AbnormalDataset(*dataset_args[:2], val_ids, *dataset_args[3:])
    if not train_ds or not val_ds:
        raise ValueError("Both abnormal train and validation datasets must be non-empty.")
    loader_options = {
        "num_workers": cfg["project"]["num_workers"],
        "pin_memory": cfg["project"].get("pin_memory", True) and device.type == "cuda",
    }
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        **loader_options,
    )

    model = MedicalNetClassifier(num_classes=3).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    start_epoch = 1
    pretrained_from = cfg["abnormal"].get("pretrained_path", "")
    resume = cfg["abnormal"].get("resume_path") or cfg["training"].get("resume", "")
    if resume:
        state = load_checkpoint(model, resume, device, optimizer, strict=True)
        start_epoch = int(state.get("epoch", 0)) + 1
    elif pretrained_from:
        matched, total = load_medicalnet_pretrained(model.encoder, pretrained_from, device)
        logger.info("Loaded MedicalNet encoder tensors: %d/%d", matched, total)

    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg["project"].get("amp", True) and device.type == "cuda"
    )
    json_logger = CompetitionJSONLLogger(
        cfg["paths"]["competition_log_dir"], "abnormal_train.jsonl",
        fallback_dir=str(Path(cfg["paths"]["output_dir"]) / "logs"),
    )
    logger.info(
        "Goal 1/stitched three-class training | model=MedicalNet ResNet-10 "
        "device=%s train_cases=%d val_cases=%d", device, len(train_ids), len(val_ids)
    )
    best_score, global_step = -float("inf"), 0
    checkpoint_path = Path(cfg["abnormal"]["checkpoint_path"])
    last_path = checkpoint_path.with_name("last.pth")
    metadata = {
        "model_name": "MedicalNet-ResNet10-3class",
        "class_names": list(cfg["abnormal"]["class_names"]),
        "target_shape": list(cfg["data"]["target_shape"]),
        "aggregation": cfg["data"].get("case_aggregation", "max"),
        "pretrained_from": pretrained_from or None,
    }

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        model.train()
        train_losses = []
        for step, (volumes, labels, accessions, _) in enumerate(train_loader, 1):
            global_step += 1
            volumes, labels = volumes.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(volumes)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            if step == 1 or step % cfg["training"]["log_every"] == 0:
                logger.info(
                    "[TRAIN] epoch=%d step=%d/%d loss=%.6f case=%s",
                    epoch, step, len(train_loader), loss.item(), accessions[0],
                )

        model.eval()
        val_losses, probabilities, labels_all, accessions_all = [], [], [], []
        with torch.no_grad():
            for volumes, labels, accessions, _ in val_loader:
                volumes, labels = volumes.to(device), labels.to(device)
                logits = model(volumes)
                val_losses.append(criterion(logits, labels).item())
                probabilities.extend(logits.softmax(1).cpu().numpy())
                labels_all.extend(labels.cpu().tolist())
                accessions_all.extend(accessions)
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        metrics = case_metrics(
            probabilities, labels_all, accessions_all,
            cfg["data"].get("case_aggregation", "max"),
        )
        train_loss = float(np.mean(train_losses))
        logger.info("[VAL] epoch=%d train_loss=%.6f val_loss=%.6f metrics=%s", epoch, train_loss, val_loss, metrics)
        extra = dict(metadata, train_loss=train_loss, val_loss=val_loss, metrics=metrics)
        save_checkpoint(model, optimizer, epoch, last_path, extra)
        score = _selection_score(metrics, val_loss)
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
