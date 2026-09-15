from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from src.common.config import load_config
from src.common.seed import seed_everything
from src.common.logging_utils import build_logger, CompetitionJSONLLogger
from src.common.train_utils import get_device, save_checkpoint
from src.data.split import split_accessions
from src.models.basic3d import VolumeClassifier
from src.tasks.sequence.dataset import SequenceDataset

def main(cfg):
    logger = build_logger("train_sequence")
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)

    xlsx = Path(cfg["paths"]["labels_dir"]) / "3_serieslabel.xlsx"
    df = pd.read_excel(xlsx)
    df["AccessionNumber"] = df["AccessionNumber"].astype(str)

    from src.data.split import get_or_create_true_case_split
    train_ids, val_ids = get_or_create_true_case_split(
        Path(cfg["paths"]["labels_dir"])/"1_abnormal.xlsx",
        Path(cfg["paths"]["output_dir"])/"splits"/"true_case_split_0.1.json",
        cfg["split"]["val_ratio"], cfg["split"]["random_state"]
    )

    logger.info("========== Sequence training ==========")
    logger.info("Train cases=%d | Val cases=%d | Device=%s", len(train_ids), len(val_ids), device)
    logger.info("SeriesLabel counts:\n%s", df["SeriesLabel"].value_counts(dropna=False).to_string())

    shape = cfg["data"]["target_shape"]
    clip = cfg["data"]["intensity_clip_percentiles"]
    train_ds = SequenceDataset(xlsx, cfg["paths"]["annotation_root"], train_ids, shape, clip)
    val_ds = SequenceDataset(xlsx, cfg["paths"]["annotation_root"], val_ids, shape, clip)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
                              num_workers=cfg["project"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg["project"]["num_workers"])

    model = VolumeClassifier(num_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()

    jlog = CompetitionJSONLLogger(
        cfg["paths"]["competition_log_dir"], "sequence_train.jsonl",
        fallback_dir=str(Path(cfg["paths"]["output_dir"]) / "logs")
    )

    best = -1
    global_step = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        for step, (x, y, acc, suid) in enumerate(train_loader, 1):
            global_step += 1
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            if step % cfg["training"]["log_every"] == 0 or step == 1:
                logger.info(
                    "[TRAIN] epoch=%d step=%d/%d global=%d loss=%.6f case=%s series=%s",
                    epoch, step, len(train_loader), global_step, loss.item(), acc[0], suid[0]
                )
                jlog.write(epoch, global_step, "train", "training",
                           "official/train_v2", loss.item(), opt.param_groups[0]["lr"])

        model.eval()
        pred, gt = [], []
        losses = []
        with torch.no_grad():
            for x, y, *_ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                losses.append(criterion(logits, y).item())
                pred.extend(logits.argmax(1).cpu().tolist())
                gt.extend(y.cpu().tolist())

        acc = accuracy_score(gt, pred) if gt else 0.0
        logger.info("[VAL] epoch=%d val_acc=%.4f val_loss=%.6f", epoch, acc,
                    sum(losses)/max(len(losses),1))
        if gt:
            logger.info("Confusion matrix:\n%s", confusion_matrix(gt, pred))

        ckpt_dir = Path(cfg["paths"]["checkpoints_dir"]) / "sequence"
        ckpt = ckpt_dir / f"epoch_{epoch}.pth"
        save_checkpoint(model, opt, epoch, ckpt, {"val_acc": acc})
        jlog.write(epoch, global_step, "val", "training", "internal/validation_v2",
                   sum(losses)/max(len(losses),1), opt.param_groups[0]["lr"], ckpt)

        if acc > best:
            best = acc
            save_checkpoint(model, opt, epoch, ckpt_dir / "best.pth", {"val_acc": acc})
            logger.info("New best sequence model: %.4f", acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
