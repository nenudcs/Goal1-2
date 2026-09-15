from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from src.common.config import load_config
from src.common.seed import seed_everything
from src.common.logging_utils import build_logger, CompetitionJSONLLogger
from src.common.train_utils import get_device, save_checkpoint
from src.data.split import split_accessions
from src.models.basic3d import VolumeClassifier
from src.tasks.abnormal.dataset import AbnormalDataset

def main(cfg):
    if cfg.get("detection", {}).get("enabled"):
        from src.detection.training import train_abnormal
        return train_abnormal(cfg)
    logger = build_logger("train_abnormal")
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)

    xlsx = Path(cfg["paths"]["labels_dir"]) / "1_abnormal.xlsx"
    df = pd.read_excel(xlsx)
    df["AccessionNumber"] = df["AccessionNumber"].astype(str)

    case_labels = (df.groupby("AccessionNumber")["Label"].first().astype(str).str.lower().to_dict())
    train_ids, val_ids = split_accessions(
        df["AccessionNumber"].unique(),
        cfg["split"]["val_ratio"],
        cfg["split"]["random_state"],
        labels=case_labels
    )

    logger.info("========== Abnormal training ==========")
    logger.info("Device: %s", device)
    logger.info("Excel: %s", xlsx)
    logger.info("Train cases=%d | Val cases=%d", len(train_ids), len(val_ids))

    for name, count in df["Label"].astype(str).str.lower().value_counts().items():
        logger.info("Label count | %-12s = %d series", name, count)

    shape = cfg["data"]["target_shape"]
    clip = cfg["data"]["intensity_clip_percentiles"]
    train_ds = AbnormalDataset(xlsx, cfg["paths"]["annotation_root"], train_ids, shape, clip)
    val_ds = AbnormalDataset(xlsx, cfg["paths"]["annotation_root"], val_ids, shape, clip)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=cfg["project"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg["project"]["num_workers"])

    model = VolumeClassifier(num_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["project"]["amp"] and device.type == "cuda")

    jlog = CompetitionJSONLLogger(
        cfg["paths"]["competition_log_dir"], "abnormal_train.jsonl",
        fallback_dir=str(Path(cfg["paths"]["output_dir"]) / "logs")
    )

    best_acc = -1
    global_step = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        running = 0.0

        for step, (x, y, acc, suid) in enumerate(train_loader, 1):
            global_step += 1
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()

            if step % cfg["training"]["log_every"] == 0 or step == 1:
                logger.info(
                    "[TRAIN] epoch=%d/%d step=%d/%d global_step=%d "
                    "loss=%.6f lr=%.3e sample_accessionession=%s sample_series=%s",
                    epoch, cfg["training"]["epochs"], step, len(train_loader), global_step,
                    loss.item(), opt.param_groups[0]["lr"], acc[0], suid[0]
                )

                jlog.write(
                    epoch=epoch, step=global_step, phase="train", mode="training",
                    data_source="official/train_v2", loss=loss.item(),
                    lr=opt.param_groups[0]["lr"]
                )

        model.eval()
        preds, gts = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, y, *_ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                preds.extend(logits.argmax(1).cpu().tolist())
                gts.extend(y.cpu().tolist())

        val_acc = accuracy_score(gts, preds) if gts else 0.0
        val_loss /= max(len(val_loader), 1)

        logger.info(
            "[VAL] epoch=%d | train_loss=%.6f | val_loss=%.6f | val_acc=%.4f",
            epoch, running/max(len(train_loader),1), val_loss, val_acc
        )

        ckpt_dir = Path(cfg["paths"]["checkpoints_dir"]) / "abnormal"
        ckpt = ckpt_dir / f"epoch_{epoch}.pth"
        save_checkpoint(model, opt, epoch, ckpt, {"val_acc": val_acc})

        jlog.write(
            epoch=epoch, step=global_step, phase="val", mode="training",
            data_source="internal/validation_v2", loss=val_loss,
            lr=opt.param_groups[0]["lr"], checkpoint=str(ckpt)
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best = ckpt_dir / "best.pth"
            save_checkpoint(model, opt, epoch, best, {"val_acc": val_acc})
            logger.info("New best abnormal model: val_acc=%.4f -> %s", val_acc, best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
