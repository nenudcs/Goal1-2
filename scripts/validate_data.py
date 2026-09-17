from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import pandas as pd

from src.common.config import load_config
from src.common.logging_utils import build_logger
from src.data.nifti import check_nifti
from src.data.paths import PathResolver, read_manifest
from src.tasks.abnormal.dataset import normalize_abnormal_labels
from src.tasks.duplicate.dataset import read_positive_pairs


def validate_abnormal(cfg, records):
    manifest = Path(cfg["paths"]["labels_dir"]) / cfg["abnormal"]["manifest"]
    if not manifest.is_file():
        records.append({"table": "1_abnormal", "type": "missing_manifest", "path": str(manifest)})
        return
    try:
        frame = normalize_abnormal_labels(
            read_manifest(manifest), cfg["abnormal"].get("label_aliases")
        )
    except Exception as error:
        records.append({"table": "1_abnormal", "type": "invalid_schema_or_label", "detail": str(error)})
        return
    resolver = PathResolver(
        cfg["paths"]["annotation_root"], cfg["data"].get("source_dirs")
    )
    for index, row in frame.iterrows():
        if not row["AccessionNumber"] or not row["SeriesUid"]:
            records.append({"table": "1_abnormal", "row": index + 2, "type": "empty_id"})
            continue
        path = resolver.series_path(
            row["AccessionNumber"], row["SeriesUid"], row["Label"]
        )
        if not check_nifti(path):
            records.append({
                "table": "1_abnormal", "row": index + 2,
                "AccessionNumber": row["AccessionNumber"],
                "SeriesUid": row["SeriesUid"],
                "type": "missing_or_broken_image", "path": str(path),
            })


def validate_duplicate(cfg, records):
    manifest = Path(cfg["paths"]["labels_dir"]) / cfg["duplicate"]["manifest"]
    if not manifest.is_file():
        records.append({"table": "2_duplicate", "type": "missing_manifest", "path": str(manifest)})
        return
    try:
        frame = read_manifest(manifest)
        required = {"src_img", "desc_img"}
        if missing := required - set(frame.columns):
            raise ValueError(f"missing columns: {sorted(missing)}")
        pairs = read_positive_pairs(manifest)
    except Exception as error:
        records.append({"table": "2_duplicate", "type": "invalid_schema", "detail": str(error)})
        return
    resolver = PathResolver(
        cfg["paths"]["annotation_root"], cfg["data"].get("source_dirs")
    )
    seen, ids = set(), set()
    for index, row in frame.iterrows():
        a, b = str(row["src_img"]).strip(), str(row["desc_img"]).strip()
        pair = tuple(sorted((a, b)))
        if not a or not b:
            records.append({"table": "2_duplicate", "row": index + 2, "type": "empty_id"})
        elif a == b:
            records.append({"table": "2_duplicate", "row": index + 2, "type": "self_pair", "id": a})
        elif pair in seen:
            records.append({"table": "2_duplicate", "row": index + 2, "type": "duplicate_pair", "detail": str(pair)})
        seen.add(pair)
        ids.update((a, b))
    available_ids = set(resolver.duplicate_accessions())
    if not available_ids:
        records.append({"table": "2_duplicate", "type": "missing_or_empty_duplicate_directory"})
    for identifier in sorted(ids | available_ids):
        case_dir = resolver.duplicate_accession_dir(identifier)
        if not resolver.list_case_series(case_dir):
            records.append({
                "table": "2_duplicate", "AccessionNumber": identifier,
                "type": "missing_or_broken_duplicate_case", "path": str(case_dir),
            })
    if not pairs:
        records.append({"table": "2_duplicate", "type": "no_valid_pairs"})


def main(cfg, tasks):
    logger = build_logger("validate_data")
    records = []
    tasks = set(tasks)
    if "goal1" in tasks or "goal2" in tasks:
        validate_abnormal(cfg, records)
    if "goal2" in tasks:
        validate_duplicate(cfg, records)
    report_dir = Path(cfg["paths"]["output_dir"]) / "data_validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "goal1and2_issues.csv"
    pd.DataFrame(records).to_csv(report, index=False, encoding="utf-8-sig")
    if records:
        logger.error("Validation found %d issue(s). Report: %s", len(records), report)
        return 1
    logger.info("Goal 1/2 validation passed. Report: %s", report)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tasks", nargs="+", choices=("goal1", "goal2"), default=("goal1", "goal2"))
    arguments = parser.parse_args()
    raise SystemExit(main(load_config(arguments.config), arguments.tasks))
