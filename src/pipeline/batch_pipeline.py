from pathlib import Path
from src.pipeline.case_pipeline import run_case
from src.pipeline.duplicate_pipeline import run_duplicate_pipeline

def run_batch(dataset_path, output_root, models, cfg, device):
    dataset_path = Path(dataset_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    case_dirs = [p for p in dataset_path.iterdir() if p.is_dir()]
    for cdir in sorted(case_dirs):
        run_case(cdir, output_root / cdir.name, models, cfg, device)

    run_duplicate_pipeline(
        dataset_path,
        output_root / "duplicate_pairs.jsonl",
        models["duplicate"],
        cfg, device
    )
