from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.common.config import load_config
from src.common.train_utils import get_device
from src.common.logging_utils import build_logger
from src.pipeline.load_models import load_all_models
from src.pipeline.batch_pipeline import run_batch

def main(cfg, input_path, output_path):
    logger = build_logger("test_pipeline")
    device = get_device(cfg)
    logger.info("Loading all models on %s ...", device)
    models = load_all_models(cfg, device)
    logger.info("Running batch pipeline | input=%s | output=%s", input_path, output_path)
    run_batch(input_path, output_path, models, cfg, device)
    logger.info("Pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.input, args.output)
