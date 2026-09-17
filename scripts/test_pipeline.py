from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.common.config import load_config
from src.common.logging_utils import build_logger
from src.common.train_utils import get_device
from src.pipeline.goal1and2_pipeline import run_goal_pipeline


def main(task, cfg, input_path, output_path):
    logger = build_logger("test_pipeline")
    device = get_device(cfg)
    logger.info("Running %s inference on %s", task, device)
    cases = run_goal_pipeline(task, input_path, output_path, cfg, device)
    logger.info("Finished %s: %d cases -> %s", task, len(cases), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("goal1", "goal2"))
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    main(
        arguments.task, load_config(arguments.config),
        arguments.input, arguments.output,
    )
