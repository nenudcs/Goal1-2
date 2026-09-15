from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import threading
from pathlib import Path
import requests
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from src.common.config import load_config
from src.common.train_utils import get_device
from src.common.logging_utils import build_logger

app = FastAPI()
logger = build_logger("competition_service")

CFG = None
MODELS = None
DEVICE = None

class CallPayload(BaseModel):
    request_id: str
    team_id: str | None = None
    track_code: str | None = None
    input: dict

def callback(request_id, evaluation_id, pred_path):
    url = CFG["service"].get("callback_url", "")
    if not url:
        logger.warning("callback_url is empty. Skip callback in local debug mode.")
        return

    payload = {
        "request_id": request_id,
        "evaluationId": evaluation_id,
        "predPath": str(pred_path),
    }

    try:
        r = requests.post(url, json=payload, timeout=30)
        logger.info("Callback status=%s body=%s", r.status_code, r.text[:300])
    except Exception:
        logger.exception("Callback failed")

def background_inference(payload):
    from src.pipeline.batch_pipeline import run_batch
    request_id = payload["request_id"]
    inp = payload["input"]
    evaluation_id = str(inp.get("evaluation_id") or inp.get("evaluationId"))
    dataset_path = inp["dataset_path"]

    answer_root = Path(CFG["paths"]["answer_root"])
    out = answer_root / evaluation_id
    out.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Start inference | request_id=%s evaluation_id=%s dataset=%s output=%s",
        request_id, evaluation_id, dataset_path, out
    )

    try:
        run_batch(dataset_path, out, MODELS, CFG, DEVICE)
        callback(request_id, evaluation_id, out)
        logger.info("Inference finished | evaluation_id=%s", evaluation_id)
    except Exception:
        logger.exception("Inference failed | evaluation_id=%s", evaluation_id)

@app.get("/health")
def health():
    return {"status": "active"}

@app.post("/call")
def call(payload: CallPayload, background_tasks: BackgroundTasks):
    # 立即返回，真正推理放后台，避免赛事 5 秒超时
    background_tasks.add_task(background_inference, payload.model_dump())
    return {"status": "received"}

def main(config_path):
    global CFG, MODELS, DEVICE
    CFG = load_config(config_path)
    if CFG.get("detection", {}).get("enabled"):
        from service.detection_server import main as detection_main
        return detection_main(config_path)
    from src.pipeline.load_models import load_all_models
    DEVICE = get_device(CFG)

    logger.info("Loading models on %s ...", DEVICE)
    MODELS = load_all_models(CFG, DEVICE)
    logger.info("Models loaded. Starting HTTP service on port %s", CFG["service"]["port"])

    uvicorn.run(app, host=CFG["service"]["host"], port=CFG["service"]["port"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
