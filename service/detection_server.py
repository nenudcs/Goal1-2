"""Durable single-worker FastAPI service. Run as a foreground process in the container."""
import argparse
import json
import logging
import os
import queue
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.common.config import load_config
from src.detection.common import atomic_json, read_json, require_container
from src.detection.submission import identifier, validate_submission

LOGGER = logging.getLogger("detection.service")


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evaluation_id: str
    dataset_path: str


class CallPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str
    team_id: str | None = None
    track_code: str | None = None
    input: InputData


class RetryPayload(BaseModel):
    request_id: str


class JobRunner:
    def __init__(self, cfg, inference, post=None):
        self.cfg, self.inference = cfg, inference
        self.post = post or requests.post
        self.directory = Path(cfg["paths"]["jobs_dir"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.lock = threading.RLock()
        self.pending = queue.Queue()
        self.stop = threading.Event()
        self.thread = None
        self.healthy = True
        settings = cfg["service"]
        if settings.get("callback_enabled"):
            if not settings.get("callback_url") or not settings.get("merge_from_template"):
                raise ValueError("Callbacks require callback_url and teammate merge_from_template; partial goals 1/2 cannot be announced as complete")
        if settings.get("callback_evaluation_id_type", "string") not in {"string", "number"}:
            raise ValueError("callback_evaluation_id_type must be string or number")
        if int(settings.get("callback_max_attempts", 5)) < 1:
            raise ValueError("callback_max_attempts must be positive")
        for path in sorted(self.directory.glob("*.json")):
            job = read_json(path)
            CallPayload.model_validate(job["payload"])
            request_id = identifier(job["payload"]["request_id"])
            evaluation_id = identifier(job["payload"]["input"]["evaluation_id"])
            if path.name != f"{request_id}.json":
                raise ValueError("Job identity mismatch on recovery")
            if Path(job["output"]).resolve() != (Path(cfg["paths"]["answer_root"]) / evaluation_id).resolve():
                raise ValueError("Job output path mismatch on recovery")
            self.jobs[request_id] = job
            if job["status"] in {"queued", "running", "callback_pending"}:
                job["status"] = "callback_pending" if job.get("phase") == "callback" else "queued"
                self._save(job)
                self.pending.put(request_id)

    def _save(self, job):
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            atomic_json(self.directory / f"{identifier(job['payload']['request_id'])}.json", job)
        except Exception:
            self.healthy = False
            raise

    def start(self):
        self.thread = threading.Thread(target=self._run, name="detection-worker", daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.pending.put(None)
        if self.thread:
            self.thread.join(timeout=5)

    def active(self):
        return self.healthy and not self.stop.is_set() and self.thread is not None and self.thread.is_alive()

    def submit(self, payload):
        request_id = identifier(payload["request_id"])
        evaluation_id = identifier(payload["input"]["evaluation_id"])
        dataset = Path(payload["input"]["dataset_path"])
        if not dataset.is_absolute() or not dataset.is_dir():
            raise ValueError("dataset_path must be an accessible absolute directory")
        if self.cfg["service"].get("callback_evaluation_id_type") == "number" and not evaluation_id.isdigit():
            raise ValueError("Numeric callback evaluationId requires digit-only evaluation_id")
        with self.lock:
            if request_id in self.jobs:
                if self.jobs[request_id]["payload"] != payload:
                    raise HTTPException(409, "request_id already refers to different input")
                return self.jobs[request_id]
            if any(j["payload"]["input"]["evaluation_id"] == evaluation_id for j in self.jobs.values()):
                raise HTTPException(409, "evaluation_id already belongs to another request")
            job = {"payload": payload, "status": "queued", "phase": "inference", "callback_attempts": 0,
                   "output": str(Path(self.cfg["paths"]["answer_root"]) / evaluation_id), "error": None}
            self._save(job)
            self.jobs[request_id] = job
            self.pending.put(request_id)
            return job

    def retry(self, request_id):
        with self.lock:
            if request_id not in self.jobs:
                raise HTTPException(404, "Unknown request")
            job = self.jobs[request_id]
            if job["status"] != "failed":
                raise HTTPException(409, "Only failed jobs may be retried")
            job["status"] = "callback_pending" if job["phase"] == "callback" else "queued"
            job["error"], job["callback_attempts"] = None, 0
            self._save(job)
            self.pending.put(request_id)
            return job

    def snapshot(self, request_id):
        with self.lock:
            if request_id not in self.jobs:
                raise HTTPException(404, "Unknown request")
            return json.loads(json.dumps(self.jobs[request_id]))

    def _callback(self, job):
        settings = self.cfg["service"]
        if not settings.get("callback_enabled"):
            job["status"] = "awaiting_merge"
            self._save(job)
            return
        if not job.get("report", {}).get("merged_other_goals"):
            raise ValueError("Recovered output has not been merged with teammate goals; callback is blocked")
        validate_submission(job["output"], job["report"]["study_ids"])
        evaluation_id = job["payload"]["input"]["evaluation_id"]
        template = settings.get("callback_pred_path_template")
        body = {"request_id": job["payload"]["request_id"],
                "evaluationId": int(evaluation_id) if settings.get("callback_evaluation_id_type") == "number" else evaluation_id,
                "predPath": template.format(evaluation_id=evaluation_id) if template else job["output"]}
        maximum = int(settings.get("callback_max_attempts", 5))
        while job["callback_attempts"] < maximum and not self.stop.is_set():
            job["callback_attempts"] += 1
            self._save(job)
            try:
                response = self.post(settings["callback_url"], json=body, timeout=30)
                if not 200 <= response.status_code < 300:
                    raise RuntimeError(f"Callback HTTP {response.status_code}")
                job["status"], job["error"] = "completed", None
                self._save(job)
                return
            except Exception as exc:
                job["error"] = f"{type(exc).__name__}: {exc}"
                self._save(job)
                if job["callback_attempts"] < maximum:
                    self.stop.wait(min(60, float(settings.get("callback_retry_seconds", 5))*2**(job["callback_attempts"]-1)))
        if not self.stop.is_set():
            job["status"] = "failed"
            self._save(job)

    def _run(self):
        while not self.stop.is_set():
            request_id = self.pending.get()
            if request_id is None:
                return
            job = self.jobs[request_id]
            try:
                if job["phase"] == "inference":
                    job["status"] = "running"
                    self._save(job)
                    template = self.cfg["service"].get("merge_from_template")
                    merge = template.format(evaluation_id=job["payload"]["input"]["evaluation_id"]) if template else None
                    report = self.inference(job["payload"]["input"]["dataset_path"], job["output"], merge)
                    if not report.get("validated"):
                        raise ValueError("Inference did not validate output completeness")
                    job.update(report=report, phase="callback", status="callback_pending")
                    self._save(job)
                self._callback(job)
            except Exception as exc:
                LOGGER.exception("Job failed: %s", request_id)
                job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                self._save(job)


def create_app(cfg=None, model_loader=None, infer=None):
    @asynccontextmanager
    async def lifespan(app):
        require_container()
        import fcntl
        from src.detection.pipeline import load_detection_models, run_detection_batch
        settings = cfg or load_config(os.environ.get("DETECTION_CONFIG", "configs/config_detection.yaml"))
        if int(settings["service"]["port"]) != 8000:
            raise ValueError("Competition service must use port 8000")
        lock_path = Path(settings["paths"]["jobs_dir"]) / ".worker.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            models = (model_loader or load_detection_models)(settings)
            def inference(dataset, output, merge):
                return (infer or run_detection_batch)(dataset, output, models, settings, merge_from=merge)
            runner = JobRunner(settings, inference)
            app.state.runner = runner
            runner.start()
            try:
                yield
            finally:
                runner.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        runner = getattr(app.state, "runner", None)
        if runner is None or not runner.active():
            raise HTTPException(503, "Model/worker not ready")
        return {"status": "active"}

    @app.post("/call")
    def call(payload: CallPayload):
        health()
        try:
            job = app.state.runner.submit(payload.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "received", "job_status": job["status"]}

    @app.get("/jobs/{request_id}")
    def job_status(request_id: str):
        return app.state.runner.snapshot(request_id)

    @app.post("/retry")
    def retry(payload: RetryPayload):
        health()
        app.state.runner.retry(payload.request_id)
        return {"status": "received"}

    return app


app = create_app()


def main(config_path):
    import uvicorn
    cfg = load_config(config_path)
    uvicorn.run(create_app(cfg), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_detection.yaml")
    main(parser.parse_args().config)
