from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from .schema import DEFAULT_MODEL_ID, JobRecord, ModelConfigInput, ModelConfigView, utc_now


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "runtime-data"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
JOB_DIR = RUNTIME_DIR / "jobs"
ANALYSIS_DIR = RUNTIME_DIR / "analyses"
CONFIG_PATH = RUNTIME_DIR / "config.json"
LEGACY_DEFAULT_MODELS = {"qwen3.5-omni-plus"}


def ensure_runtime_dirs() -> None:
    for path in (UPLOAD_DIR, JOB_DIR, ANALYSIS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def safe_filename(name: str) -> str:
    clean = Path(name or "video.mp4").name
    clean = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", clean, flags=re.UNICODE)
    return clean[:120] or "video.mp4"


class ConfigStore:
    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self.path = path
        self._lock = threading.RLock()
        persisted = read_json(path, {}) or {}
        self.base_url = persisted.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model_id = persisted.get("model_id", DEFAULT_MODEL_ID)
        if self.model_id in LEGACY_DEFAULT_MODELS:
            self.model_id = DEFAULT_MODEL_ID
            write_json(self.path, {"base_url": self.base_url, "model_id": self.model_id})
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

    def update(self, config: ModelConfigInput) -> ModelConfigView:
        with self._lock:
            self.base_url = config.base_url
            self.model_id = config.model_id.strip() or DEFAULT_MODEL_ID
            if config.api_key and config.api_key.strip():
                self.api_key = config.api_key.strip()
            write_json(self.path, {"base_url": self.base_url, "model_id": self.model_id})
            return self.view()

    def view(self) -> ModelConfigView:
        with self._lock:
            key = self.api_key
            return ModelConfigView(
                base_url=self.base_url,
                model_id=self.model_id,
                configured=bool(key) or os.getenv("QWEN_MOCK_MODE") == "1",
                masked_key=("••••••" + key[-4:]) if key else "未配置",
                mock_mode=os.getenv("QWEN_MOCK_MODE") == "1",
            )


class JobStore:
    def __init__(self, job_dir: Path = JOB_DIR) -> None:
        self.job_dir = job_dir
        self._lock = threading.RLock()
        ensure_runtime_dirs()

    def save(self, job: JobRecord) -> JobRecord:
        with self._lock:
            job.updated_at = utc_now()
            write_json(self.job_dir / f"{job.id}.json", job.model_dump())
            return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            payload = read_json(self.job_dir / f"{job_id}.json")
            return JobRecord.model_validate(payload) if payload else None

    def update(self, job_id: str, **changes: Any) -> JobRecord:
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        for key, value in changes.items():
            setattr(job, key, value)
        return self.save(job)
