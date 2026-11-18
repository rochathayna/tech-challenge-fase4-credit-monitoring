"""Logs estruturados (JSON Lines) centralizados em logs/pipeline.jsonl.

Cada evento tem: timestamp, run_id, stage, event, status, duração e campos extras.
O dashboard e o MLflow leem esse mesmo arquivo — é a fonte única de verdade operacional.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import LOGS_DIR

LOG_FILE = LOGS_DIR / "pipeline.jsonl"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]

_console = logging.getLogger("credit_monitoring")
if not _console.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    _console.addHandler(h)
    _console.setLevel(logging.INFO)


def log_event(stage: str, event: str, status: str = "info", **fields) -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "stage": stage,
        "event": event,
        "status": status,
        **fields,
    }
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    level = {"error": logging.ERROR, "alert": logging.WARNING, "warning": logging.WARNING}.get(status, logging.INFO)
    extras = " ".join(f"{k}={v}" for k, v in fields.items() if not isinstance(v, (list, dict)))
    _console.log(level, f"[{stage}] {event} ({status}) {extras}")
    return record


@contextmanager
def stage_timer(stage: str, **fields):
    """Mede latência da etapa e registra sucesso/falha."""
    start = time.perf_counter()
    log_event(stage, "stage_started", "info", **fields)
    try:
        yield
    except Exception as exc:  # registra e repropaga
        log_event(stage, "stage_failed", "error", duration_s=round(time.perf_counter() - start, 3),
                  error=repr(exc), **fields)
        raise
    log_event(stage, "stage_succeeded", "success", duration_s=round(time.perf_counter() - start, 3), **fields)


def read_logs() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    return [json.loads(line) for line in LOG_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
