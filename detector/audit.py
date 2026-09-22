from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "data" / "audit.jsonl"
_lock = threading.Lock()
_MAX_BYTES = 2_000_000


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(action: str, *, actor: str = "", parking_id: str = "", ip: str = "", ok: bool = True, detail: str = "") -> None:
    entry = {
        "ts": _utc(),
        "action": action,
        "ok": ok,
        "actor": actor or "-",
        "parking_id": parking_id or "-",
        "ip": ip or "-",
        "detail": (detail or "")[:200],
    }
    line = json.dumps(entry, ensure_ascii=False)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            if LOG_PATH.exists() and LOG_PATH.stat().st_size > _MAX_BYTES:
                rotated = LOG_PATH.with_suffix(".jsonl.old")
                rotated.unlink(missing_ok=True)
                LOG_PATH.replace(rotated)
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError:
        pass
