from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
SECRETS_PATH = DATA_ROOT / ".camera_secrets.json"


def _read() -> dict:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not SECRETS_PATH.exists():
        return {}
    try:
        payload = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write(payload: dict) -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    SECRETS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def camera_login(parking_id: str) -> tuple[str, str]:
    row = _read().get(parking_id) or {}
    return str(row.get("camera_user") or ""), str(row.get("camera_pass") or "")


def set_camera_auth(parking_id: str, user: str | None, password: str | None) -> None:
    data = _read()
    row = dict(data.get(parking_id) or {})
    if user is not None:
        row["camera_user"] = user
    if password is not None:
        row["camera_pass"] = password
    if row.get("camera_user") or row.get("camera_pass"):
        data[parking_id] = row
    else:
        data.pop(parking_id, None)
    _write(data)


def migrate_parking_config(parking_id: str) -> None:
    path = DATA_ROOT / parking_id / "config.json"
    if not path.is_file():
        return
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(cfg, dict):
        return
    if "camera_user" not in cfg and "camera_pass" not in cfg:
        return
    user = str(cfg.pop("camera_user", "") or "")
    password = str(cfg.pop("camera_pass", "") or "")
    if user or password:
        stored_user, stored_pass = camera_login(parking_id)
        set_camera_auth(parking_id, user or stored_user, password or stored_pass)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
