from __future__ import annotations

import random
import time

from dataset import pending_frames, save_captured_frame, too_similar
from occupancy import analyze_frame, load_spaces

BUFFER_SIZE = 10
MIN_WAIT = 18
MAX_WAIT = 75

_state: dict[str, dict] = {}


def _entry(parking_id: str) -> dict:
    return _state.setdefault(
        parking_id,
        {
            "ids": [],
            "images": [],
            "next_at": time.time() + random.uniform(8, 20),
        },
    )


def buffer_status(parking_id: str) -> dict:
    entry = _entry(parking_id)
    wait = max(0, int(entry["next_at"] - time.time()))
    return {
        "buffer_size": len(entry["ids"]),
        "buffer_target": BUFFER_SIZE,
        "buffer_full": len(entry["ids"]) >= BUFFER_SIZE,
        "next_sample_in": wait,
        "sampling": bool(load_spaces(parking_id)) and len(entry["ids"]) < BUFFER_SIZE,
    }


def maybe_sample(parking_id: str, frame, occupancy: dict | None = None) -> dict | None:
    if frame is None or not load_spaces(parking_id):
        return None
    entry = _entry(parking_id)
    if len(entry["ids"]) >= BUFFER_SIZE:
        return None
    if time.time() < entry["next_at"]:
        return None
    for previous in entry["images"]:
        if too_similar(previous, frame):
            entry["next_at"] = time.time() + random.uniform(MIN_WAIT, MAX_WAIT)
            return None
    spaces = occupancy
    if spaces is None:
        status, _ = analyze_frame(parking_id, frame)
        spaces = status.get("spaces") or {}
    record = save_captured_frame(
        parking_id,
        frame,
        yolo_spaces=spaces,
        source="auto_buffer",
    )
    entry["ids"].append(record["id"])
    entry["images"].append(frame.copy())
    entry["next_at"] = time.time() + random.uniform(MIN_WAIT, MAX_WAIT)
    return record


def consume_buffer(parking_id: str) -> list[str]:
    entry = _entry(parking_id)
    used = list(entry["ids"])
    entry["ids"] = []
    entry["images"] = []
    entry["next_at"] = time.time() + random.uniform(8, 25)
    return used


def pending_buffer_frames(parking_id: str) -> list[dict]:
    ids = set(_entry(parking_id)["ids"])
    return [row for row in pending_frames(parking_id) if row.get("id") in ids or row.get("source") == "auto_buffer"]
