from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from occupancy import (
    _read_json,
    _utc_now,
    _write_json,
    assign_vehicles_to_spaces,
    complete_space_labels,
    detect_vehicles,
    extract_features,
    load_meta,
    load_spaces,
    profile_dir,
    refresh_sample_counts,
    count_samples,
    seed_occupancy_map,
)

PENDING_TTL_HOURS = 48


def frames_dir(parking_id: str) -> Path:
    path = profile_dir(parking_id) / "frames"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_path(parking_id: str) -> Path:
    return profile_dir(parking_id) / "dataset.json"


def load_dataset(parking_id: str) -> dict:
    return _read_json(dataset_path(parking_id), {"frames": []})


def save_dataset(parking_id: str, data: dict) -> None:
    _write_json(dataset_path(parking_id), data)


def discard_frame_image(parking_id: str, record: dict | None) -> None:
    if not record:
        return
    name = record.get("file") or ""
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        return
    folder = frames_dir(parking_id).resolve()
    path = (folder / name).resolve()
    try:
        path.relative_to(folder)
    except ValueError:
        return
    if path.is_file():
        path.unlink()
        record["file_deleted"] = True


def _frame_age_hours(row: dict) -> float:
    text = str(row.get("created_at") or "").replace("Z", "+00:00")
    try:
        created = datetime.fromisoformat(text)
    except ValueError:
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds() / 3600.0)


def purge_expired_frames(parking_id: str) -> int:
    data = load_dataset(parking_id)
    kept = []
    removed = 0
    for row in data.get("frames", []):
        if row.get("reviewed"):
            discard_frame_image(parking_id, row)
            kept.append(row)
            continue
        if _frame_age_hours(row) > PENDING_TTL_HOURS:
            discard_frame_image(parking_id, row)
            removed += 1
            continue
        kept.append(row)
    if removed or any(row.get("file_deleted") for row in kept):
        data["frames"] = kept
        save_dataset(parking_id, data)
    return removed


_rel_cache: dict[str, tuple[float, dict]] = {}


def _ratio(hits: int, total: int) -> float | None:
    if not total:
        return None
    return round(hits / total, 4)


def evaluate_review_frame(parking_id: str, record: dict, space_ids: list[str] | None = None) -> dict | None:
    if not record.get("reviewed"):
        return None
    predicted_raw = record.get("yolo_spaces")
    if not predicted_raw:
        return None
    ids = space_ids or [str(row["id"]) for row in load_spaces(parking_id)]
    predicted = {sid: bool(predicted_raw.get(sid, False)) for sid in ids}
    actual_raw = record.get("spaces") or {}
    actual = {sid: bool(actual_raw.get(sid, False)) for sid in ids}
    source = str(record.get("source") or "")
    if source == "live_error":
        keys = [str(key) for key in (record.get("corrected") or []) if key in actual]
    else:
        keys = list(ids)
    if not keys:
        return None
    spaces: dict[str, dict] = {}
    hits = 0
    false_free = 0
    false_occupied = 0
    for space_id in keys:
        pred = bool(predicted.get(space_id, False))
        truth = bool(actual.get(space_id, False))
        ok = pred == truth
        if ok:
            hits += 1
        elif truth and not pred:
            false_free += 1
        else:
            false_occupied += 1
        spaces[space_id] = {"predicted": pred, "actual": truth, "ok": ok}
    total = len(keys)
    stall_n = len(ids)
    return {
        "frame_id": record.get("id"),
        "at": record.get("reviewed_at") or record.get("created_at"),
        "source": source,
        "n": total,
        "hits": hits,
        "misses": total - hits,
        "accuracy": _ratio(hits, total),
        "false_free": false_free,
        "false_occupied": false_occupied,
        "full": total >= max(2, stall_n // 2),
        "spaces": spaces,
    }


def reliability_report(parking_id: str) -> dict:
    path = dataset_path(parking_id)
    mtime = path.stat().st_mtime if path.exists() else 0
    cached = _rel_cache.get(parking_id)
    if cached and cached[0] == mtime:
        return cached[1]
    space_rows = load_spaces(parking_id)
    space_ids = [str(row["id"]) for row in space_rows]
    space_count = len(space_ids)
    evals = []
    for row in load_dataset(parking_id).get("frames", []):
        item = evaluate_review_frame(parking_id, row, space_ids)
        if item:
            evals.append(item)
    per_space: dict[str, dict] = {
        space_id: {
            "id": space_id,
            "n": 0,
            "hits": 0,
            "false_free": 0,
            "false_occupied": 0,
            "last_ok": None,
        }
        for space_id in space_ids
    }
    hits = 0
    checks = 0
    false_free = 0
    false_occupied = 0
    for item in evals:
        hits += int(item["hits"])
        checks += int(item["n"])
        false_free += int(item["false_free"])
        false_occupied += int(item["false_occupied"])
        for space_id, cell in (item.get("spaces") or {}).items():
            bucket = per_space.setdefault(
                space_id,
                {
                    "id": space_id,
                    "n": 0,
                    "hits": 0,
                    "false_free": 0,
                    "false_occupied": 0,
                    "last_ok": None,
                },
            )
            bucket["n"] += 1
            if cell.get("ok"):
                bucket["hits"] += 1
            elif cell.get("actual") and not cell.get("predicted"):
                bucket["false_free"] += 1
            else:
                bucket["false_occupied"] += 1
            bucket["last_ok"] = bool(cell.get("ok"))
    spaces = []
    for space_id in space_ids or sorted(per_space):
        bucket = per_space[space_id]
        n = int(bucket["n"])
        spaces.append(
            {
                **bucket,
                "accuracy": _ratio(int(bucket["hits"]), n),
            }
        )
    spaces.sort(key=lambda row: (row["accuracy"] is None, row["accuracy"] if row["accuracy"] is not None else 1, row["id"]))
    full = [item for item in evals if item.get("full")]
    recent = full[-8:]
    previous = full[-16:-8] if len(full) >= 12 else full[: max(0, len(full) - len(recent))]
    recent_hits = sum(int(item["hits"]) for item in recent)
    recent_n = sum(int(item["n"]) for item in recent)
    prev_hits = sum(int(item["hits"]) for item in previous)
    prev_n = sum(int(item["n"]) for item in previous)
    recent_acc = _ratio(recent_hits, recent_n)
    prev_acc = _ratio(prev_hits, prev_n)
    if recent_acc is None or prev_acc is None or recent_n < 8 or prev_n < 8:
        trend = "gutxi"
        delta = None
    else:
        delta = round(recent_acc - prev_acc, 4)
        if delta >= 0.04:
            trend = "hobetzen"
        elif delta <= -0.04:
            trend = "okertzen"
        else:
            trend = "egonkor"
    meta = load_meta(parking_id)
    report = {
        "parking_id": parking_id,
        "frames": len(evals),
        "full_frames": len(full),
        "checks": checks,
        "hits": hits,
        "misses": checks - hits,
        "accuracy": _ratio(hits, checks),
        "false_free": false_free,
        "false_occupied": false_occupied,
        "recent_accuracy": recent_acc,
        "recent_checks": recent_n,
        "previous_accuracy": prev_acc,
        "previous_checks": prev_n,
        "trend": trend,
        "trend_delta": delta,
        "model_accuracy": meta.get("accuracy"),
        "model_trained": bool(meta.get("approved_training") and meta.get("model_trained")),
        "spaces": spaces,
        "history": [
            {
                "at": item.get("at"),
                "frame_id": item.get("frame_id"),
                "accuracy": item.get("accuracy"),
                "hits": item.get("hits"),
                "n": item.get("n"),
            }
            for item in full[-30:]
        ],
    }
    _rel_cache[parking_id] = (mtime, report)
    return report


def reliability_summary(parking_id: str) -> dict:
    report = reliability_report(parking_id)
    return {
        "reliability_accuracy": report.get("accuracy"),
        "reliability_recent": report.get("recent_accuracy"),
        "reliability_trend": report.get("trend"),
        "reliability_checks": report.get("checks"),
        "reliability_frames": report.get("frames"),
    }


def dataset_stats(parking_id: str) -> dict:
    frames = load_dataset(parking_id).get("frames", [])
    reviewed = [row for row in frames if row.get("reviewed")]
    pending = [row for row in frames if not row.get("reviewed")]
    corrections = sum(len(row.get("corrected") or []) for row in reviewed)
    free, occupied = count_samples(parking_id)
    meta = load_meta(parking_id)
    return {
        "frames_total": len(frames),
        "frames_reviewed": len(reviewed),
        "frames_pending": len(pending),
        "corrections": corrections,
        "samples_free": free,
        "samples_occupied": occupied,
        "model_trained": bool(meta.get("approved_training") and meta.get("model_trained")),
        "accuracy": meta.get("accuracy"),
        **reliability_summary(parking_id),
    }


def too_similar(previous, current) -> bool:
    if previous is None or current is None:
        return False
    a = cv2.resize(previous, (64, 64)).astype(np.float32)
    b = cv2.resize(current, (64, 64)).astype(np.float32)
    return float(np.mean(np.abs(a - b))) < 5.5


def save_captured_frame(
    parking_id: str,
    frame: np.ndarray,
    yolo_spaces: dict[str, bool],
    source: str = "batch",
    reviewed: bool = False,
    spaces: dict | None = None,
    corrected: list | None = None,
) -> dict:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{stamp}.jpg"
    cv2.imwrite(str(frames_dir(parking_id) / filename), frame)
    record = {
        "id": stamp,
        "file": filename,
        "created_at": _utc_now(),
        "source": source,
        "reviewed": reviewed,
        "yolo_spaces": yolo_spaces,
        "spaces": spaces if spaces is not None else dict(yolo_spaces),
        "corrected": corrected or [],
    }
    data = load_dataset(parking_id)
    data.setdefault("frames", []).append(record)
    save_dataset(parking_id, data)
    return record


def get_frame_record(parking_id: str, frame_id: str) -> dict:
    for row in load_dataset(parking_id).get("frames", []):
        if row.get("id") == frame_id:
            return row
    raise ValueError("Irudia ez da aurkitu")


def load_frame_image(parking_id: str, frame_id: str):
    record = get_frame_record(parking_id, frame_id)
    path = frames_dir(parking_id) / record["file"]
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Irudia ezin da irakurri")
    return image, record


def pending_frames(parking_id: str) -> list[dict]:
    return [row for row in load_dataset(parking_id).get("frames", []) if not row.get("reviewed")]


def public_frame(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "created_at": row.get("created_at"),
        "source": row.get("source"),
        "reviewed": bool(row.get("reviewed")),
        "spaces": row.get("spaces") or {},
        "yolo_spaces": row.get("yolo_spaces") or {},
        "corrected": row.get("corrected") or [],
    }


def training_queue(parking_id: str) -> dict:
    frames = load_dataset(parking_id).get("frames", [])
    pending = [public_frame(row) for row in frames if not row.get("reviewed")]
    recent = [public_frame(row) for row in frames[-16:]]
    return {"pending": pending, "recent": list(reversed(recent)), "pending_count": len(pending)}


def review_frame(parking_id: str, frame_id: str, spaces: dict[str, bool]) -> tuple[dict, np.ndarray]:
    data = load_dataset(parking_id)
    record = None
    for row in data.get("frames", []):
        if row.get("id") == frame_id:
            record = row
            break
    if record is None:
        raise ValueError("Irudia ez da aurkitu")
    proposed = complete_space_labels(parking_id, record.get("yolo_spaces") or record.get("spaces") or {})
    cleaned = complete_space_labels(parking_id, spaces)
    corrected = [key for key, value in cleaned.items() if proposed.get(key) != value]
    record["spaces"] = cleaned
    record["corrected"] = corrected
    record["reviewed"] = True
    record["reviewed_at"] = _utc_now()
    save_dataset(parking_id, data)
    image, _ = load_frame_image(parking_id, frame_id)
    record["source"] = "review"
    _ingest_frame_labels(parking_id, image, record, weight=2.0 if corrected else 1.5)
    return record, image


def _ingest_frame_labels(
    parking_id: str,
    frame,
    record: dict,
    weight: float = 1.0,
    only_corrected: bool = False,
    seed_empty_free: bool = False,
) -> None:
    from occupancy import MIN_SAMPLES_PER_CLASS
    from occupancy import _read_json as read_labels
    from occupancy import _write_json as write_labels

    space_defs = {row["id"]: row for row in load_spaces(parking_id)}
    space_list = list(space_defs.values())
    vehicles = detect_vehicles(frame, spaces=space_list)
    _, per_space = assign_vehicles_to_spaces(space_list, vehicles, frame.shape)
    labels_path = profile_dir(parking_id) / "labels.json"
    labels = read_labels(labels_path, [])
    spaces_map = dict(record.get("spaces") or {})
    if only_corrected:
        spaces_map = {
            key: spaces_map[key]
            for key in (record.get("corrected") or [])
            if key in spaces_map
        }
    feature_map = {}

    def _append_label(space_id: str, occupied: bool, features, label_weight: float, source: str) -> None:
        feature_map[space_id] = features
        labels.append(
            {
                "space_id": space_id,
                "occupied": bool(occupied),
                "file": record.get("file"),
                "frame_id": record.get("id"),
                "features": features.tolist(),
                "weight": label_weight,
                "source": source,
                "approved": True,
                "created_at": record.get("created_at") or _utc_now(),
            }
        )

    for space_id, occupied in spaces_map.items():
        space = space_defs.get(space_id)
        if space is None:
            continue
        view = per_space.get(space_id) or {}
        features = extract_features(
            frame,
            space,
            vehicles,
            own_vehicles=view.get("own") or [],
            foreign_vehicles=view.get("foreign") or [],
            when=record.get("created_at"),
        )
        if features is None:
            continue
        _append_label(
            space_id,
            bool(occupied),
            features,
            3.0 if space_id in (record.get("corrected") or []) else weight,
            "review" if record.get("reviewed") else record.get("source", "batch"),
        )

    if seed_empty_free:
        free_now = sum(1 for row in labels if not row.get("occupied"))
        need = max(0, MIN_SAMPLES_PER_CLASS - free_now)
        candidates = []
        for space_id, space in space_defs.items():
            if space_id in feature_map:
                continue
            view = per_space.get(space_id) or {}
            if view.get("own"):
                continue
            features = extract_features(
                frame,
                space,
                vehicles,
                own_vehicles=view.get("own") or [],
                foreign_vehicles=view.get("foreign") or [],
                when=record.get("created_at"),
            )
            if features is None or len(features) < 34:
                continue
            edges = float(features[32])
            lap = float(features[33])
            if edges > 0.16:
                continue
            candidates.append((edges + 0.25 * lap, space_id, features))
        candidates.sort()
        for _, space_id, features in candidates[: max(need, 0)]:
            _append_label(space_id, False, features, 1.4, "auto_free")
            spaces_map[space_id] = False

    write_labels(labels_path, labels)
    seed_occupancy_map(parking_id, spaces_map, feature_map, frame=frame)
    discard_frame_image(parking_id, record)
    data = load_dataset(parking_id)
    for row in data.get("frames", []):
        if row.get("id") == record.get("id"):
            row["file_deleted"] = True
            break
    save_dataset(parking_id, data)
    return refresh_sample_counts(parking_id)


def save_live_correction(
    parking_id: str,
    frame,
    occupancy: dict[str, bool],
    space_id: str,
    occupied: bool,
    large_vehicle: bool = False,
    yolo_spaces: dict | None = None,
) -> dict:
    current = complete_space_labels(parking_id, yolo_spaces if yolo_spaces is not None else occupancy)
    current[space_id] = occupied
    if yolo_spaces is None:
        from occupancy import predict_occupancy

        yolo_spaces = predict_occupancy(parking_id, frame)
        current = complete_space_labels(parking_id, yolo_spaces)
        current[space_id] = occupied
    record = save_captured_frame(
        parking_id,
        frame,
        yolo_spaces=yolo_spaces,
        source="live_error",
        reviewed=True,
        spaces=current,
        corrected=[space_id],
    )
    record["large_vehicle"] = large_vehicle
    data = load_dataset(parking_id)
    for row in data.get("frames", []):
        if row.get("id") == record["id"]:
            row["large_vehicle"] = large_vehicle
            break
    save_dataset(parking_id, data)
    counts = _ingest_frame_labels(
        parking_id, frame, record, weight=1.6, only_corrected=False, seed_empty_free=True
    )
    record["samples_free"] = counts.get("samples_free", 0)
    record["samples_occupied"] = counts.get("samples_occupied", 0)
    return record
