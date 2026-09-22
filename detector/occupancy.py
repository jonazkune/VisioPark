from __future__ import annotations

import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DATA_ROOT = Path(__file__).resolve().parent / "data"
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
LARGE_CLASS_IDS = {5, 7}
MIN_SAMPLES_PER_CLASS = 4
STICKY_TTL_SECONDS = 3600.0
FEATURE_CORE = 38
FEATURE_DIM = 42
APPEARANCE_DIM = 34
CONFIRM_TO_OCCUPIED = 3
CONFIRM_TO_FREE = 3
LIGHT_ORDER = ("night", "dim", "day", "bright")
_model_cache: dict[str, tuple[float, object]] = {}
_proto_cache: dict[str, tuple[float, dict]] = {}
_sticky: dict[str, dict[str, dict]] = {}
_history: dict[str, dict[str, dict]] = {}
_history_loaded: set[str] = set()

try:
    from ultralytics import YOLO

    _YOLO = None
    HAS_YOLO = True
    _YOLO_ERROR = None
except Exception as exc:
    HAS_YOLO = False
    _YOLO = None
    _YOLO_ERROR = str(exc)

YOLO_WEIGHTS = Path(__file__).resolve().parent / "yolov8n.pt"


def _local_hour(when=None) -> float:
    if when is None:
        dt = datetime.now().astimezone()
    elif isinstance(when, datetime):
        dt = when.astimezone() if when.tzinfo else when.replace(tzinfo=timezone.utc).astimezone()
    else:
        text = str(when).replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = datetime.now().astimezone()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone()
    return dt.hour + dt.minute / 60.0


def _time_of_day_features(when=None) -> list[float]:
    hour = _local_hour(when)
    angle = 2.0 * np.pi * (hour / 24.0)
    return [float(np.sin(angle)), float(np.cos(angle))]


def _scene_brightness(frame: np.ndarray) -> float:
    if frame is None or frame.size == 0:
        return 0.5
    small = cv2.resize(frame, (64, 64))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return float(gray.mean() / 255.0)


def _day_period(when=None) -> str:
    hour = _local_hour(when)
    if hour < 6.5 or hour >= 21.0:
        return "night"
    if hour < 11.0:
        return "morning"
    if hour < 16.5:
        return "midday"
    return "evening"


def lighting_bin(scene_brightness: float, when=None) -> str:
    """Few lighting buckets: clock hour plus how bright the photo actually is."""
    period = _day_period(when)
    value = float(np.clip(scene_brightness, 0.0, 1.0))
    if value < 0.18:
        light = "night"
    elif value < 0.32:
        light = "dim"
    elif value < 0.55:
        light = "day"
    else:
        light = "bright"
    return f"{period}_{light}"


def _light_from_bin(bin_id: str) -> str:
    if "_" in (bin_id or ""):
        return str(bin_id).rsplit("_", 1)[-1]
    return str(bin_id or "day")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _align_features(row: dict) -> list[float] | None:
    raw = row.get("features")
    if not isinstance(raw, list) or not raw:
        return None
    feats = [float(v) for v in raw]
    if len(feats) == FEATURE_DIM:
        return feats
    if len(feats) == FEATURE_CORE:
        extra = _time_of_day_features(row.get("created_at"))
        extra.extend([0.5, 0.5])
        return feats + extra
    return None


def profile_dir(parking_id: str) -> Path:
    safe = "".join(ch for ch in parking_id if ch.isalnum() or ch in "-_") or "parking"
    path = DATA_ROOT / safe
    path.mkdir(parents=True, exist_ok=True)
    (path / "samples").mkdir(exist_ok=True)
    return path


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_config(parking_id: str) -> dict:
    return _read_json(profile_dir(parking_id) / "config.json", {})


def save_config(parking_id: str, config: dict) -> None:
    current = load_config(parking_id)
    incoming = dict(config)
    user = incoming.pop("camera_user", None)
    password = incoming.pop("camera_pass", None)
    current.update(incoming)
    current.pop("camera_user", None)
    current.pop("camera_pass", None)
    if user is not None or password is not None:
        from credentials import set_camera_auth

        set_camera_auth(parking_id, user, password)
    _write_json(profile_dir(parking_id) / "config.json", current)


def ensure_display_token(parking_id: str) -> str:
    token = str(load_config(parking_id).get("display_token") or "")
    if len(token) < 16:
        token = secrets.token_urlsafe(24)
        save_config(parking_id, {"display_token": token})
    return token


def rotate_display_token(parking_id: str) -> str:
    token = secrets.token_urlsafe(24)
    save_config(parking_id, {"display_token": token})
    return token


def parking_id_for_display_token(token: str) -> str | None:
    raw = (token or "").strip()
    if len(raw) < 16:
        return None
    if not DATA_ROOT.exists():
        return None
    for path in DATA_ROOT.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        stored = str(load_config(path.name).get("display_token") or "")
        if stored and len(stored) == len(raw) and secrets.compare_digest(stored, raw):
            return path.name
    return None


def load_spaces(parking_id: str) -> list[dict]:
    data = _read_json(profile_dir(parking_id) / "spaces.json", {"spaces": []})
    return data.get("spaces", [])


def save_spaces(parking_id: str, spaces: list[dict]) -> None:
    _write_json(profile_dir(parking_id) / "spaces.json", {"spaces": spaces})


def load_meta(parking_id: str) -> dict:
    return _read_json(
        profile_dir(parking_id) / "meta.json",
        {
            "model_trained": False,
            "samples_free": 0,
            "samples_occupied": 0,
            "accuracy": None,
        },
    )


def _save_meta(parking_id: str, meta: dict) -> None:
    _write_json(profile_dir(parking_id) / "meta.json", meta)


def _is_approved_label(row: dict) -> bool:
    if row.get("approved") is True:
        return True
    return row.get("source") in {"live_error", "review"}


def count_samples(parking_id: str) -> tuple[int, int]:
    labels = [row for row in _read_json(profile_dir(parking_id) / "labels.json", []) if _is_approved_label(row)]
    free = sum(1 for row in labels if not row.get("occupied"))
    occupied = sum(1 for row in labels if row.get("occupied"))
    return free, occupied


def refresh_sample_counts(parking_id: str) -> dict:
    free, occupied = count_samples(parking_id)
    meta = load_meta(parking_id)
    meta["samples_free"] = free
    meta["samples_occupied"] = occupied
    _save_meta(parking_id, meta)
    return meta


def yolo_model():
    global _YOLO
    if not HAS_YOLO:
        return None
    if _YOLO is None:
        weights = str(YOLO_WEIGHTS) if YOLO_WEIGHTS.exists() else "yolov8n.pt"
        _YOLO = YOLO(weights)
    return _YOLO


def warmup_yolo() -> str:
    if not HAS_YOLO:
        return f"YOLO falta: {_YOLO_ERROR or 'ultralytics ez dago instalatuta'}"
    try:
        yolo_model()
        return "YOLO prest"
    except Exception as exc:
        return f"YOLO ezin da kargatu: {exc}"


def _stalls_roi(frame: np.ndarray, spaces: list[dict] | None) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    if not spaces:
        return 0, 0, width, height
    xs: list[int] = []
    ys: list[int] = []
    stall_heights: list[int] = []
    for space in spaces:
        pts = space_points(space, width, height)
        if pts is None or len(pts) == 0:
            continue
        xs.extend(int(x) for x in pts[:, 0])
        ys.extend(int(y) for y in pts[:, 1])
        _, _, _, bh = cv2.boundingRect(pts)
        stall_heights.append(max(1, int(bh)))
    if not xs:
        return 0, 0, width, height
    span_x = max(1, max(xs) - min(xs))
    span_y = max(1, max(ys) - min(ys))
    typical_h = int(np.median(stall_heights)) if stall_heights else 24
    pad_x = max(32, int(0.14 * span_x))
    # Stall polygons are ground footprints; car bodies sit well above them in the photo.
    pad_top = max(int(0.16 * height), int(4.5 * typical_h), int(0.65 * span_y))
    pad_bot = max(20, int(0.10 * span_y))
    x1 = max(0, min(xs) - pad_x)
    y1 = max(0, min(ys) - pad_top)
    x2 = min(width, max(xs) + pad_x)
    y2 = min(height, max(ys) + pad_bot)
    if x2 - x1 < 80 or y2 - y1 < 80:
        return 0, 0, width, height
    return x1, y1, x2, y2


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _nms_vehicles(boxes: list[dict], iou_thr: float = 0.55) -> list[dict]:
    ordered = sorted(boxes, key=lambda row: float(row.get("conf") or 0.0), reverse=True)
    kept: list[dict] = []
    for row in ordered:
        bbox = vehicle_bbox(row)
        if any(_box_iou(bbox, vehicle_bbox(other)) >= iou_thr for other in kept):
            continue
        kept.append(row)
    return kept


def _yolo_boxes(
    model,
    image: np.ndarray,
    ox: int,
    oy: int,
    frame_shape: tuple[int, ...],
    conf: float,
    imgsz: int,
    max_det: int = 200,
) -> list[dict]:
    if image is None or image.size == 0:
        return []
    ih, iw = image.shape[:2]
    if ih < 16 or iw < 16:
        return []
    brightness = _scene_brightness(image)
    if brightness < 0.16 and image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        light, a, b = cv2.split(lab)
        light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
        image = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)
    use_conf = min(conf, 0.05) if brightness < 0.22 else conf
    result = model.predict(
        image,
        verbose=False,
        imgsz=imgsz,
        conf=use_conf,
        iou=0.45,
        classes=sorted(VEHICLE_CLASS_IDS),
        max_det=max_det,
    )[0]
    boxes: list[dict] = []
    if result.boxes is None:
        return boxes
    for box in result.boxes:
        cls = int(box.cls[0])
        if cls not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        bbox = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
        score = float(box.conf[0]) if box.conf is not None else 0.0
        boxes.append(
            {
                "bbox": bbox,
                "cls": cls,
                "conf": score,
                "large": is_large_vehicle(cls, bbox, frame_shape),
            }
        )
    return boxes


def _stall_view(frame: np.ndarray, space: dict) -> tuple[np.ndarray, int, int] | None:
    height, width = frame.shape[:2]
    pts = space_points(space, width, height)
    if pts is None or len(pts) == 0:
        return None
    x, y, bw, bh = cv2.boundingRect(pts)
    if bw < 4 or bh < 4:
        return None
    pad_x = max(12, int(0.40 * bw))
    pad_top = max(28, int(max(3.8 * bh, 0.95 * bw)))
    pad_bot = max(8, int(0.30 * bh))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_top)
    x2 = min(width, x + bw + pad_x)
    y2 = min(height, y + bh + pad_bot)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop, x1, y1


def _letterbox(crop: np.ndarray, cell_w: int, cell_h: int) -> tuple[np.ndarray, float, int, int]:
    ch, cw = crop.shape[:2]
    scale = min(cell_w / float(max(1, cw)), cell_h / float(max(1, ch)))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    ox = (cell_w - nw) // 2
    oy = (cell_h - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas, scale, ox, oy


def _detect_on_stall_mosaic(model, frame: np.ndarray, spaces: list[dict]) -> list[dict]:
    views: list[tuple[dict, np.ndarray, int, int]] = []
    for space in spaces:
        view = _stall_view(frame, space)
        if view is None:
            continue
        crop, ox, oy = view
        views.append((space, crop, ox, oy))
    if not views:
        return []
    cell_w, cell_h = 160, 192
    cols = max(1, int(np.ceil(np.sqrt(len(views)))))
    rows = max(1, int(np.ceil(len(views) / cols)))
    mosaic = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    cells: list[dict] = []
    for index, (space, crop, ox, oy) in enumerate(views):
        row, col = divmod(index, cols)
        cell, scale, lx, ly = _letterbox(crop, cell_w, cell_h)
        mosaic[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = cell
        cells.append(
            {
                "space": space,
                "ox": ox,
                "oy": oy,
                "scale": scale,
                "lx": lx,
                "ly": ly,
                "row": row,
                "col": col,
            }
        )
    imgsz = 640 if max(mosaic.shape[0], mosaic.shape[1]) >= 400 else 480
    raw = _yolo_boxes(model, mosaic, 0, 0, frame.shape, conf=0.04, imgsz=imgsz, max_det=300)
    mapped: list[dict] = []
    for vehicle in raw:
        x1, y1, x2, y2 = vehicle_bbox(vehicle)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        col = int(cx // cell_w)
        row = int(cy // cell_h)
        if col < 0 or row < 0 or col >= cols or row >= rows:
            continue
        index = row * cols + col
        if index >= len(cells):
            continue
        cell = cells[index]
        scale = float(cell["scale"]) or 1.0
        local_x1 = x1 - col * cell_w - cell["lx"]
        local_y1 = y1 - row * cell_h - cell["ly"]
        local_x2 = x2 - col * cell_w - cell["lx"]
        local_y2 = y2 - row * cell_h - cell["ly"]
        bbox = (
            int(cell["ox"] + local_x1 / scale),
            int(cell["oy"] + local_y1 / scale),
            int(cell["ox"] + local_x2 / scale),
            int(cell["oy"] + local_y2 / scale),
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        mapped.append(
            {
                "bbox": bbox,
                "cls": vehicle.get("cls", 2),
                "conf": float(vehicle.get("conf") or 0.0),
                "large": is_large_vehicle(int(vehicle.get("cls") or 2), bbox, frame.shape),
                "from_stall": True,
            }
        )
    return mapped


def vehicle_bbox(vehicle) -> tuple[int, int, int, int]:
    if isinstance(vehicle, dict):
        if "bbox" in vehicle:
            x1, y1, x2, y2 = vehicle["bbox"]
            return int(x1), int(y1), int(x2), int(y2)
        return (
            int(vehicle["x1"]),
            int(vehicle["y1"]),
            int(vehicle["x2"]),
            int(vehicle["y2"]),
        )
    x1, y1, x2, y2 = vehicle
    return int(x1), int(y1), int(x2), int(y2)


def is_large_vehicle(cls: int, bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> bool:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1)) / float(max(1, width * height))
    width_ratio = (x2 - x1) / float(max(1, width))
    if cls in LARGE_CLASS_IDS and (area >= 0.045 or width_ratio >= 0.20):
        return True
    return area >= 0.10 or width_ratio >= 0.32


def detect_vehicles(frame: np.ndarray, spaces: list[dict] | None = None) -> list[dict]:
    model = yolo_model()
    if model is None:
        return []
    brightness = _scene_brightness(frame)
    conf = 0.06 if brightness < 0.28 else 0.10
    if spaces:
        ox1, oy1, ox2, oy2 = _stalls_roi(frame, spaces)
        crop = frame[oy1:oy2, ox1:ox2]
        if crop.size:
            boxes = _yolo_boxes(model, crop, ox1, oy1, frame.shape, conf=conf, imgsz=512, max_det=50)
            if boxes:
                return boxes
    return _yolo_boxes(model, frame, 0, 0, frame.shape, conf=conf, imgsz=512, max_det=50)


def spaces_covered_by_large(spaces: list[dict], vehicles: list, frame_shape: tuple[int, ...]) -> list[str]:
    covered: list[str] = []
    bulky = [v for v in vehicles if isinstance(v, dict)]
    if not bulky:
        return covered
    height, width = frame_shape[:2]
    for space in spaces:
        cache = _space_mask(space, width, height)
        if cache is None:
            continue
        for vehicle in bulky:
            metrics = _vehicle_metrics(cache, vehicle)
            large = bool(vehicle.get("large"))
            if metrics["body_overlap"] >= (0.12 if large else 0.22) or metrics["ground_hit"]:
                covered.append(space["id"])
                break
    return covered


def space_points(space: dict, width: int, height: int) -> np.ndarray:
    raw = space.get("points")
    if isinstance(raw, list) and len(raw) >= 3:
        pts = []
        for point in raw:
            x = float(point.get("x", 0.0) if isinstance(point, dict) else point[0])
            y = float(point.get("y", 0.0) if isinstance(point, dict) else point[1])
            pts.append(
                [
                    int(max(0.0, min(1.0, x)) * (width - 1)),
                    int(max(0.0, min(1.0, y)) * (height - 1)),
                ]
            )
        return np.array(pts, dtype=np.int32)
    x1, y1, x2, y2 = abs_rect(space, width, height)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)


def abs_rect(space: dict, width: int, height: int) -> tuple[int, int, int, int]:
    pts = space.get("points")
    if isinstance(pts, list) and pts:
        xs = [float(p.get("x", 0) if isinstance(p, dict) else p[0]) for p in pts]
        ys = [float(p.get("y", 0) if isinstance(p, dict) else p[1]) for p in pts]
        x1 = int(max(0.0, min(xs)) * width)
        y1 = int(max(0.0, min(ys)) * height)
        x2 = int(min(1.0, max(xs)) * width)
        y2 = int(min(1.0, max(ys)) * height)
        if x2 <= x1 or y2 <= y1:
            return 0, 0, 0, 0
        return x1, y1, x2, y2
    x1 = int(max(0.0, min(1.0, float(space.get("x1", 0)))) * width)
    y1 = int(max(0.0, min(1.0, float(space.get("y1", 0)))) * height)
    x2 = int(max(0.0, min(1.0, float(space.get("x2", 0)))) * width)
    y2 = int(max(0.0, min(1.0, float(space.get("y2", 0)))) * height)
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0
    return x1, y1, x2, y2


def _space_mask(space: dict, width: int, height: int) -> dict | None:
    pts = space_points(space, width, height)
    x, y, w, h = cv2.boundingRect(pts)
    if w < 2 or h < 2:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts - np.array([[x, y]])], 255)
    kernel = max(5, int(0.08 * min(w, h)))
    if kernel % 2 == 0:
        kernel += 1
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    return {
        "pts": pts,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "mask": mask,
        "dilated": dilated,
        "area": max(1, int(cv2.countNonZero(mask))),
    }


REF_SIZE = (48, 48)
_empty_ref_cache: dict[tuple[str, str, str], np.ndarray] = {}


def _empty_ref_path(parking_id: str, bin_id: str, space_id: str) -> Path:
    safe_bin = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in bin_id) or "day"
    safe_space = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in space_id) or "space"
    folder = profile_dir(parking_id) / "empty_refs" / safe_bin
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{safe_space}.png"


def stall_empty_crop(frame: np.ndarray, space: dict) -> np.ndarray | None:
    height, width = frame.shape[:2]
    cache = _space_mask(space, width, height)
    if cache is None:
        return None
    x, y, w, h = cache["x"], cache["y"], cache["w"], cache["h"]
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    masked = cv2.bitwise_and(gray, gray, mask=cache["mask"])
    return cv2.resize(masked, REF_SIZE, interpolation=cv2.INTER_AREA)


def save_empty_stall_ref(parking_id: str, space_id: str, bin_id: str, crop: np.ndarray) -> None:
    if crop is None or crop.size == 0:
        return
    path = _empty_ref_path(parking_id, bin_id, space_id)
    cv2.imwrite(str(path), crop)
    _empty_ref_cache[(parking_id, bin_id, space_id)] = crop


def load_empty_stall_ref(parking_id: str, space_id: str, bin_id: str) -> np.ndarray | None:
    key = (parking_id, bin_id, space_id)
    if key in _empty_ref_cache:
        return _empty_ref_cache[key]
    path = _empty_ref_path(parking_id, bin_id, space_id)
    if not path.exists():
        _empty_ref_cache[key] = None
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    _empty_ref_cache[key] = image
    return image


def empty_ref_score(current: np.ndarray | None, reference: np.ndarray | None) -> float | None:
    """1.0 = current looks like the empty reference; 0.0 = very different."""
    if current is None or reference is None or current.size == 0 or reference.size == 0:
        return None
    if current.shape != reference.shape:
        current = cv2.resize(current, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    mad = float(np.mean(np.abs(current.astype(np.float32) - reference.astype(np.float32)))) / 255.0
    return float(np.clip(1.0 - 2.8 * mad, 0.0, 1.0))


def _footprint_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    top = int(y1 + 0.58 * max(1, y2 - y1))
    return x1, min(top, y2 - 1), x2, y2


def _ground_points(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    points: list[tuple[int, int]] = []
    for dx in (-0.14, -0.05, 0.0, 0.05, 0.14):
        for dy in (0.02, 0.08, 0.16, 0.24):
            points.append((int(cx + dx * bw), int(y2 - dy * bh)))
    return points


def _mask_overlap(cache: dict, bx1: int, by1: int, bx2: int, by2: int) -> float:
    ix1, iy1 = max(bx1, cache["x"]), max(by1, cache["y"])
    ix2, iy2 = min(bx2, cache["x"] + cache["w"]), min(by2, cache["y"] + cache["h"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    mx1, my1 = ix1 - cache["x"], iy1 - cache["y"]
    mx2, my2 = ix2 - cache["x"], iy2 - cache["y"]
    inter = int(cv2.countNonZero(cache["mask"][my1:my2, mx1:mx2]))
    return inter / cache["area"]


def _vehicle_metrics(cache: dict, vehicle) -> dict:
    vx1, vy1, vx2, vy2 = vehicle_bbox(vehicle)
    points = _ground_points((vx1, vy1, vx2, vy2))
    hits = 0
    dilated_hits = 0
    for gx, gy in points:
        lx, ly = gx - cache["x"], gy - cache["y"]
        if 0 <= ly < cache["h"] and 0 <= lx < cache["w"]:
            if cache["mask"][ly, lx] > 0:
                hits += 1
            if cache["dilated"][ly, lx] > 0:
                dilated_hits += 1
    fx1, fy1, fx2, fy2 = _footprint_bbox((vx1, vy1, vx2, vy2))
    body_overlap = _mask_overlap(cache, vx1, vy1, vx2, vy2)
    foot_overlap = _mask_overlap(cache, fx1, fy1, fx2, fy2)
    ground_score = hits / float(max(1, len(points)))
    if hits == 0 and dilated_hits >= 4:
        ground_score = max(ground_score, 0.08)
    return {
        "overlap": float(foot_overlap),
        "body_overlap": float(body_overlap),
        "ground_hit": hits >= 2,
        "ground_score": float(ground_score),
    }


def occupancy_from_vehicles(
    space: dict,
    vehicles: list,
    frame_shape: tuple[int, ...],
) -> tuple[float, bool]:
    height, width = frame_shape[:2]
    cache = _space_mask(space, width, height)
    if cache is None:
        return 0.0, False
    best = 0.0
    ground_hit = False
    for vehicle in vehicles:
        metrics = _vehicle_metrics(cache, vehicle)
        best = max(best, metrics["overlap"])
        ground_hit = ground_hit or metrics["ground_hit"]
    return float(best), ground_hit


def assign_vehicles_to_spaces(
    spaces: list[dict],
    vehicles: list,
    frame_shape: tuple[int, ...],
) -> tuple[dict[int, str], dict[str, dict]]:
    """Each car belongs to the stall where its wheels sit, not where its body overlaps in the photo."""
    height, width = frame_shape[:2]
    caches = {space["id"]: _space_mask(space, width, height) for space in spaces}
    assignment: dict[int, str] = {}
    for index, vehicle in enumerate(vehicles):
        best_id = None
        best_key = (-1.0, -1.0)
        for space in spaces:
            cache = caches.get(space["id"])
            if cache is None:
                continue
            metrics = _vehicle_metrics(cache, vehicle)
            key = (metrics["ground_score"], metrics["overlap"])
            if key > best_key:
                best_key = key
                best_id = space["id"]
        ground_score, overlap = best_key
        large = bool(vehicle.get("large")) if isinstance(vehicle, dict) else False
        if best_id and (
            ground_score >= 0.08
            or overlap >= 0.14
            or (large and overlap >= 0.12)
        ):
            assignment[index] = best_id
        elif best_id and overlap < 0.20:
            runners = []
            for space in spaces:
                cache = caches.get(space["id"])
                if cache is None:
                    continue
                metrics = _vehicle_metrics(cache, vehicle)
                runners.append((metrics["overlap"], space["id"]))
            runners.sort(reverse=True)
            if runners and runners[0][0] >= 0.14:
                if len(runners) == 1 or runners[0][0] >= (runners[1][0] + 0.05):
                    assignment[index] = runners[0][1]

    per_space = {
        space["id"]: {
            "own": [],
            "foreign": [],
            "own_overlap": 0.0,
            "own_ground": False,
            "body_overlap": 0.0,
            "spill": 0.0,
            "conf": 0.0,
        }
        for space in spaces
    }
    for index, vehicle in enumerate(vehicles):
        owner = assignment.get(index)
        large = bool(vehicle.get("large")) if isinstance(vehicle, dict) else False
        for space in spaces:
            cache = caches.get(space["id"])
            if cache is None:
                continue
            metrics = _vehicle_metrics(cache, vehicle)
            row = per_space[space["id"]]
            row["body_overlap"] = max(row["body_overlap"], metrics["body_overlap"])
            extra = metrics["body_overlap"] >= (0.12 if large else 0.20)
            if owner == space["id"] or extra:
                row["own"].append(vehicle)
                row["own_overlap"] = max(row["own_overlap"], metrics["overlap"], metrics["body_overlap"])
                row["own_ground"] = row["own_ground"] or metrics["ground_hit"] or metrics["body_overlap"] >= 0.18
                row["conf"] = max(
                    row["conf"],
                    float(vehicle.get("conf") or 0.0) if isinstance(vehicle, dict) else 0.0,
                )
            else:
                row["foreign"].append(vehicle)
                row["spill"] = max(row["spill"], metrics["overlap"])
    return assignment, per_space


def stall_appearance_score(frame: np.ndarray, space: dict, foreign_vehicles: list) -> float:
    height, width = frame.shape[:2]
    cache = _space_mask(space, width, height)
    if cache is None:
        return 0.0
    mask = cache["mask"].copy()
    x, y, w, h = cache["x"], cache["y"], cache["w"], cache["h"]
    for vehicle in foreign_vehicles:
        vx1, vy1, vx2, vy2 = vehicle_bbox(vehicle)
        fx1, fy1, fx2, fy2 = _footprint_bbox((vx1, vy1, vx2, vy2))
        rx1, ry1 = max(0, fx1 - x), max(0, fy1 - y)
        rx2, ry2 = min(w, fx2 - x), min(h, fy2 - y)
        if rx2 > rx1 and ry2 > ry1:
            mask[ry1:ry2, rx1:rx2] = 0
    remaining = int(cv2.countNonZero(mask))
    if remaining < 24:
        return 0.0
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    edges = float(cv2.Canny(masked, 40, 120).mean() / 255.0)
    return float(np.clip(edges / 0.12, 0.0, 1.0))


def extract_features(
    frame: np.ndarray,
    space: dict,
    vehicles: list,
    own_vehicles: list | None = None,
    foreign_vehicles: list | None = None,
    when=None,
    scene_brightness: float | None = None,
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    cache = _space_mask(space, width, height)
    if cache is None or cache["w"] < 4 or cache["h"] < 4:
        return None
    x, y, w, h = cache["x"], cache["y"], cache["w"], cache["h"]
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    mask = cache["mask"].copy()
    foreign = foreign_vehicles or []
    own = own_vehicles if own_vehicles is not None else vehicles
    for vehicle in foreign:
        vx1, vy1, vx2, vy2 = vehicle_bbox(vehicle)
        rx1, ry1 = max(0, vx1 - x), max(0, vy1 - y)
        rx2, ry2 = min(w, vx2 - x), min(h, vy2 - y)
        if rx2 > rx1 and ry2 > ry1:
            mask[ry1:ry2, rx1:rx2] = 0
    if int(cv2.countNonZero(mask)) < 12:
        mask = cache["mask"]
    crop = cv2.bitwise_and(crop, crop, mask=mask)
    crop = cv2.resize(crop, (64, 64))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = np.concatenate(
        [
            cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten(),
            cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten(),
            cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten(),
        ]
    )
    hist = hist / (hist.sum() + 1e-6)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = float(cv2.Canny(gray, 50, 150).mean() / 255.0)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0)
    overlap, ground_hit = occupancy_from_vehicles(space, own, frame.shape)
    if ground_hit:
        overlap = max(overlap, 0.85)
    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    area = (w * h) / (width * height)
    stall_bright = float(hsv[:, :, 2].mean() / 255.0)
    scene = float(scene_brightness) if scene_brightness is not None else _scene_brightness(frame)
    context = _time_of_day_features(when) + [scene, stall_bright]
    return np.concatenate([hist, [edges, lap, overlap, cx, cy, area], context])


def add_sample(parking_id: str, frame: np.ndarray, space_id: str, occupied: bool) -> dict:
    space_list = load_spaces(parking_id)
    spaces = {row["id"]: row for row in space_list}
    space = spaces.get(space_id)
    if space is None:
        raise ValueError(f"Plaza ezezaguna: {space_id}")
    vehicles = detect_vehicles(frame, spaces=space_list)
    _, per_space = assign_vehicles_to_spaces(space_list, vehicles, frame.shape)
    view = per_space.get(space_id) or {}
    features = extract_features(
        frame,
        space,
        vehicles,
        own_vehicles=view.get("own") or [],
        foreign_vehicles=view.get("foreign") or [],
    )
    if features is None:
        raise ValueError("Ezin izan da plaza moztu")
    labels_path = profile_dir(parking_id) / "labels.json"
    labels = _read_json(labels_path, [])
    labels.append(
        {
            "space_id": space_id,
            "occupied": occupied,
            "features": features.tolist(),
            "created_at": _utc_now(),
            "approved": True,
            "source": "live_error",
        }
    )
    _write_json(labels_path, labels)
    return refresh_sample_counts(parking_id)


def _load_model(parking_id: str):
    meta = load_meta(parking_id)
    if not meta.get("approved_training"):
        return None
    path = profile_dir(parking_id) / "model.joblib"
    if not path.exists():
        _model_cache.pop(parking_id, None)
        return None
    mtime = path.stat().st_mtime
    cached = _model_cache.get(parking_id)
    if cached and cached[0] == mtime:
        return cached[1]
    model = joblib.load(path)
    _model_cache[parking_id] = (mtime, model)
    return model


def model_is_active(meta: dict | None = None, parking_id: str | None = None) -> bool:
    meta = meta if meta is not None else load_meta(parking_id or "")
    return bool(meta.get("approved_training") and meta.get("model_trained"))


def training_gap(parking_id: str) -> dict:
    free, occupied = count_samples(parking_id)
    return {
        "samples_free": free,
        "samples_occupied": occupied,
        "need_free": max(0, MIN_SAMPLES_PER_CLASS - free),
        "need_occupied": max(0, MIN_SAMPLES_PER_CLASS - occupied),
    }


def train_model(parking_id: str) -> dict:
    labels = [
        row
        for row in _read_json(profile_dir(parking_id) / "labels.json", [])
        if _is_approved_label(row) and _align_features(row)
    ]
    free = [row for row in labels if not row.get("occupied")]
    occupied = [row for row in labels if row.get("occupied")]
    if len(free) < MIN_SAMPLES_PER_CLASS or len(occupied) < MIN_SAMPLES_PER_CLASS:
        raise ValueError(
            f"Gutxienez {MIN_SAMPLES_PER_CLASS} adibide libre eta {MIN_SAMPLES_PER_CLASS} okupatu behar dira. "
            f"Orain: {len(free)} libre, {len(occupied)} okupatu."
        )
    x = np.array([_align_features(row) for row in labels], dtype=np.float32)
    y = np.array([1 if row["occupied"] else 0 for row in labels], dtype=np.int32)
    weights = np.array([float(row.get("weight", 1.0)) for row in labels], dtype=np.float32)
    model = RandomForestClassifier(
        n_estimators=220,
        class_weight="balanced_subsample",
        random_state=42,
        min_samples_leaf=2,
        max_depth=14,
        oob_score=len(labels) >= 12,
        n_jobs=1,
    )
    model.fit(x, y, sample_weight=weights)
    if getattr(model, "oob_score_", None) is not None:
        accuracy = float(model.oob_score_)
    else:
        accuracy = float(model.score(x, y, sample_weight=weights))
    joblib.dump(model, profile_dir(parking_id) / "model.joblib")
    _model_cache.pop(parking_id, None)
    meta = refresh_sample_counts(parking_id)
    meta.update(
        {
            "model_trained": True,
            "accuracy": round(accuracy, 3),
            "trained_at": _utc_now(),
            "n_labels": len(labels),
            "need_free": 0,
            "need_occupied": 0,
            "approved_training": True,
        }
    )
    _save_meta(parking_id, meta)
    return meta


def complete_space_labels(parking_id: str, spaces: dict | None) -> dict[str, bool]:
    incoming = {str(key): bool(value) for key, value in (spaces or {}).items() if value is not None}
    return {str(row["id"]): bool(incoming.get(str(row["id"]), False)) for row in load_spaces(parking_id)}


def remember_correction(parking_id: str, space_id: str, occupied: bool, overlap: float = 0.0) -> None:
    store = _sticky.setdefault(parking_id, {})
    store[space_id] = {
        "occupied": bool(occupied),
        "until": time.time() + STICKY_TTL_SECONDS,
        "overlap": float(overlap),
    }
    seed_stall_state(parking_id, space_id, occupied, lock=True)


def apply_sticky(parking_id: str, occupancy: dict[str, bool], scores: dict) -> dict[str, bool]:
    now = time.time()
    store = _sticky.get(parking_id) or {}
    expired: list[str] = []
    for space_id, item in store.items():
        if now > float(item.get("until") or 0):
            expired.append(space_id)
            continue
        info = scores.get(space_id) or {}
        want = bool(item.get("occupied"))
        overlap = float(info.get("overlap") or 0.0)
        ground = bool(info.get("ground_hit"))
        proba = info.get("proba")
        empty_score = info.get("empty_score")
        if want:
            flow = str(info.get("flow") or "")
            body = float(info.get("body_overlap") or 0.0)
            clearly_empty = (
                (empty_score is not None and float(empty_score) >= 0.80)
                or (proba is not None and float(proba) <= 0.22)
                or flow in {"gone", "leaving"}
                or (overlap < 0.10 and body < 0.10 and not ground)
            )
            if clearly_empty and overlap < 0.14 and not ground:
                expired.append(space_id)
                continue
        elif ground and overlap >= max(0.28, float(item.get("overlap") or 0) + 0.18):
            expired.append(space_id)
            continue
        occupancy[space_id] = want
    for space_id in expired:
        store.pop(space_id, None)
    return occupancy


def _appearance_vec(features) -> np.ndarray | None:
    if features is None:
        return None
    arr = np.asarray(features, dtype=np.float32).flatten()
    if arr.size < APPEARANCE_DIM:
        return None
    return arr[:APPEARANCE_DIM].copy()


def _appear_dist(a, b) -> float | None:
    if a is None or b is None:
        return None
    hist = 0.5 * float(np.abs(a[:32] - b[:32]).sum())
    extra = float(np.linalg.norm(a[32:APPEARANCE_DIM] - b[32:APPEARANCE_DIM]))
    return hist + 0.45 * extra


def _ema_vec(old, new, alpha: float = 0.18):
    if new is None:
        return old
    if old is None:
        return new
    return (1.0 - alpha) * old + alpha * new


def _blank_stall_row() -> dict:
    return {
        "occupied": None,
        "pending": None,
        "pending_n": 0,
        "feat_free": None,
        "feat_occ": None,
        "refs": {},
        "stable_n": 0,
        "user_lock": False,
        "user_lock_until": 0.0,
    }


def _scene_from_features(features) -> float:
    arr = np.asarray(features, dtype=np.float32).flatten()
    if arr.size >= 41:
        return float(np.clip(arr[40], 0.0, 1.0))
    return 0.5


def _bin_slot(row: dict, bin_id: str) -> dict:
    refs = row.setdefault("refs", {})
    slot = refs.get(bin_id)
    if not isinstance(slot, dict):
        slot = {"free": None, "occ": None}
        refs[bin_id] = slot
    return slot


def _templates_for(row: dict, bin_id: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    refs = row.get("refs") if isinstance(row.get("refs"), dict) else {}
    slot = refs.get(bin_id) if isinstance(refs.get(bin_id), dict) else None
    feat_free = _appearance_vec(slot.get("free")) if slot else None
    feat_occ = _appearance_vec(slot.get("occ")) if slot else None
    if feat_free is not None or feat_occ is not None:
        return feat_free, feat_occ
    light = _light_from_bin(bin_id)
    idx = LIGHT_ORDER.index(light) if light in LIGHT_ORDER else 2
    for step in (1, -1, 2, -2):
        near = LIGHT_ORDER[(idx + step) % len(LIGHT_ORDER)]
        for key, value in refs.items():
            if not isinstance(value, dict):
                continue
            if _light_from_bin(str(key)) != near:
                continue
            feat_free = _appearance_vec(value.get("free"))
            feat_occ = _appearance_vec(value.get("occ"))
            if feat_free is not None or feat_occ is not None:
                return feat_free, feat_occ
    return _appearance_vec(row.get("feat_free")), _appearance_vec(row.get("feat_occ"))


def _store_template(row: dict, bin_id: str, occupied: bool, vec, ema: bool = False) -> None:
    if vec is None:
        return
    slot = _bin_slot(row, bin_id)
    key = "occ" if occupied else "free"
    slot[key] = _ema_vec(slot.get(key), vec) if ema else vec
    if occupied:
        row["feat_occ"] = slot[key]
    else:
        row["feat_free"] = slot[key]


def _load_history(parking_id: str) -> dict[str, dict]:
    store = _history.setdefault(parking_id, {})
    if parking_id in _history_loaded:
        return store
    _history_loaded.add(parking_id)
    raw = _read_json(profile_dir(parking_id) / "occupancy_state.json", {})
    if not isinstance(raw, dict):
        return store
    for sid, row in raw.items():
        occupied = None
        feat_free = None
        feat_occ = None
        refs = {}
        if isinstance(row, bool):
            occupied = row
        elif isinstance(row, dict):
            if row.get("occupied") is not None:
                occupied = bool(row.get("occupied"))
            feat_free = _appearance_vec(row.get("feat_free"))
            feat_occ = _appearance_vec(row.get("feat_occ"))
            raw_refs = row.get("refs")
            if isinstance(raw_refs, dict):
                for key, value in raw_refs.items():
                    if not isinstance(value, dict):
                        continue
                    refs[str(key)] = {
                        "free": _appearance_vec(value.get("free")),
                        "occ": _appearance_vec(value.get("occ")),
                    }
        else:
            continue
        current = store.get(sid) or _blank_stall_row()
        if current.get("occupied") is None:
            current["occupied"] = occupied
            current["stable_n"] = 4 if occupied is not None else 0
        if isinstance(row, dict):
            if row.get("user_lock"):
                current["user_lock"] = True
                current["user_lock_until"] = float(row.get("user_lock_until") or 0)
        if current.get("feat_free") is None:
            current["feat_free"] = feat_free
        if current.get("feat_occ") is None:
            current["feat_occ"] = feat_occ
        merged = current.setdefault("refs", {})
        for key, value in refs.items():
            slot = merged.setdefault(key, {"free": None, "occ": None})
            if slot.get("free") is None:
                slot["free"] = value.get("free")
            if slot.get("occ") is None:
                slot["occ"] = value.get("occ")
        store[sid] = current
    return store


def _save_history(parking_id: str) -> None:
    store = _history.get(parking_id) or {}
    payload = {}
    for sid, row in store.items():
        if row.get("occupied") is None:
            continue
        item = {"occupied": bool(row["occupied"])}
        if row.get("user_lock"):
            item["user_lock"] = True
            item["user_lock_until"] = float(row.get("user_lock_until") or 0)
        if row.get("feat_free") is not None:
            item["feat_free"] = [float(x) for x in np.asarray(row["feat_free"]).tolist()]
        if row.get("feat_occ") is not None:
            item["feat_occ"] = [float(x) for x in np.asarray(row["feat_occ"]).tolist()]
        refs_out = {}
        for key, value in (row.get("refs") or {}).items():
            if not isinstance(value, dict):
                continue
            slot = {}
            if value.get("free") is not None:
                slot["free"] = [float(x) for x in np.asarray(value["free"]).tolist()]
            if value.get("occ") is not None:
                slot["occ"] = [float(x) for x in np.asarray(value["occ"]).tolist()]
            if slot:
                refs_out[str(key)] = slot
        if refs_out:
            item["refs"] = refs_out
        payload[sid] = item
    _write_json(profile_dir(parking_id) / "occupancy_state.json", payload)


def previous_occupancy_map(parking_id: str) -> dict[str, bool]:
    store = _load_history(parking_id)
    return {
        sid: bool(row["occupied"])
        for sid, row in store.items()
        if row.get("occupied") is not None
    }


def _label_prototypes(parking_id: str) -> dict:
    path = profile_dir(parking_id) / "labels.json"
    mtime = path.stat().st_mtime if path.exists() else 0.0
    cached = _proto_cache.get(parking_id)
    if cached and cached[0] == mtime:
        return cached[1]

    def _mean(vecs: list[np.ndarray]) -> np.ndarray | None:
        if not vecs:
            return None
        return np.mean(np.stack(vecs, axis=0), axis=0)

    occ: list[np.ndarray] = []
    free: list[np.ndarray] = []
    by_space: dict[str, dict[str, list]] = {}
    for row in _read_json(path, []):
        if not _is_approved_label(row):
            continue
        vec = _appearance_vec(row.get("features"))
        if vec is None:
            continue
        sid = str(row.get("space_id") or "")
        slot = by_space.setdefault(sid, {"occ": [], "free": []})
        if row.get("occupied"):
            occ.append(vec)
            slot["occ"].append(vec)
        else:
            free.append(vec)
            slot["free"].append(vec)
    proto = {
        "occ": _mean(occ),
        "free": _mean(free),
        "n_occ": len(occ),
        "n_free": len(free),
        "spaces": {
            sid: {"occ": _mean(slot["occ"]), "free": _mean(slot["free"])}
            for sid, slot in by_space.items()
        },
    }
    _proto_cache[parking_id] = (mtime, proto)
    return proto


def _prototype_vote(parking_id: str, space_id: str, features, appearance: float) -> bool | None:
    proto = _label_prototypes(parking_id)
    vec = _appearance_vec(features)
    if vec is None or not proto["n_occ"]:
        return None
    local = proto["spaces"].get(space_id) or {}
    occ = local.get("occ") if local.get("occ") is not None else proto["occ"]
    free = local.get("free") if local.get("free") is not None else proto["free"]
    d_occ = _appear_dist(vec, occ)
    d_free = _appear_dist(vec, free)
    if d_occ is not None and d_free is not None:
        if d_occ + 0.04 < d_free:
            return True
        if d_free + 0.04 < d_occ:
            return False
    if d_occ is not None and d_occ < 0.40:
        return True
    if appearance >= 0.55 and (d_occ is None or d_occ < 0.62):
        return True
    return None


def seed_stall_state(
    parking_id: str,
    space_id: str,
    occupied: bool,
    features=None,
    persist: bool = True,
    lock: bool = False,
) -> None:
    store = _load_history(parking_id)
    row = store.setdefault(space_id, _blank_stall_row())
    row["occupied"] = bool(occupied)
    row["pending"] = None
    row["pending_n"] = 0
    row["stable_n"] = 8
    if lock:
        row["user_lock"] = True
        row["user_lock_until"] = time.time() + STICKY_TTL_SECONDS
        if occupied:
            row["feat_free"] = None
            for slot in (row.get("refs") or {}).values():
                if isinstance(slot, dict):
                    slot["free"] = None
    vec = _appearance_vec(features)
    if vec is not None:
        bin_id = lighting_bin(_scene_from_features(features))
        _store_template(row, bin_id, occupied, vec, ema=False)
        if occupied:
            row["feat_occ"] = vec
        else:
            row["feat_free"] = vec
    if persist:
        _save_history(parking_id)


def seed_occupancy_map(
    parking_id: str,
    spaces_map: dict[str, bool],
    features_by_space: dict | None = None,
    frame=None,
) -> None:
    features_by_space = features_by_space or {}
    bin_id = lighting_bin(_scene_brightness(frame)) if frame is not None else None
    space_defs = {row["id"]: row for row in load_spaces(parking_id)} if frame is not None else {}
    for space_id, occupied in spaces_map.items():
        seed_stall_state(parking_id, space_id, bool(occupied), features_by_space.get(space_id), persist=False)
        if occupied or frame is None or not bin_id:
            continue
        space = space_defs.get(space_id)
        crop = stall_empty_crop(frame, space) if space is not None else None
        if crop is not None:
            save_empty_stall_ref(parking_id, space_id, bin_id, crop)
    _save_history(parking_id)


def stabilize_occupancy(
    parking_id: str,
    occupancy: dict[str, bool],
    scores: dict[str, dict],
) -> dict[str, bool]:
    """Publish a stall change only after several consecutive frames agree."""
    store = _load_history(parking_id)
    dirty = False
    for space_id, evidence in list(occupancy.items()):
        row = store.setdefault(space_id, _blank_stall_row())
        prev = row.get("occupied")
        if prev is None:
            prev = False
            row["occupied"] = False
            dirty = True
        info = scores.get(space_id) or {}
        vec = _appearance_vec(info.get("features"))
        overlap = float(info.get("overlap") or 0.0)
        ground = bool(info.get("ground_hit"))
        conf = float(info.get("conf") or 0.0)
        bin_id = str(info.get("light_bin") or lighting_bin(float(info.get("scene") or 0.5)))
        feat_free, feat_occ = _templates_for(row, bin_id)
        d_free = _appear_dist(vec, feat_free)
        d_occ = _appear_dist(vec, feat_occ)
        looks_empty = False
        empty_score = info.get("empty_score")
        body = float(info.get("body_overlap") or 0.0)
        appearance = float(info.get("appearance") or 0.0)
        if empty_score is not None and float(empty_score) >= 0.90 and appearance < 0.14 and body < 0.08:
            looks_empty = True
        strong_occ = (
            (ground and overlap >= 0.14 and conf >= 0.20)
            or (overlap >= 0.28 and conf >= 0.28)
            or body >= 0.16
        )
        lock_until = float(row.get("user_lock_until") or 0)
        flow = str(info.get("flow") or "")
        if flow in {"entering", "parked"} and (ground or overlap >= 0.12 or body >= 0.12):
            strong_occ = True
        if flow in {"leaving", "gone"}:
            looks_empty = True
        if flow == "false_occupy":
            evidence = False
        if flow == "false_free" and (ground or overlap >= 0.18 or body >= 0.16):
            evidence = True
        if row.get("user_lock") and time.time() < lock_until:
            locked = bool(row.get("occupied"))
            if locked and looks_empty:
                row["user_lock"] = False
                dirty = True
            elif locked and not looks_empty:
                occupancy[space_id] = True
                row["pending"] = None
                row["pending_n"] = 0
                continue
            elif not locked and not strong_occ:
                occupancy[space_id] = False
                row["pending"] = None
                row["pending_n"] = 0
                continue
            else:
                row["user_lock"] = False
                dirty = True
        if evidence and looks_empty and not strong_occ:
            evidence = False
        evidence = bool(evidence)

        if evidence == bool(prev):
            if row.get("pending") is not None or int(row.get("pending_n") or 0):
                dirty = True
            row["pending"] = None
            row["pending_n"] = 0
            row["stable_n"] = int(row.get("stable_n") or 0) + 1
            if vec is not None and row["stable_n"] >= 8 and row["stable_n"] % 8 == 0:
                _store_template(row, bin_id, prev, vec, ema=True)
            occupancy[space_id] = bool(prev)
            crop = info.get("empty_crop")
            if (
                not prev
                and crop is not None
                and overlap < 0.06
                and row["stable_n"] in {6, 30, 90}
            ):
                save_empty_stall_ref(parking_id, space_id, bin_id, crop)
            continue

        need = CONFIRM_TO_FREE if prev and not evidence else CONFIRM_TO_OCCUPIED
        if prev and not evidence and flow in {"gone", "leaving"}:
            need = 1 if flow == "gone" else 2
        target = evidence
        if row.get("pending") == target:
            row["pending_n"] = int(row.get("pending_n") or 0) + 1
        else:
            row["pending"] = target
            row["pending_n"] = 1
        dirty = True
        if row["pending_n"] >= need:
            row["occupied"] = target
            row["pending"] = None
            row["pending_n"] = 0
            row["stable_n"] = 0
            occupancy[space_id] = target
            _store_template(row, bin_id, target, vec, ema=False)
        else:
            occupancy[space_id] = bool(prev)

    if dirty:
        _save_history(parking_id)
    return occupancy


def _fuse_decision(
    own_overlap: float,
    own_ground: bool,
    proba: float | None,
    spill: float,
    threshold: float,
    appearance: float = 0.0,
    previous: bool | None = None,
    conf: float = 0.0,
) -> bool:
    occupy_thr = max(float(threshold), 0.14)
    has_car = (own_ground and own_overlap >= 0.08) or own_overlap >= occupy_thr
    if conf and conf < 0.06 and own_overlap < 0.22:
        has_car = False
    if spill >= 0.22 and spill > own_overlap and own_overlap < 0.28:
        has_car = False
    if has_car:
        if proba is not None and proba <= 0.18 and appearance < 0.22:
            return False
        return True
    if proba is not None and proba >= 0.68:
        return True
    if previous and proba is not None and proba >= 0.52 and own_overlap >= 0.10:
        return True
    if previous and appearance >= 0.62 and own_overlap >= 0.08:
        return True
    if appearance >= 0.70:
        return True
    return False


def infer_occupancy(
    parking_id: str,
    frame: np.ndarray,
    spaces: list[dict] | None = None,
    meta: dict | None = None,
    config: dict | None = None,
    vehicles: list[dict] | None = None,
) -> tuple[dict[str, bool], dict[str, dict], list[dict]]:
    spaces = spaces if spaces is not None else load_spaces(parking_id)
    config = config if config is not None else load_config(parking_id)
    meta = meta if meta is not None else load_meta(parking_id)
    threshold = float(config.get("overlap_threshold", 0.12))
    vehicles = detect_vehicles(frame, spaces=spaces) if vehicles is None else vehicles
    model = _load_model(parking_id) if model_is_active(meta) else None
    occupancy: dict[str, bool] = {}
    scores: dict[str, dict] = {}
    previous = previous_occupancy_map(parking_id)
    scene = _scene_brightness(frame)
    bin_id = lighting_bin(scene)
    _, per_space = assign_vehicles_to_spaces(spaces, vehicles, frame.shape)

    for space in spaces:
        space_id = space["id"]
        view = per_space.get(space_id) or {}
        own = view.get("own") or []
        foreign = view.get("foreign") or []
        own_overlap = float(view.get("own_overlap") or 0.0)
        own_ground = bool(view.get("own_ground"))
        spill = float(view.get("spill") or 0.0)
        prev_occ = previous.get(space_id)
        empty_crop = stall_empty_crop(frame, space)
        empty_score = empty_ref_score(empty_crop, load_empty_stall_ref(parking_id, space_id, bin_id))
        looks_like_empty = empty_score is not None and empty_score >= 0.90
        proto = _label_prototypes(parking_id)
        has_proto = bool(proto.get("n_occ"))
        if not own and not prev_occ and spill < 0.12 and model is None and not has_proto:
            occupancy[space_id] = False
            scores[space_id] = {
                "overlap": 0.0,
                "ground_hit": False,
                "proba": None,
                "conf": 0.0,
                "spill": spill,
                "appearance": 0.0,
                "features": None,
                "scene": scene,
                "light_bin": bin_id,
                "empty_score": empty_score,
                "empty_crop": empty_crop,
            }
            continue
        features = extract_features(
            frame,
            space,
            vehicles,
            own_vehicles=own,
            foreign_vehicles=foreign,
            scene_brightness=scene,
        )
        proba = None
        if model is not None and features is not None:
            try:
                proba = float(model.predict_proba([features])[0][1])
            except Exception:
                proba = None
        appearance = stall_appearance_score(frame, space, foreign)
        body = float(view.get("body_overlap") or 0.0)
        occupied = _fuse_decision(
            own_overlap,
            own_ground,
            proba,
            spill,
            threshold,
            appearance,
            previous=previous.get(space_id),
            conf=float(view.get("conf") or 0.0),
        )
        vote = _prototype_vote(parking_id, space_id, features, appearance)
        if body >= 0.14:
            occupied = True
        elif vote is True:
            occupied = True
        elif vote is False and not (own_ground and own_overlap >= 0.16) and body < 0.10:
            occupied = False
        if (
            looks_like_empty
            and body < 0.10
            and not (own_ground and own_overlap >= 0.22)
            and vote is not True
        ):
            if proba is None or proba < 0.55:
                occupied = False
        if proto.get("n_occ", 0) >= 8 and proto["n_occ"] >= 3 * max(1, int(proto.get("n_free") or 0)):
            clearly_empty = looks_like_empty and appearance < 0.16 and body < 0.08
            car_now = appearance >= 0.22 or body >= 0.12 or (own_ground and own_overlap >= 0.12)
            if not clearly_empty and (car_now or vote is True):
                occupied = True
        occupancy[space_id] = bool(occupied)
        scores[space_id] = {
            "overlap": own_overlap,
            "ground_hit": own_ground,
            "body_overlap": body,
            "proba": proba,
            "conf": float(view.get("conf") or 0.0),
            "spill": spill,
            "appearance": appearance,
            "features": None if features is None else features.tolist(),
            "scene": scene,
            "light_bin": bin_id,
            "empty_score": empty_score,
            "empty_crop": empty_crop,
        }

    for space_id in spaces_covered_by_large(spaces, vehicles, frame.shape):
        occupancy[space_id] = True
        info = scores.setdefault(space_id, {})
        info["overlap"] = max(float(info.get("overlap") or 0), 0.35)
        info["body_overlap"] = max(float(info.get("body_overlap") or 0), 0.22)
    return occupancy, scores, vehicles


def predict_occupancy(
    parking_id: str,
    frame: np.ndarray,
    vehicles: list[dict] | None = None,
) -> dict[str, bool]:
    occupancy, scores, vehicles = infer_occupancy(parking_id, frame, vehicles=vehicles)
    from tracker import apply_stall_flow, apply_temporal

    spaces = load_spaces(parking_id)
    occupancy, _, _ = apply_temporal(parking_id, spaces, vehicles, occupancy, frame.shape, scores)
    occupancy, _ = apply_stall_flow(parking_id, spaces, vehicles, occupancy, frame.shape, scores)
    occupancy = stabilize_occupancy(parking_id, occupancy, scores)
    occupancy = apply_sticky(parking_id, occupancy, scores)
    return occupancy


def describe_correction_effect(
    parking_id: str,
    space_id: str,
    occupied: bool,
    predicted,
    train_meta: dict,
) -> dict:
    state = "OKUPATUTA" if occupied else "LIBRE"
    trained = bool(train_meta.get("just_trained"))
    model_agrees = predicted is not None and bool(predicted) == bool(occupied)
    need_free = int(train_meta.get("need_free") or 0)
    need_occupied = int(train_meta.get("need_occupied") or 0)
    accuracy = train_meta.get("accuracy")
    if trained and model_agrees:
        effect = "learned"
        kind = "ok"
        message = (
            f"Zuzenketa algoritmoan sartu da: {space_id} → {state}. "
            f"Modeloa birentrenatu da eta orain plaza hau zuzen sailkatzen du."
        )
    elif trained and not model_agrees:
        effect = "trained_disagrees"
        kind = "warn"
        message = (
            f"Zuzenketa gordeta eta modelo birentrenatuta ({space_id} → {state}), "
            f"baina fotograma honetan algoritmoak oraindik ez du asmatzen. "
            f"Beste zuzenketa batzuek hobetuko dute."
        )
    elif need_free or need_occupied:
        effect = "saved_need_more"
        kind = "warn"
        missing = []
        if need_free:
            missing.append(f"{need_free} libre")
        if need_occupied:
            missing.append(f"{need_occupied} okupatu")
        message = (
            f"Zuzenketa gordeta: {space_id} → {state}. Adibidea sartu da, "
            f"baina modeloak oraindik ez du ikasi. Falta: {' eta '.join(missing)}. "
            f"Orain: {train_meta.get('samples_free', 0)} libre, {train_meta.get('samples_occupied', 0)} okupatu."
        )
    else:
        effect = "saved"
        kind = "warn"
        message = (
            f"Zuzenketa gordeta: {space_id} → {state}. Adibidea entrenamenduan sartu da, "
            f"baina algoritmoak fotograma honetan oraindik ez du zuzen sailkatzen."
        )
    if trained and accuracy is not None:
        message += f" Zehaztasuna: {accuracy}."
    return {
        "effect": effect,
        "kind": kind,
        "message": message,
        "learned": effect == "learned",
        "model_agrees": model_agrees,
        "just_trained": trained,
        "need_free": need_free,
        "need_occupied": need_occupied,
    }


def annotate_occupancy(
    frame: np.ndarray,
    spaces: list[dict],
    occupancy: dict[str, bool],
    vehicles: list | None = None,
    occluded: list[str] | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    for space in spaces:
        occupied = occupancy.get(space["id"])
        hidden = space["id"] in (occluded or [])
        if hidden:
            color = (30, 140, 220)
        elif occupied is True:
            color = (40, 40, 200)
        elif occupied is False:
            color = (40, 180, 70)
        else:
            color = (160, 150, 40)
        pts = space_points(space, frame.shape[1], frame.shape[0])
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0, annotated)
        cv2.polylines(annotated, [pts], True, color, 2, cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(annotated, (int(px), int(py)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        if hidden:
            state = "EZKUTU"
        else:
            state = "OKUP" if occupied is True else ("LIBRE" if occupied is False else "?")
        cv2.putText(
            annotated,
            f"{space['id']} {state}",
            (int(pts[0][0]) + 4, max(16, int(pts[0][1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    for vehicle in vehicles or []:
        vx1, vy1, vx2, vy2 = vehicle_bbox(vehicle)
        color = (40, 90, 255) if isinstance(vehicle, dict) and vehicle.get("large") else (255, 180, 0)
        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), color, 2 if color == (40, 90, 255) else 1)
    return annotated


def analyze_frame(
    parking_id: str,
    frame: np.ndarray,
    vehicles: list[dict] | None = None,
) -> tuple[dict, np.ndarray]:
    spaces = load_spaces(parking_id)
    config = load_config(parking_id)
    meta = load_meta(parking_id)
    occupancy, scores, vehicles = infer_occupancy(
        parking_id, frame, spaces, meta, config, vehicles=vehicles
    )

    from tracker import apply_stall_flow, apply_temporal

    occupancy, occluded, hidden_events = apply_temporal(
        parking_id, spaces, vehicles, occupancy, frame.shape, scores
    )
    occupancy, flow_events = apply_stall_flow(
        parking_id, spaces, vehicles, occupancy, frame.shape, scores
    )
    occupancy = stabilize_occupancy(parking_id, occupancy, scores)
    occupancy = apply_sticky(parking_id, occupancy, scores)
    store = _load_history(parking_id)
    changed = False
    for space_id, occ in occupancy.items():
        row = store.setdefault(space_id, _blank_stall_row())
        sticky = (_sticky.get(parking_id) or {}).get(space_id)
        if sticky and bool(occ) != bool(row.get("occupied")):
            row["occupied"] = bool(occ)
            row["pending"] = None
            row["pending_n"] = 0
            changed = True
    if changed:
        _save_history(parking_id)
    annotated = annotate_occupancy(frame, spaces, occupancy, vehicles, occluded)

    status = {
        "ok": True,
        "spaces": occupancy,
        "vehicles": len(vehicles),
        "large_vehicles": sum(1 for v in vehicles if v.get("large")),
        "occluded": occluded,
        "hidden_freed": hidden_events,
        "flow": {sid: ev for sid, ev in (flow_events or {}).items() if ev and ev != "none"},
        "camera_ok": True,
        "calibrated": bool(spaces),
        "model_trained": model_is_active(meta),
        "yolo_ok": bool(HAS_YOLO),
        "samples_free": meta.get("samples_free", 0),
        "samples_occupied": meta.get("samples_occupied", 0),
        "accuracy": meta.get("accuracy"),
        "updated_at": _utc_now(),
        "error": None,
    }
    return status, annotated
