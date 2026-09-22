from __future__ import annotations

import time

from occupancy import occupancy_from_vehicles, space_points, vehicle_bbox


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / area


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _approach(history: list, stall_cx: float, stall_cy: float) -> float:
    """+1 toward stall, -1 away, 0 still. Uses last two tracked centers."""
    if len(history) < 2:
        return 0.0
    x0, y0, _ = history[-2]
    x1, y1, _ = history[-1]
    vx, vy = x1 - x0, y1 - y0
    speed = (vx * vx + vy * vy) ** 0.5
    if speed < 5.0:
        return 0.0
    dx, dy = stall_cx - x0, stall_cy - y0
    dist = (dx * dx + dy * dy) ** 0.5 or 1.0
    return float((vx * dx + vy * dy) / (speed * dist))


_state: dict[str, dict] = {}


def _parking_state(parking_id: str, occupancy: dict[str, bool]) -> dict:
    return _state.setdefault(
        parking_id,
        {
            "tracks": [],
            "next_id": 1,
            "occupancy": dict(occupancy),
            "occluded": set(),
            "events": [],
            "stalls": {},
        },
    )


def apply_temporal(
    parking_id: str,
    spaces: list[dict],
    vehicles: list[dict],
    occupancy: dict[str, bool],
    frame_shape: tuple[int, ...],
    scores: dict | None = None,
) -> tuple[dict[str, bool], list[str], list[str]]:
    """Keep hidden stalls stable and free them when a car leaves from behind a large vehicle."""
    height, width = frame_shape[:2]
    now = time.time()
    state = _parking_state(parking_id, occupancy)
    state["prev_occupancy"] = dict(state.get("occupancy") or occupancy)
    previous = dict(state.get("prev_occupancy") or occupancy)
    previous_occluded = set(state.get("occluded") or [])
    scores = scores or {}

    used: set[int] = set()
    tracks: list[dict] = []
    disappeared: list[dict] = []
    for track in state["tracks"]:
        best_score = 0.0
        best_index = None
        for index, vehicle in enumerate(vehicles):
            if index in used:
                continue
            score = _iou(track["bbox"], vehicle_bbox(vehicle))
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= 0.22:
            vehicle = vehicles[best_index]
            used.add(best_index)
            bbox = vehicle_bbox(vehicle)
            cx, cy = _center(bbox)
            history = list(track.get("history") or [])
            history.append((cx, cy, now))
            track.update(
                {
                    "bbox": bbox,
                    "large": bool(vehicle.get("large")),
                    "cls": vehicle.get("cls"),
                    "missing": 0,
                    "history": history[-20:],
                }
            )
            if len(history) >= 3:
                if cy > history[0][1] + 28:
                    track["moved_out"] = True
                if bbox[0] < 12 or bbox[2] > width - 12 or bbox[3] > height - 12:
                    track["moved_out"] = True
            tracks.append(track)
        else:
            track["missing"] = int(track.get("missing") or 0) + 1
            if track["missing"] >= 3:
                disappeared.append(track)
            else:
                tracks.append(track)

    for index, vehicle in enumerate(vehicles):
        if index in used:
            continue
        bbox = vehicle_bbox(vehicle)
        cx, cy = _center(bbox)
        track = {
            "id": state["next_id"],
            "bbox": bbox,
            "large": bool(vehicle.get("large")),
            "cls": vehicle.get("cls"),
            "missing": 0,
            "new": True,
            "history": [(cx, cy, now)],
            "emerged_from_occlusion": False,
            "moved_out": False,
        }
        state["next_id"] += 1
        tracks.append(track)

    space_meta = []
    for space in spaces:
        pts = space_points(space, width, height)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        space_meta.append((space["id"], cx, cy))

    occluded: set[str] = set()
    large_vehicles = [v for v in vehicles if v.get("large")]
    for vehicle in large_vehicles:
        bbox = vehicle_bbox(vehicle)
        x1, y1, x2, y2 = bbox
        for space_id, cx, cy in space_meta:
            aligned = x1 - 0.04 * width <= cx <= x2 + 0.04 * width
            if not aligned:
                continue
            space = next(row for row in spaces if row["id"] == space_id)
            score, ground_hit = occupancy_from_vehicles(space, [bbox], frame_shape)
            if ground_hit and score >= 0.28:
                occupancy[space_id] = True
            behind = cy < y1 + 0.38 * (y2 - y1)
            sitting_here = ground_hit or score >= 0.45
            if behind and aligned and not sitting_here:
                occluded.add(space_id)

    for space_id in occluded:
        if space_id in previous:
            occupancy[space_id] = bool(previous[space_id])

    events: list[str] = []
    for track in tracks:
        if not track.get("new"):
            continue
        cx, cy = _center(track["bbox"])
        for vehicle in large_vehicles:
            vx1, vy1, vx2, vy2 = vehicle_bbox(vehicle)
            near_top = vy1 - 50 <= track["bbox"][3] <= vy1 + 0.45 * (vy2 - vy1)
            aligned = vx1 - 20 <= cx <= vx2 + 20
            if near_top and aligned:
                track["emerged_from_occlusion"] = True

    def _free_hidden(track: dict) -> None:
        if track.get("freed_hidden"):
            return
        if not (track.get("emerged_from_occlusion") or track.get("moved_out")):
            return
        tx, _ty = _center(track["bbox"])
        candidates = []
        hidden = occluded | previous_occluded
        for space_id, cx, cy in space_meta:
            if space_id not in hidden:
                continue
            if not occupancy.get(space_id):
                continue
            if abs(cx - tx) <= width * 0.14:
                candidates.append((cy, space_id))
        if not candidates:
            return
        candidates.sort()
        freed = candidates[0][1]
        occupancy[freed] = False
        events.append(freed)
        track["freed_hidden"] = True

    for track in disappeared:
        _free_hidden(track)
    for track in tracks:
        if track.get("emerged_from_occlusion") and track.get("moved_out"):
            _free_hidden(track)
            track["emerged_from_occlusion"] = False
        track.pop("new", None)

    state["tracks"] = tracks
    state["lost"] = disappeared
    state["occupancy"] = dict(occupancy)
    state["occluded"] = occluded
    state["events"] = events
    return occupancy, sorted(occluded), events


def _blank_flow() -> dict:
    return {
        "phase": "empty",
        "overlaps": [],
        "enter_n": 0,
        "leave_n": 0,
        "event": "none",
        "track_id": None,
    }


def apply_stall_flow(
    parking_id: str,
    spaces: list[dict],
    vehicles: list[dict],
    occupancy: dict[str, bool],
    frame_shape: tuple[int, ...],
    scores: dict | None = None,
) -> tuple[dict[str, bool], dict[str, str]]:
    """Confirm occupy/free using entry and exit motion, not a single snapshot.

    Reuses YOLO boxes already tracked in apply_temporal. No extra model.
    """
    height, width = frame_shape[:2]
    scores = scores or {}
    state = _parking_state(parking_id, occupancy)
    stalls: dict[str, dict] = state.setdefault("stalls", {})
    tracks = list(state.get("tracks") or [])
    lost = list(state.get("lost") or [])
    published = dict(state.get("prev_occupancy") or occupancy)

    centers: dict[str, tuple[float, float]] = {}
    for space in spaces:
        pts = space_points(space, width, height)
        centers[space["id"]] = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

    stall_of_track: dict[int, tuple[str, float, bool]] = {}
    for track in tracks:
        bbox = track.get("bbox")
        if not bbox:
            continue
        best_id = None
        best_score = 0.0
        best_ground = False
        for space in spaces:
            score, ground = occupancy_from_vehicles(space, [bbox], frame_shape)
            if score > best_score:
                best_id = space["id"]
                best_score = score
                best_ground = ground
        if best_id and (best_score >= 0.06 or best_ground):
            stall_of_track[int(track["id"])] = (best_id, best_score, best_ground)
            track["stall"] = best_id

    lost_by_stall: dict[str, dict] = {}
    for track in lost:
        sid = track.get("stall")
        if sid:
            lost_by_stall[sid] = track

    events: dict[str, str] = {}
    for space in spaces:
        space_id = space["id"]
        row = stalls.setdefault(space_id, _blank_flow())
        info = scores.get(space_id) or {}
        overlap = float(info.get("overlap") or info.get("body_overlap") or 0.0)
        body = float(info.get("body_overlap") or 0.0)
        ground = bool(info.get("ground_hit"))
        empty_score = info.get("empty_score")
        snapshot = bool(occupancy.get(space_id))
        was = bool(published.get(space_id, False))
        hist = list(row.get("overlaps") or [])
        hist.append(overlap)
        row["overlaps"] = hist[-8:]
        prev_overlap = hist[-2] if len(hist) >= 2 else overlap
        delta = overlap - prev_overlap

        assigned = None
        assigned_score = 0.0
        approach = 0.0
        for track in tracks:
            mapped = stall_of_track.get(int(track["id"]))
            if not mapped or mapped[0] != space_id:
                continue
            if mapped[1] >= assigned_score:
                assigned = track
                assigned_score = mapped[1]
        if assigned:
            cx, cy = centers.get(space_id, (0.0, 0.0))
            approach = _approach(assigned.get("history") or [], cx, cy)
            row["track_id"] = assigned.get("id")
        else:
            row["track_id"] = None

        entering = (
            (delta >= 0.06 and overlap >= 0.08)
            or (approach >= 0.35 and overlap >= 0.07)
            or (assigned is not None and overlap >= 0.12 and not was and (ground or body >= 0.10))
        )
        no_car = (not ground) and assigned is None and overlap < 0.12 and body < 0.12
        weak_car = (not ground) and overlap < 0.16 and body < 0.14
        leaving = (
            (delta <= -0.05 and was)
            or (approach <= -0.28 and was)
            or (space_id in lost_by_stall and was)
            or (was and no_car)
            or (was and not snapshot and weak_car)
        )
        clearly_empty = no_car and (
            empty_score is None or float(empty_score) >= 0.62 or overlap < 0.06
        )
        clearly_parked = ground or overlap >= 0.22 or body >= 0.16

        phase = str(row.get("phase") or "empty")
        event = "none"
        decision = snapshot

        if was:
            if leaving or clearly_empty:
                row["leave_n"] = int(row.get("leave_n") or 0) + 1
                row["enter_n"] = 0
                phase = "leaving"
                event = "leaving"
            elif clearly_parked:
                row["leave_n"] = max(0, int(row.get("leave_n") or 0) - 1)
                phase = "occupied"
                event = "parked"
            need_leave = 1 if (clearly_empty or no_car) else 2
            if int(row.get("leave_n") or 0) >= need_leave or clearly_empty:
                decision = False
                phase = "empty"
                event = "gone"
            elif snapshot and clearly_parked:
                decision = True
            elif not snapshot and clearly_parked:
                decision = True
                event = "false_free"
                phase = "occupied"
            elif not snapshot:
                decision = False
                phase = "leaving" if leaving else "empty"
                event = "leaving" if leaving else "gone"
            else:
                decision = True
        else:
            if entering:
                row["enter_n"] = int(row.get("enter_n") or 0) + 1
                row["leave_n"] = 0
                phase = "entering"
                event = "entering"
            else:
                row["enter_n"] = 0
            if int(row.get("enter_n") or 0) >= 2 or (entering and clearly_parked):
                decision = True
                phase = "occupied"
                event = "parked"
            elif snapshot and not entering and not clearly_parked and overlap < 0.18:
                decision = False
                event = "false_occupy"
                phase = "empty"
            elif not snapshot:
                decision = False
                phase = "empty"

        occupancy[space_id] = bool(decision)
        row["phase"] = phase
        row["event"] = event
        events[space_id] = event
        info["flow"] = event
        info["flow_phase"] = phase
        info["flow_delta"] = round(delta, 4)
        scores[space_id] = info

    state["stalls"] = stalls
    state["occupancy"] = dict(occupancy)
    state["lost"] = []
    state["flow_events"] = events
    return occupancy, events
