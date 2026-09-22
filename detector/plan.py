from __future__ import annotations

from occupancy import _read_json, _write_json, load_spaces, profile_dir


def plan_path(parking_id: str):
    return profile_dir(parking_id) / "plan.json"


def _rect(x: float, y: float, w: float, h: float) -> list[dict]:
    return [
        {"x": x, "y": y},
        {"x": x + w, "y": y},
        {"x": x + w, "y": y + h},
        {"x": x, "y": y + h},
    ]


AISLE = "~"


def _is_space(value: str) -> bool:
    return bool(value) and value != AISLE


def _clean_cells(rows: int, cols: int, raw) -> list[list[str]]:
    rows = max(1, min(30, int(rows or 1)))
    cols = max(1, min(30, int(cols or 1)))
    source = raw if isinstance(raw, list) else []
    cells: list[list[str]] = []
    seen: set[str] = set()
    for r in range(rows):
        line = source[r] if r < len(source) and isinstance(source[r], list) else []
        row: list[str] = []
        for c in range(cols):
            value = str(line[c] if c < len(line) else "").strip()
            if value in {AISLE, "-", "aisle", "pasabidea"}:
                value = AISLE
            elif value in {".", "·"}:
                value = ""
            if _is_space(value) and value in seen:
                value = ""
            if _is_space(value):
                seen.add(value)
            row.append(value)
        cells.append(row)
    return cells


def matrix_to_stalls(rows: int, cols: int, cells: list[list[str]]) -> list[dict]:
    stalls = []
    if rows <= 0 or cols <= 0:
        return stalls
    cell_w = 0.88 / cols
    cell_h = 0.88 / rows
    pad_x = cell_w * 0.08
    pad_y = cell_h * 0.08
    for r, row in enumerate(cells):
        for c, space_id in enumerate(row):
            if not _is_space(space_id):
                continue
            stalls.append(
                {
                    "id": space_id,
                    "points": _rect(
                        0.06 + c * cell_w + pad_x,
                        0.06 + r * cell_h + pad_y,
                        cell_w - 2 * pad_x,
                        cell_h - 2 * pad_y,
                    ),
                }
            )
    return stalls


def empty_plan() -> dict:
    return {
        "kind": "matrix",
        "rows": 0,
        "cols": 0,
        "cells": [],
        "stalls": [],
        "defined": False,
        "seeded": False,
    }


def load_plan(parking_id: str) -> dict:
    data = _read_json(plan_path(parking_id), {})
    rows = int(data.get("rows") or 0)
    cols = int(data.get("cols") or 0)
    cells = data.get("cells")
    if rows > 0 and cols > 0 and isinstance(cells, list):
        cleaned = _clean_cells(rows, cols, cells)
        return {
            "kind": "matrix",
            "rows": rows,
            "cols": cols,
            "cells": cleaned,
            "stalls": matrix_to_stalls(rows, cols, cleaned),
            "defined": True,
            "seeded": False,
        }
    stalls = data.get("stalls") or []
    if stalls:
        return {
            "kind": "polygon",
            "rows": 0,
            "cols": 0,
            "cells": [],
            "outline": data.get("outline") or [],
            "stalls": stalls,
            "defined": True,
            "seeded": False,
        }
    return empty_plan()


def save_plan(parking_id: str, payload: dict) -> dict:
    rows = int(payload.get("rows") or 0)
    cols = int(payload.get("cols") or 0)
    if rows <= 0 or cols <= 0:
        data = empty_plan()
        plan_path(parking_id).unlink(missing_ok=True)
        return data
    cells = _clean_cells(rows, cols, payload.get("cells"))
    known = {space["id"] for space in load_spaces(parking_id)}
    if known:
        cells = [
            [cell if (not _is_space(cell) or cell in known) else "" for cell in row]
            for row in cells
        ]
    data = {
        "kind": "matrix",
        "rows": len(cells),
        "cols": cols,
        "cells": cells,
        "stalls": matrix_to_stalls(len(cells), cols, cells),
        "defined": True,
        "seeded": False,
    }
    _write_json(plan_path(parking_id), data)
    return data
