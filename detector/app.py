from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import cv2
import uvicorn
import audit
import auth
import credentials
import mailer
from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from camera import grab_frame
from dataset import (
    dataset_stats,
    load_frame_image,
    pending_frames,
    public_frame,
    purge_expired_frames,
    reliability_report,
    reliability_summary,
    review_frame,
    save_captured_frame,
    save_live_correction,
    too_similar,
    training_queue,
)

from occupancy import (
    HAS_YOLO,
    analyze_frame,
    describe_correction_effect,
    detect_vehicles,
    ensure_display_token,
    load_config,
    load_meta,
    load_spaces,
    occupancy_from_vehicles,
    parking_id_for_display_token,
    predict_occupancy,
    profile_dir,
    refresh_sample_counts,
    remember_correction,
    rotate_display_token,
    save_config,
    save_spaces,
    spaces_covered_by_large,
    train_model,
    training_gap,
    warmup_yolo,
)
from sampler import buffer_status, maybe_sample
from plan import load_plan, save_plan

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
if CONFIG_PATH.exists():
    APP_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
else:
    APP_CONFIG = {"host": "0.0.0.0", "port": 8000, "interval_seconds": 0.5}

auth.ensure_bootstrap(APP_CONFIG)
_secrets_removed = False
for key in ("admin_password", "admin_pin"):
    if key in APP_CONFIG:
        APP_CONFIG.pop(key, None)
        _secrets_removed = True
if "host" not in APP_CONFIG or APP_CONFIG.get("host") == "0.0.0.0":
    APP_CONFIG["host"] = "127.0.0.1"
    _secrets_removed = True
if "allowed_hosts" not in APP_CONFIG:
    APP_CONFIG["allowed_hosts"] = ["localhost", "127.0.0.1"]
    _secrets_removed = True
if _secrets_removed and CONFIG_PATH.exists():
    CONFIG_PATH.write_text(json.dumps(APP_CONFIG, indent=2) + "\n", encoding="utf-8")

_data_root = ROOT / "data"
if _data_root.exists():
    for path in _data_root.iterdir():
        if path.is_dir() and not path.name.startswith("."):
            credentials.migrate_parking_config(path.name)

state_lock = threading.Lock()
last_frames: dict[str, object] = {}
last_annotated: dict[str, object] = {}
last_status: dict[str, dict] = {}
capture_jobs: dict[str, dict] = {}
_last_capture_image: dict[str, object] = {}
_display_hits: dict[str, list[float]] = {}
_register_hits: dict[str, list[float]] = {}

app = FastAPI(title="ParkingAI", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=APP_CONFIG.get("cors_origins") or ["http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1:8080", "http://localhost:8080"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=list(APP_CONFIG.get("allowed_hosts") or ["localhost", "127.0.0.1"]),
)
STATIC_DIR = ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=STATIC_DIR), name="ui")


def _page(name: str, missing: str):
    page = STATIC_DIR / name
    if not page.exists():
        raise HTTPException(404, missing)
    return FileResponse(page, headers={"Cache-Control": "no-store"})


@app.middleware("http")
async def security_headers(request: Request, call_next):
    path = request.url.path
    token = ""
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        token = header.split(" ", 1)[1].strip()
    token = token or request.cookies.get(auth.SESSION_COOKIE) or ""
    user = auth.user_from_token(token)
    if user and (user.get("must_change_password") or user.get("must_set_email") or user.get("must_verify_email")):
        allowed = (
            path in {
                "/",
                "/health",
                "/auth/login",
                "/auth/logout",
                "/auth/me",
                "/auth/password",
                "/auth/register",
                "/auth/forgot",
                "/auth/reset",
                "/auth/invite",
                "/auth/verify",
                "/auth/resend",
                "/auth/email",
                "/berreskuratu",
                "/pantaila",
            }
            or path.startswith("/ui/")
            or path.startswith("/display/")
        )
        if not allowed:
            return JSONResponse(
                {
                    "detail": "Lehenik kontua osatu behar duzu",
                    "must_change_password": bool(user.get("must_change_password")),
                    "must_set_email": bool(user.get("must_set_email")),
                    "must_verify_email": bool(user.get("must_verify_email")),
                },
                status_code=403,
            )
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    )
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if request.url.path.startswith("/ui/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


class RegisterBody(BaseModel):
    parking_id: str
    camera_url: str = ""
    space_ids: list[str] = Field(default_factory=list)
    camera_user: str = ""
    camera_pass: str = ""
    name: str = ""


class SpacesBody(BaseModel):
    parking_id: str
    spaces: list[dict]


class LabelBody(BaseModel):
    parking_id: str
    space_id: str
    occupied: bool
    large_vehicle: bool = False


class TrainBody(BaseModel):
    parking_id: str


class CaptureBody(BaseModel):
    parking_id: str
    count: int = 50
    interval_seconds: float = 4.0


class PlanBody(BaseModel):
    parking_id: str
    rows: int = 0
    cols: int = 0
    cells: list[list[str]] = Field(default_factory=list)
    outline: list[dict] = Field(default_factory=list)
    stalls: list[dict] = Field(default_factory=list)


class PinBody(BaseModel):
    pin: str


class ReviewBody(BaseModel):
    parking_id: str
    frame_id: str
    spaces: dict[str, bool] = Field(default_factory=dict)


class GrabBody(BaseModel):
    parking_id: str


class LoginBody(BaseModel):
    username: str
    password: str


class PasswordBody(BaseModel):
    current_password: str
    new_password: str


class ForgotBody(BaseModel):
    identity: str
    email: str = ""


class ResetBody(BaseModel):
    token: str
    new_password: str


class TokenBody(BaseModel):
    token: str


class EmailBody(BaseModel):
    email: str


class SignupBody(BaseModel):
    email: str
    password: str
    name: str = ""
    username: str = ""


class CreateUserBody(BaseModel):
    username: str
    email: str
    name: str = ""
    role: str = "user"
    password: str = ""


class CreateParkingBody(BaseModel):
    name: str
    parking_id: str = ""
    camera_url: str = ""
    visibility: str = "public"


class VisibilityBody(BaseModel):
    parking_id: str
    visibility: str


class PrivilegeBody(BaseModel):
    parking_id: str
    username: str
    privilege: str


class FavoriteBody(BaseModel):
    parking_id: str
    favorite: bool


def _try_train(parking_id: str) -> dict:
    gap = training_gap(parking_id)
    try:
        meta = train_model(parking_id)
        return {**meta, **gap, "need_free": 0, "need_occupied": 0, "just_trained": True, "train_hint": None}
    except ValueError as exc:
        counts = refresh_sample_counts(parking_id)
        return {**counts, **gap, "just_trained": False, "train_hint": str(exc)}


def _set_session_cookie(request: Request, response: Response, token: str) -> None:
    response.set_cookie(
        auth.SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        max_age=int(auth.SESSION_TTL),
        path="/",
    )


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else ""


def _is_loopback(request: Request) -> bool:
    host = _client_ip(request)
    return host in {"127.0.0.1", "::1", "localhost"}


def _public_base(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _attach_mail(payload: dict, request: Request, mail: dict, link: str) -> dict:
    if mail.get("sent"):
        payload["delivery"] = "smtp"
        return payload
    payload["delivery"] = "outbox"
    payload["message"] = (
        "Korreoa ez da bidali: SMTP ez dago konfiguratuta. "
        "Gmail-era ez da joango. Ireki esteka hau ordenagailu honetan."
    )
    if _is_loopback(request) and link:
        payload["preview"] = link
    if mail.get("error"):
        payload["mail_error"] = mail["error"]
    return payload


def _account_mail(kind: str, to: str, link: str) -> dict:
    subjects = {
        "reset": "ParkingAI: pasahitza berrezarri",
        "invite": "ParkingAI: kontua osatu",
        "verify": "ParkingAI: korreoa berretsi",
    }
    bodies = {
        "reset": (
            "Pasahitza berrezartzeko esteka, 30 minutuz balio duena:\n"
            f"{link}\n\nEz bazara izan, ez ikusi mezua."
        ),
        "invite": (
            "ParkingAI kontua osatzeko esteka, 48 orduz balio duena:\n"
            f"{link}\n\nPasahitzak gutxienez 12 karaktere, maiuskula, minuskula eta zenbaki bat behar ditu."
        ),
        "verify": f"Korreoa berresteko esteka, 24 orduz balio duena:\n{link}\n",
    }
    return mailer.send_mail(to, subjects[kind], bodies[kind])


def _rate_ok(bucket: dict[str, list[float]], key: str, *, limit: int, window: float) -> bool:
    now = time.time()
    recent = [stamp for stamp in bucket.get(key, []) if now - stamp < window]
    if len(recent) >= limit:
        bucket[key] = recent
        return False
    recent.append(now)
    bucket[key] = recent
    return True


def _grab(parking_id: str):
    config = load_config(parking_id)
    user, password = credentials.camera_login(parking_id)
    return grab_frame(config.get("camera_url", ""), user, password)


def _display_payload(parking_id: str) -> dict:
    config = load_config(parking_id)
    spaces = load_spaces(parking_id)
    plan = load_plan(parking_id)
    with state_lock:
        status = dict(last_status.get(parking_id) or {})
    occupancy = dict(status.get("spaces") or {})
    ids = [space["id"] for space in spaces]
    if not ids:
        ids = list(occupancy)
    free = sum(1 for space_id in ids if occupancy.get(space_id) is False)
    busy = sum(1 for space_id in ids if occupancy.get(space_id) is True)
    running = bool(status.get("camera_ok") and status.get("ok", True))
    public_spaces = {space_id: bool(occupancy[space_id]) for space_id in ids if space_id in occupancy}
    return {
        "name": config.get("name") or parking_id,
        "ok": running,
        "free": free,
        "occupied": busy,
        "total": len(ids),
        "spaces": public_spaces,
        "occluded": status.get("occluded") or [],
        "updated_at": status.get("updated_at"),
        "plan": {
            "rows": plan.get("rows") or 0,
            "cols": plan.get("cols") or 0,
            "cells": plan.get("cells") or [],
            "defined": bool(plan.get("defined")),
        },
        "error": None if running else "Sistema ez dago martxan.",
    }


def _jpeg(frame) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame.copy(), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        raise HTTPException(500, "JPEG kodeketak huts egin du")
    return encoded.tobytes()


def get_current_user(
    authorization: str | None = Header(default=None),
    parkingai_session: str | None = Cookie(default=None),
) -> dict:
    raw = ""
    if authorization and authorization.lower().startswith("bearer "):
        raw = authorization.split(" ", 1)[1].strip()
    raw = raw or parkingai_session or ""
    user = auth.user_from_token(raw)
    if not user:
        raise HTTPException(401, "Saioa hasi behar da")
    return user


def _parking(parking_id: str) -> str:
    try:
        return auth.safe_parking_id(parking_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _require_view(user: dict, parking_id: str) -> str:
    parking_id = _parking(parking_id)
    if not auth.can_view(user, parking_id):
        raise HTTPException(403, "Aparkaleku hau ez dago zuretzat")
    return parking_id


def _require_edit(user: dict, parking_id: str) -> str:
    parking_id = _parking(parking_id)
    if not auth.can_edit(user, parking_id):
        raise HTTPException(403, "Ez duzu aparkaleku hau aldatzeko baimenik")
    return parking_id


def _require_camera(user: dict, parking_id: str) -> str:
    parking_id = _parking(parking_id)
    if not auth.can_view_camera(user, parking_id):
        raise HTTPException(403, "Kameraren irudia administratzaileentzat da")
    return parking_id


def _image_response(content: bytes) -> Response:
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, private", "Pragma": "no-cache"},
    )


def _public_status(payload: dict) -> dict:
    running = bool(payload.get("camera_ok") and payload.get("ok", True))
    return {
        "ok": bool(payload.get("ok")),
        "spaces": payload.get("spaces") or {},
        "vehicles": int(payload.get("vehicles") or 0),
        "occluded": payload.get("occluded") or [],
        "camera_ok": bool(payload.get("camera_ok")),
        "calibrated": bool(payload.get("calibrated")),
        "updated_at": payload.get("updated_at"),
        "error": None if running else "Sistema ez dago martxan.",
    }


def _require_users(user: dict, parking_id: str) -> None:
    if not auth.can_manage_users(user, parking_id):
        raise HTTPException(403, "Ez duzu erabiltzaileak kudeatzeko baimenik")


def _slug(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in (name or "").strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "parking"


def _parking_record(parking_id: str, user: dict) -> dict:
    config = load_config(parking_id)
    spaces = load_spaces(parking_id)
    meta = load_meta(parking_id)
    with state_lock:
        status = dict(last_status.get(parking_id) or {})
    occupancy = dict(status.get("spaces") or {})
    free = sum(1 for space in spaces if occupancy.get(space["id"]) is False)
    busy = sum(1 for space in spaces if occupancy.get(space["id"]) is True)
    can_edit = auth.can_edit(user, parking_id)
    record = {
        "id": parking_id,
        "name": config.get("name") or parking_id,
        "camera_url": (config.get("camera_url") or "") if can_edit else "",
        "owner_id": config.get("owner_id"),
        "member_ids": config.get("member_ids") or [] if can_edit else [],
        "spaces": spaces if can_edit else [],
        "space_count": len(spaces),
        "occupancy": occupancy,
        "free_spaces": free,
        "occupied_spaces": busy,
        "model_trained": bool(meta.get("approved_training") and meta.get("model_trained")),
        "samples_free": meta.get("samples_free", 0) if can_edit else 0,
        "samples_occupied": meta.get("samples_occupied", 0) if can_edit else 0,
        "can_edit": can_edit,
        "can_view_camera": auth.can_view_camera(user, parking_id),
        "can_manage_users": auth.can_manage_users(user, parking_id),
        "privilege": auth.privilege_of(user, parking_id),
        "visibility": auth.parking_visibility(parking_id),
        "assigned": auth.is_assigned(user, parking_id),
        "favorite": auth.is_favorite(user, parking_id),
        "acl": auth.parking_acl(parking_id) if auth.can_manage_users(user, parking_id) else {},
        "camera_ok": status.get("camera_ok"),
        "error": status.get("error"),
        "occluded": status.get("occluded") or [],
    }
    if can_edit:
        record.update(reliability_summary(parking_id))
    return record


def _active_parking_ids() -> list[str]:
    data_root = ROOT / "data"
    if not data_root.exists():
        return []
    skip = {"outbox"}
    ids = []
    for path in data_root.iterdir():
        if not path.is_dir() or path.name.startswith(".") or path.name in skip:
            continue
        if not (path / "config.json").exists():
            continue
        ids.append(path.name)
    return ids


def _capture_parking(parking_id: str) -> None:
    frame = _grab(parking_id)
    if frame is None:
        with state_lock:
            status = dict(last_status.get(parking_id) or {})
            status["camera_ok"] = False
            status["error"] = "Kamera irudirik ez. Egiaztatu URL-a eta sarea."
            last_status[parking_id] = status
        return
    with state_lock:
        last_frames[parking_id] = frame


def _analyze_parking(parking_id: str) -> None:
    with state_lock:
        cached = last_frames.get(parking_id)
        frame = None if cached is None else cached.copy()
    if frame is None:
        return
    status, annotated = analyze_frame(parking_id, frame)
    maybe_sample(parking_id, frame, occupancy=status.get("spaces") or {})
    with state_lock:
        last_annotated[parking_id] = annotated
        last_status[parking_id] = status


def _capture_loop() -> None:
    interval = max(0.35, float(APP_CONFIG.get("interval_seconds", 0.5)))
    while True:
        started = time.perf_counter()
        for parking_id in _active_parking_ids():
            try:
                _capture_parking(parking_id)
            except Exception as exc:
                with state_lock:
                    status = dict(last_status.get(parking_id) or {})
                    status["camera_ok"] = False
                    status["error"] = str(exc)
                    last_status[parking_id] = status
        remaining = interval - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)


def _analyze_loop() -> None:
    cycles = 0
    while True:
        ids = _active_parking_ids()
        if not ids:
            time.sleep(0.4)
            continue
        for parking_id in ids:
            try:
                _analyze_parking(parking_id)
                cycles += 1
                if cycles % 8 == 0:
                    purge_expired_frames(parking_id)
            except Exception as exc:
                with state_lock:
                    last_status[parking_id] = {
                        "ok": False,
                        "spaces": (last_status.get(parking_id) or {}).get("spaces") or {},
                        "vehicles": 0,
                        "camera_ok": False,
                        "calibrated": False,
                        "model_trained": False,
                        "samples_free": 0,
                        "samples_occupied": 0,
                        "updated_at": None,
                        "error": str(exc),
                    }
        time.sleep(0.02)


def _loop() -> None:
    _capture_loop()


@app.on_event("startup")
def startup() -> None:
    print("ParkingAI detektagailua: http://127.0.0.1:%s" % APP_CONFIG.get("port", 8000))
    print("Menua: http://127.0.0.1:%s" % APP_CONFIG.get("port", 8000))
    print("Konfigurazioa: http://127.0.0.1:%s/konfigurazioa" % APP_CONFIG.get("port", 8000))
    print("Egoera: http://127.0.0.1:%s/egoera" % APP_CONFIG.get("port", 8000))
    print("Entrenamendua: http://127.0.0.1:%s/entrenatu" % APP_CONFIG.get("port", 8000))
    print(warmup_yolo())
    if not HAS_YOLO:
        print("Kotxe-detekzioa desgaituta dago YOLO gabe.")
    threading.Thread(target=_capture_loop, daemon=True).start()
    threading.Thread(target=_analyze_loop, daemon=True).start()


@app.get("/")
def home():
    return _page("menu.html", "Menua falta da")


@app.get("/aparkalekuak")
def all_parkings_page():
    return _page("menu.html", "Menua falta da")


@app.get("/konfigurazioa")
def konfigurazioa():
    return _page("index.html", "Interfazea falta da")


@app.get("/egoera")
def egoera():
    return _page("status.html", "Egoera orria falta da")


@app.get("/fidagarritasuna")
def fidagarritasuna():
    return _page("fidagarritasuna.html", "Fidagarritasun orria falta da")


@app.get("/pantaila")
def pantaila():
    return _page("pantaila.html", "Pantaila falta da")


@app.get("/berreskuratu")
def berreskuratu():
    return _page("menu.html", "Menua falta da")


@app.post("/auth/login")
def auth_login(body: LoginBody, request: Request, response: Response):
    ip = _client_ip(request)
    try:
        data = auth.login(body.username, body.password, client_id=ip)
    except ValueError as exc:
        msg = str(exc)
        audit.record("login", actor=body.username, ip=ip, ok=False, detail=msg)
        if "korreo" in msg.lower() or "gonbidapena" in msg.lower():
            raise HTTPException(403, msg) from exc
        raise HTTPException(401, msg) from exc
    audit.record("login", actor=data["user"].get("username"), ip=ip, ok=True)
    _set_session_cookie(request, response, data["token"])
    return data


@app.post("/auth/register")
def auth_register(body: SignupBody, request: Request):
    ip = _client_ip(request)
    if not _rate_ok(_register_hits, ip, limit=5, window=3600.0):
        raise HTTPException(429, "Saiakera gehiegi. Itxaron pixka bat.")
    try:
        created, raw = auth.register_account(body.email, body.password, body.name, body.username)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    link = _public_base(request) + "/berreskuratu?t=" + raw + "&m=verify"
    mail = _account_mail("verify", body.email, link)
    audit.record("register", actor=created.get("username"), ip=ip, ok=True)
    payload = {
        "ok": True,
        "user": created,
        "message": "Berrespen-mezua bidali dugu. Begiratu korreoa saioa hasi aurretik.",
    }
    return _attach_mail(payload, request, mail, link)


@app.post("/auth/logout")
def auth_logout(
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
    parkingai_session: str | None = Cookie(default=None),
):
    raw = ""
    if authorization and authorization.lower().startswith("bearer "):
        raw = authorization.split(" ", 1)[1].strip()
    token = raw or parkingai_session or ""
    actor = (auth.user_from_token(token) or {}).get("username") or "-"
    auth.logout(token)
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    audit.record("logout", actor=actor, ip=_client_ip(request), ok=True)
    return {"ok": True}


@app.post("/auth/password")
def auth_password(
    body: PasswordBody,
    request: Request,
    response: Response,
    user: dict = Depends(get_current_user),
):
    try:
        data = auth.change_password(user["username"], body.current_password, body.new_password)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    audit.record("password_change", actor=user["username"], ip=_client_ip(request), ok=True)
    _set_session_cookie(request, response, data["token"])
    return {"ok": True, "user": data["user"]}


@app.post("/auth/forgot")
def auth_forgot(body: ForgotBody, request: Request):
    local = _is_loopback(request)
    result = auth.request_password_reset(body.identity, body.email, allow_bind=local)
    payload = {
        "ok": True,
        "message": "Korreoa existitzen bada, mezua bidali dugu. Begiratu sarrera-ontzia eta spam.",
    }
    if result:
        link = _public_base(request) + "/berreskuratu?t=" + result["token"]
        mail = _account_mail("reset", result["email"], link)
        audit.record(
            "password_reset_request",
            actor=result["user"].get("username"),
            ip=_client_ip(request),
            ok=True,
        )
        _attach_mail(payload, request, mail, link)
    return payload


@app.get("/auth/token")
def auth_token_peek(t: str = "", m: str = "reset"):
    purpose = {"invite": "invite", "verify": "verify"}.get(m, "reset")
    user = auth.peek_account_token(t, purpose)
    if user is None:
        raise HTTPException(400, "Esteka baliogabea edo iraungia")
    return {"user": {"username": user.get("username"), "name": user.get("name"), "email": user.get("email")}}


@app.post("/auth/reset")
def auth_reset(body: ResetBody, request: Request, response: Response):
    try:
        data = auth.complete_password_reset(body.token, body.new_password)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    audit.record("password_reset", actor=data["user"].get("username"), ip=_client_ip(request), ok=True)
    _set_session_cookie(request, response, data["token"])
    return {"ok": True, "user": data["user"]}


@app.post("/auth/invite")
def auth_invite_complete(body: ResetBody, request: Request, response: Response):
    try:
        data = auth.complete_invite(body.token, body.new_password)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    audit.record("invite_complete", actor=data["user"].get("username"), ip=_client_ip(request), ok=True)
    _set_session_cookie(request, response, data["token"])
    return {"ok": True, "user": data["user"]}


@app.post("/auth/verify")
def auth_verify(body: TokenBody, request: Request, response: Response):
    try:
        data = auth.complete_email_verification(body.token)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    audit.record("email_verify", actor=data["user"].get("username"), ip=_client_ip(request), ok=True)
    _set_session_cookie(request, response, data["token"])
    return {"ok": True, "user": data["user"]}


@app.post("/auth/resend")
def auth_resend(body: ForgotBody, request: Request):
    user = auth.find_identity(body.identity) or auth.find_user_by_email(body.email)
    payload = {
        "ok": True,
        "message": "Korreoa existitzen bada eta berretsi gabe badago, mezua bidali dugu.",
    }
    if user and user.get("email") and not user.get("email_verified"):
        result = auth.request_email_verification(user["username"])
        if result:
            link = _public_base(request) + "/berreskuratu?t=" + result["token"] + "&m=verify"
            mail = _account_mail("verify", result["email"], link)
            _attach_mail(payload, request, mail, link)
    return payload


@app.post("/auth/email")
def auth_set_email(
    body: EmailBody,
    request: Request,
    user: dict = Depends(get_current_user),
):
    try:
        result = auth.set_account_email(user["username"], body.email)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    link = _public_base(request) + "/berreskuratu?t=" + result["token"] + "&m=verify"
    mail = _account_mail("verify", result["email"], link)
    audit.record("email_set", actor=user["username"], ip=_client_ip(request), ok=True)
    payload = {"ok": True, "user": result["user"], "message": "Berrespen-mezua bidali dugu."}
    return _attach_mail(payload, request, mail, link)


@app.get("/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    return {"user": user}


@app.get("/users")
def users_list(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(403, "Administratzailea bakarrik")
    return {"users": auth.list_users()}


@app.post("/users")
def users_create(
    body: CreateUserBody,
    request: Request,
    user: dict = Depends(get_current_user),
):
    if user.get("role") != "admin":
        raise HTTPException(403, "Administratzailea bakarrik")
    if body.role == "admin" and user.get("role") != "admin":
        raise HTTPException(403, "Adminik ezin da sortu")
    try:
        created, raw = auth.invite_user(body.username, body.email, body.name, body.role)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    link = _public_base(request) + "/berreskuratu?t=" + raw + "&m=invite"
    mail = _account_mail("invite", body.email, link)
    audit.record("user_invite", actor=user["username"], detail=body.username, ok=True)
    payload = {
        "ok": True,
        "user": created,
        "message": "Gonbidapena korreoz bidali da. Erabiltzaileak pasahitza han sortuko du.",
    }
    return _attach_mail(payload, request, mail, link)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/entrenatu")
def entrenatu():
    return _page("entrenatu.html", "Entrenamendu orria falta da")


@app.get("/parkings")
def list_parkings(user: dict = Depends(get_current_user)):
    items = []
    for parking_id in _active_parking_ids():
        if parking_id.startswith("."):
            continue
        if auth.can_view(user, parking_id):
            items.append(_parking_record(parking_id, user))
    return {"parkings": items}


@app.get("/parking")
def get_parking(parking_id: str, user: dict = Depends(get_current_user)):
    _require_view(user, parking_id)
    return _parking_record(parking_id, user)


@app.post("/parkings")
def create_parking(body: CreateParkingBody, user: dict = Depends(get_current_user)):
    if user.get("role") not in {"admin", "owner"}:
        raise HTTPException(403, "Aparkalekua sortzeko jabea edo admin izan behar duzu")
    parking_id = body.parking_id.strip() or _slug(body.name)
    profile_dir(parking_id)
    current = load_config(parking_id)
    visibility = "private" if str(body.visibility or "").strip().lower() == "private" else "public"
    save_config(
        parking_id,
        {
            "name": body.name.strip() or parking_id,
            "camera_url": body.camera_url.strip() or current.get("camera_url", ""),
            "owner_id": current.get("owner_id") or user["username"],
            "member_ids": current.get("member_ids") or [user["username"]],
            "visibility": visibility,
        },
    )
    auth.assign_parking(user["username"], parking_id, True)
    if body.camera_url.strip():
        _analyze_parking(parking_id)
    return _parking_record(parking_id, user)


@app.post("/parkings/favorite")
def set_parking_favorite(body: FavoriteBody, user: dict = Depends(get_current_user)):
    parking_id = _parking(body.parking_id)
    try:
        updated = auth.set_favorite(user, parking_id, bool(body.favorite))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "ok": True,
        "user": updated,
        "parking": _parking_record(parking_id, updated),
    }


@app.post("/parkings/privilege")
def set_parking_privilege(body: PrivilegeBody, user: dict = Depends(get_current_user)):
    _require_users(user, body.parking_id)
    try:
        updated = auth.set_privilege(user, body.parking_id, body.username, body.privilege)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    audit.record(
        "privilege_set",
        actor=user["username"],
        parking_id=body.parking_id,
        detail=f"{body.username}={body.privilege}",
        ok=True,
    )
    return {"ok": True, "user": updated, "parking": _parking_record(body.parking_id, user)}


@app.post("/parkings/visibility")
def set_parking_visibility(body: VisibilityBody, user: dict = Depends(get_current_user)):
    parking_id = _parking(body.parking_id)
    try:
        value = auth.set_visibility(user, parking_id, body.visibility)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    audit.record(
        "visibility_set",
        actor=user["username"],
        parking_id=parking_id,
        detail=value,
        ok=True,
    )
    return {"ok": True, "visibility": value, "parking": _parking_record(parking_id, user)}


@app.get("/parking/acl")
def get_parking_acl(parking_id: str, user: dict = Depends(get_current_user)):
    _require_users(user, parking_id)
    acl = auth.parking_acl(parking_id)
    actor = auth.privilege_of(user, parking_id)
    people = []
    for row in auth.list_users():
        if row.get("role") == "admin":
            priv = "admin"
        elif row.get("username") == load_config(parking_id).get("owner_id"):
            priv = "owner"
        else:
            priv = acl.get(row["username"], "none")
        people.append({**row, "privilege": priv})
    return {
        "parking_id": parking_id,
        "visibility": auth.parking_visibility(parking_id),
        "can_set_visibility": auth.can_edit(user, parking_id),
        "users": people,
        "can_assign_steward": actor in {"admin", "owner"},
    }


@app.get("/parking/display")
def get_parking_display(parking_id: str, user: dict = Depends(get_current_user)):
    parking_id = _require_edit(user, parking_id)
    token = ensure_display_token(parking_id)
    return {"token": token, "path": "/pantaila?t=" + token}


@app.post("/parking/display/rotate")
def rotate_parking_display(parking_id: str, user: dict = Depends(get_current_user)):
    parking_id = _require_edit(user, parking_id)
    token = rotate_display_token(parking_id)
    audit.record("display_rotate", actor=user["username"], parking_id=parking_id, ok=True)
    return {"token": token, "path": "/pantaila?t=" + token}


@app.get("/display/{token}")
def public_display(token: str, request: Request):
    ip = _client_ip(request)
    if not _rate_ok(_display_hits, ip, limit=40, window=10.0):
        raise HTTPException(429, "Saiakera gehiegi")
    parking_id = parking_id_for_display_token(token)
    if not parking_id:
        raise HTTPException(404, "Pantaila-esteka baliogabea")
    payload = _display_payload(parking_id)
    return JSONResponse(payload, headers={"Cache-Control": "no-store, private"})


@app.post("/admin/login")
def admin_login(body: PinBody, user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(403, "Administratzailea bakarrik")
    expected = str(os.environ.get("PARKINGAI_ADMIN_PIN") or "").strip()
    if not expected:
        raise HTTPException(404, "PIN desgaituta")
    if body.pin != expected:
        raise HTTPException(403, "PIN okerra")
    return {"ok": True}


@app.get("/plan")
def get_plan(parking_id: str, user: dict = Depends(get_current_user)):
    _require_view(user, parking_id)
    return load_plan(parking_id)


@app.post("/plan")
def set_plan(body: PlanBody, user: dict = Depends(get_current_user)):
    _require_edit(user, body.parking_id)
    return {
        "ok": True,
        **save_plan(
            body.parking_id,
            {
                "rows": body.rows,
                "cols": body.cols,
                "cells": body.cells,
                "outline": body.outline,
                "stalls": body.stalls,
            },
        ),
    }


@app.post("/register")
def register(body: RegisterBody, user: dict = Depends(get_current_user)):
    profile_dir(body.parking_id)
    current = load_config(body.parking_id)
    existing = bool(current) or bool(load_spaces(body.parking_id))
    if existing:
        _require_edit(user, body.parking_id)
    elif user.get("role") not in {"admin", "owner"}:
        raise HTTPException(403, "Aparkaleku berria jabeak edo adminak sortzen du")
    payload = {
        "camera_user": body.camera_user or current.get("camera_user", ""),
        "camera_pass": body.camera_pass or current.get("camera_pass", ""),
        "space_ids": body.space_ids or current.get("space_ids") or [],
        "owner_id": current.get("owner_id") or user["username"],
    }
    if body.camera_url.strip():
        payload["camera_url"] = body.camera_url.strip()
    elif current.get("camera_url"):
        payload["camera_url"] = current.get("camera_url")
    if body.name.strip():
        payload["name"] = body.name.strip()
    elif not current.get("name"):
        payload["name"] = body.parking_id
    save_config(body.parking_id, payload)
    if not current.get("owner_id"):
        auth.assign_parking(user["username"], body.parking_id, True)
    _analyze_parking(body.parking_id)
    return {"ok": True, **_parking_record(body.parking_id, user)}


@app.post("/spaces")
def set_spaces(body: SpacesBody, user: dict = Depends(get_current_user)):
    _require_edit(user, body.parking_id)
    cleaned = []
    for space in body.spaces:
        if "id" not in space:
            continue
        raw_points = space.get("points")
        if isinstance(raw_points, list) and len(raw_points) >= 3:
            points = [
                {"x": float(p["x"]), "y": float(p["y"])}
                for p in raw_points
                if isinstance(p, dict) and "x" in p and "y" in p
            ]
        elif all(k in space for k in ("x1", "y1", "x2", "y2")):
            points = [
                {"x": float(space["x1"]), "y": float(space["y1"])},
                {"x": float(space["x2"]), "y": float(space["y1"])},
                {"x": float(space["x2"]), "y": float(space["y2"])},
                {"x": float(space["x1"]), "y": float(space["y2"])},
            ]
        else:
            continue
        if len(points) < 3:
            continue
        cleaned.append({"id": space["id"], "points": points})
    save_spaces(body.parking_id, cleaned)
    _analyze_parking(body.parking_id)
    return {"ok": True, "count": len(cleaned)}


@app.post("/label")
def label(body: LabelBody, user: dict = Depends(get_current_user)):
    _require_edit(user, body.parking_id)
    with state_lock:
        cached = last_frames.get(body.parking_id)
        frame = None if cached is None else cached.copy()
    if frame is None:
        frame = _grab(body.parking_id)
    if frame is None:
        raise HTTPException(400, "Ez dago irudirik etiketatzeko")
    try:
        occupancy = {}
        with state_lock:
            occupancy = dict((last_status.get(body.parking_id) or {}).get("spaces") or {})
        spaces_now = load_spaces(body.parking_id)
        for row in spaces_now:
            occupancy.setdefault(row["id"], False)
        space = next((row for row in spaces_now if row["id"] == body.space_id), None)
        if space is None:
            raise HTTPException(400, "Plaza ezezaguna")
        vehicles = detect_vehicles(frame, spaces=spaces_now)
        before = predict_occupancy(body.parking_id, frame, vehicles)
        record = save_live_correction(
            body.parking_id,
            frame,
            occupancy,
            body.space_id,
            body.occupied,
            large_vehicle=body.large_vehicle,
            yolo_spaces=before,
        )
        extra_ids = []
        if body.large_vehicle:
            extra_ids = spaces_covered_by_large(spaces_now, vehicles, frame.shape)
        meta = _try_train(body.parking_id)
        predicted = predict_occupancy(body.parking_id, frame, vehicles).get(body.space_id)
        overlap, _ = occupancy_from_vehicles(space, vehicles, frame.shape)
        remember_correction(body.parking_id, body.space_id, body.occupied, overlap)
        for space_id in extra_ids:
            remember_correction(body.parking_id, space_id, True, overlap)
        effect = describe_correction_effect(
            body.parking_id, body.space_id, body.occupied, predicted, meta
        )
        status, annotated = analyze_frame(body.parking_id, frame, vehicles=vehicles)
        current = dict(status.get("spaces") or {})
        current[body.space_id] = body.occupied
        if body.large_vehicle:
            for space_id in extra_ids:
                current[space_id] = True
        status["spaces"] = current
        status["model_trained"] = bool(meta.get("approved_training") and meta.get("model_trained"))
        status["samples_free"] = meta.get("samples_free", 0)
        status["samples_occupied"] = meta.get("samples_occupied", 0)
        status["accuracy"] = meta.get("accuracy")
        status["learn_effect"] = effect["effect"]
        status["learn_message"] = effect["message"]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    with state_lock:
        last_status[body.parking_id] = status
        last_frames[body.parking_id] = frame
        last_annotated[body.parking_id] = annotated
    return {
        "ok": True,
        "accepted": True,
        "frame_id": record["id"],
        "space_id": body.space_id,
        "occupied": body.occupied,
        "predicted": predicted,
        **effect,
        **meta,
        "spaces": current,
        "large_vehicle": body.large_vehicle,
        "covered": extra_ids,
    }


@app.post("/train")
def train(body: TrainBody, user: dict = Depends(get_current_user)):
    _require_edit(user, body.parking_id)
    try:
        meta = train_model(body.parking_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _analyze_parking(body.parking_id)
    samples = int(meta.get("samples_free", 0)) + int(meta.get("samples_occupied", 0))
    message = (
        f"Algoritmoa birentrenatu da zuk berretsitako {samples} adibiderekin. "
        f"Zehaztasuna: {meta.get('accuracy')}."
    )
    return {
        "ok": True,
        "accepted": True,
        "effect": "learned",
        "kind": "ok",
        "message": message,
        "learned": True,
        "accuracy": meta.get("accuracy"),
        "samples": samples,
        "model_trained": True,
        **buffer_status(body.parking_id),
        **meta,
    }


@app.get("/status")
def status(parking_id: str, detail: bool = False, user: dict = Depends(get_current_user)):
    _require_view(user, parking_id)
    with state_lock:
        cached = last_status.get(parking_id)
    if cached is None:
        payload = {
            "ok": False,
            "spaces": {},
            "vehicles": 0,
            "camera_ok": False,
            "calibrated": bool(load_spaces(parking_id)),
            "updated_at": None,
            "error": "Sistema ez dago martxan.",
        }
        if not auth.can_view_camera(user, parking_id):
            return _public_status(payload)
        meta = refresh_sample_counts(parking_id)
        payload.update(
            {
                "model_trained": bool(meta.get("approved_training") and meta.get("model_trained")),
                "samples_free": meta.get("samples_free", 0),
                "samples_occupied": meta.get("samples_occupied", 0),
                "error": "Aparkaleku hau oraindik ez da analizatu. Erregistratu kamera URL-a.",
            }
        )
        return _with_stats(parking_id, payload, detail=detail)
    if not auth.can_view_camera(user, parking_id):
        return _public_status(cached)
    return _with_stats(parking_id, cached, detail=detail)


def _with_stats(parking_id: str, payload: dict, detail: bool = False) -> dict:
    merged = dict(payload)
    merged.update(buffer_status(parking_id))
    merged.pop("data_path", None)
    if not detail:
        return merged
    stats = dataset_stats(parking_id)
    merged.update(
        {
            "frames_total": stats["frames_total"],
            "frames_reviewed": stats["frames_reviewed"],
            "frames_pending": stats["frames_pending"],
            "corrections": stats["corrections"],
            "samples_free": stats["samples_free"],
            "samples_occupied": stats["samples_occupied"],
        }
    )
    return merged


@app.get("/raw")
def raw(parking_id: str, user: dict = Depends(get_current_user)):
    _require_camera(user, parking_id)
    with state_lock:
        frame = last_frames.get(parking_id)
    if frame is None:
        frame = _grab(parking_id)
    if frame is None:
        raise HTTPException(404, "Kamera irudirik ez")
    return _image_response(_jpeg(frame))


@app.get("/live")
def live(parking_id: str, user: dict = Depends(get_current_user)):
    _require_camera(user, parking_id)

    def frames():
        while True:
            with state_lock:
                cached = last_frames.get(parking_id)
                frame = None if cached is None else cached
            if frame is not None:
                payload = _jpeg(frame)
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(payload)).encode("ascii")
                    + b"\r\n\r\n"
                    + payload
                    + b"\r\n"
                )
            time.sleep(0.4)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/snapshot")
def snapshot(parking_id: str, user: dict = Depends(get_current_user)):
    _require_camera(user, parking_id)
    with state_lock:
        frame = last_annotated.get(parking_id) or last_frames.get(parking_id)
    if frame is None:
        raise HTTPException(404, "Argazkirik ez")
    return _image_response(_jpeg(frame))


def _capture_worker(parking_id: str, count: int, interval: float) -> None:
    previous = _last_capture_image.get(parking_id)
    attempts = 0
    while capture_jobs.get(parking_id, {}).get("running") and capture_jobs[parking_id]["done"] < count:
        attempts += 1
        if attempts > count * 5:
            break
        config = load_config(parking_id)
        with state_lock:
            cached = last_frames.get(parking_id)
            frame = None if cached is None else cached.copy()
        if frame is None:
            frame = _grab(parking_id)
        if frame is None or too_similar(previous, frame):
            time.sleep(max(1.0, interval))
            continue
        status, _ = analyze_frame(parking_id, frame)
        record = save_captured_frame(
            parking_id,
            frame,
            yolo_spaces=status.get("spaces") or {},
            source="batch",
        )
        previous = frame
        _last_capture_image[parking_id] = frame
        capture_jobs[parking_id]["done"] += 1
        capture_jobs[parking_id]["last_id"] = record["id"]
        time.sleep(max(1.0, interval))
    if parking_id in capture_jobs:
        capture_jobs[parking_id]["running"] = False


@app.post("/capture/start")
def capture_start(body: CaptureBody, user: dict = Depends(get_current_user)):
    _require_edit(user, body.parking_id)
    if not load_spaces(body.parking_id):
        raise HTTPException(400, "Lehenik gorde plazak")
    job = capture_jobs.get(body.parking_id)
    if job and job.get("running"):
        return job
    capture_jobs[body.parking_id] = {
        "running": True,
        "target": max(1, min(body.count, 200)),
        "done": 0,
        "interval_seconds": body.interval_seconds,
        "last_id": None,
    }
    threading.Thread(
        target=_capture_worker,
        args=(body.parking_id, capture_jobs[body.parking_id]["target"], body.interval_seconds),
        daemon=True,
    ).start()
    return capture_jobs[body.parking_id]


@app.get("/capture/status")
def capture_status(parking_id: str, user: dict = Depends(get_current_user)):
    _require_edit(user, parking_id)
    job = capture_jobs.get(parking_id) or {"running": False, "target": 0, "done": 0}
    stats = dataset_stats(parking_id)
    return {**job, **stats}


@app.get("/dataset/reliability")
def get_dataset_reliability(parking_id: str, user: dict = Depends(get_current_user)):
    _require_edit(user, parking_id)
    return reliability_report(parking_id)


@app.get("/dataset/stats")
def get_dataset_stats(parking_id: str, user: dict = Depends(get_current_user)):
    _require_edit(user, parking_id)
    return {**dataset_stats(parking_id), **buffer_status(parking_id)}


@app.post("/dataset/grab")
def dataset_grab(body: GrabBody, user: dict = Depends(get_current_user)):
    _require_edit(user, body.parking_id)
    if not load_spaces(body.parking_id):
        raise HTTPException(400, "Lehenik gorde plazak konfigurazioan")
    with state_lock:
        cached = last_frames.get(body.parking_id)
        frame = None if cached is None else cached.copy()
    if frame is None:
        frame = _grab(body.parking_id)
    if frame is None:
        raise HTTPException(400, "Ez dago irudirik hartzeko")
    occupancy = predict_occupancy(body.parking_id, frame)
    record = save_captured_frame(
        body.parking_id,
        frame,
        yolo_spaces=occupancy,
        source="manual",
    )
    return {
        "ok": True,
        "accepted": True,
        "message": "Irudia gordeta. Markatu plaza bakoitza libre edo okupatu, eta gorde.",
        "kind": "ok",
        "frame": public_frame(record),
        **training_queue(body.parking_id),
        **dataset_stats(body.parking_id),
        **buffer_status(body.parking_id),
    }


@app.get("/dataset/pending")
def get_pending(parking_id: str, user: dict = Depends(get_current_user)):
    _require_edit(user, parking_id)
    queue = training_queue(parking_id)
    return {**queue, "frames": queue["pending"], **buffer_status(parking_id), **dataset_stats(parking_id)}


@app.get("/dataset/image")
def dataset_image(parking_id: str, frame_id: str, user: dict = Depends(get_current_user)):
    _require_edit(user, parking_id)
    try:
        image, _ = load_frame_image(parking_id, frame_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return _image_response(_jpeg(image))


@app.post("/dataset/review")
def dataset_review(body: ReviewBody, user: dict = Depends(get_current_user)):
    _require_edit(user, body.parking_id)
    try:
        record, image = review_frame(body.parking_id, body.frame_id, body.spaces)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    meta = _try_train(body.parking_id)
    predicted_map = predict_occupancy(body.parking_id, image)
    corrected = record.get("corrected") or []
    for sid, occupied in (record.get("spaces") or {}).items():
        remember_correction(body.parking_id, sid, bool(occupied))
    if corrected:
        space_id = corrected[0]
        occupied = bool((record.get("spaces") or {}).get(space_id))
        effect = describe_correction_effect(
            body.parking_id, space_id, occupied, predicted_map.get(space_id), meta
        )
    else:
        effect = {
            "effect": "confirmed",
            "kind": "ok",
            "message": "Irudia zuzena: algoritmoak asmatu du. Adibideak entrenamenduan sartu dira"
            + (" eta modelo birentrenatu da." if meta.get("just_trained") else "."),
            "learned": bool(meta.get("just_trained")),
            "model_agrees": True,
            "just_trained": bool(meta.get("just_trained")),
        }
    return {
        "ok": True,
        "accepted": True,
        "frame": record,
        **effect,
        "just_trained": bool(meta.get("just_trained")),
        "train_hint": meta.get("train_hint"),
        **dataset_stats(body.parking_id),
        **meta,
        "reliability": reliability_report(body.parking_id),
        "message": effect["message"],
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=APP_CONFIG.get("host", "0.0.0.0"),
        port=int(APP_CONFIG.get("port", 8000)),
        reload=False,
    )
