from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTBOX = ROOT / "data" / "outbox"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _smtp_settings() -> dict:
    settings = {}
    config_path = ROOT / "config.json"
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8")).get("smtp") or {}
            if isinstance(raw, dict):
                settings.update(raw)
        except (OSError, json.JSONDecodeError):
            pass
    env = {
        "host": os.environ.get("PARKINGAI_SMTP_HOST"),
        "port": os.environ.get("PARKINGAI_SMTP_PORT"),
        "user": os.environ.get("PARKINGAI_SMTP_USER"),
        "password": os.environ.get("PARKINGAI_SMTP_PASSWORD"),
        "from": os.environ.get("PARKINGAI_SMTP_FROM"),
    }
    for key, value in env.items():
        if value:
            settings[key] = value
    return settings


def send_mail(to: str, subject: str, text: str) -> dict:
    """Bidali SMTP bidez, edo gorde data/outbox-en garapenerako."""
    to = (to or "").strip()
    OUTBOX.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "@._-+" else "_" for ch in to)[:80]
    path = OUTBOX / f"{_utc_stamp()}-{safe}.txt"
    path.write_text(f"To: {to}\nSubject: {subject}\n\n{text}\n", encoding="utf-8")

    smtp = _smtp_settings()
    host = str(smtp.get("host") or "").strip()
    sent = False
    error = ""
    if host and to:
        try:
            port = int(smtp.get("port") or 587)
            user = str(smtp.get("user") or "")
            password = str(smtp.get("password") or "")
            sender = str(smtp.get("from") or user or "parkingai@localhost")
            message = EmailMessage()
            message["From"] = sender
            message["To"] = to
            message["Subject"] = subject
            message.set_content(text)
            with smtplib.SMTP(host, port, timeout=12) as smtp:
                smtp.starttls()
                if user:
                    smtp.login(user, password)
                smtp.send_message(message)
            sent = True
        except (OSError, smtplib.SMTPException, ValueError) as exc:
            error = str(exc)[:200]
    return {"stored": str(path), "sent": sent, "error": error}
