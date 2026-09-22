from __future__ import annotations

import re
from urllib.parse import urljoin

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

HEADERS = {
    "User-Agent": "Mozilla/5.0 ParkingAI/1.0",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

_rtsp_caps: dict[str, cv2.VideoCapture] = {}


def grab_frame(url: str, user: str = "", password: str = "") -> np.ndarray | None:
    if not url:
        return None
    lowered = url.lower().strip()
    if lowered.startswith("rtsp://"):
        return _grab_video(url)
    frame = _grab_http(url, user, password, depth=0)
    if frame is not None:
        return frame
    return _grab_video(url)


def _auths(user: str, password: str):
    yield None
    if user:
        yield HTTPBasicAuth(user, password)
        yield HTTPDigestAuth(user, password)


def _decode_bytes(content: bytes) -> np.ndarray | None:
    if not content:
        return None
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image


def _grab_http(url: str, user: str, password: str, depth: int = 0) -> np.ndarray | None:
    last_error = None
    for auth in _auths(user, password):
        try:
            response = requests.get(url, timeout=3.5, auth=auth, headers=HEADERS, stream=True)
            if response.status_code >= 400:
                last_error = response.status_code
                continue
            ctype = (response.headers.get("content-type") or "").lower()
            if "multipart" in ctype or "mjpeg" in ctype or "mjpg" in ctype:
                frame = _first_mjpeg_frame(response)
                if frame is not None:
                    return frame
            content = response.content
            frame = _decode_bytes(content)
            if frame is not None:
                return frame
            if depth < 1 and (
                "html" in ctype
                or content[:32].lstrip().lower().startswith(b"<!doctype")
                or b"<html" in content[:400].lower()
            ):
                nested = _image_from_html(url, content.decode("utf-8", errors="ignore"), user, password)
                if nested is not None:
                    return nested
        except Exception as exc:
            last_error = exc
    print(f"Kamera HTTP irakurketak huts egin du: {last_error}")
    return None


def _first_mjpeg_frame(response) -> np.ndarray | None:
    buffer = b""
    for chunk in response.iter_content(chunk_size=4096):
        if not chunk:
            break
        buffer += chunk
        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")
        if start != -1 and end != -1 and end > start:
            return _decode_bytes(buffer[start : end + 2])
        if len(buffer) > 8_000_000:
            break
    return None


def _image_from_html(page_url: str, html: str, user: str, password: str) -> np.ndarray | None:
    patterns = [
        r'property=["\']og:image["\']\s+content=["\']([^"\']+)',
        r'content=["\']([^"\']+)["\']\s+property=["\']og:image["\']',
        r'<img[^>]+src=["\']([^"\']+)["\']',
        r'https?://[^"\']+\.(?:jpg|jpeg|png|mjpg)(?:\?[^"\']*)?',
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(re.findall(pattern, html, flags=re.I))
    seen = set()
    for raw in candidates:
        abs_url = urljoin(page_url, raw.strip())
        if abs_url in seen or abs_url == page_url:
            continue
        seen.add(abs_url)
        if any(skip in abs_url.lower() for skip in ("logo", "icon", "sprite", "svg", "favicon")):
            continue
        frame = _grab_http(abs_url, user, password, depth=1)
        if frame is not None and min(frame.shape[:2]) >= 120:
            return frame
    return None


def _grab_video(url: str) -> np.ndarray | None:
    cap = _rtsp_caps.get(url)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _rtsp_caps[url] = cap
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    if not ok:
        cap.release()
        _rtsp_caps.pop(url, None)
        return None
    return frame
