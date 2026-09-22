from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
from pathlib import Path

from occupancy import load_config, profile_dir, save_config

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
USERS_PATH = DATA_ROOT / "users.json"
ITERATIONS = 120_000
SESSION_TTL = 60 * 60 * 12
SESSION_COOKIE = "parkingai_session"
USERNAME_RE = re.compile(r"^[a-z0-9._-]{3,32}$")
EMAIL_RE = re.compile(r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$", re.I)
PARKING_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
WEAK_PASSWORDS = {
    "admin123",
    "password",
    "password1",
    "password12",
    "12345678",
    "123456789",
    "1234567890",
    "qwerty123",
    "qwerty1234",
    "parkingai",
    "parkingai1",
    "letmein123",
    "changeme123",
}
RESET_TTL = 30 * 60
INVITE_TTL = 48 * 60 * 60
VERIFY_TTL = 24 * 60 * 60
TOKENS_PATH = DATA_ROOT / "auth_tokens.json"
_sessions: dict[str, dict] = {}
_failures: dict[str, list[float]] = {}


def _now() -> float:
    return time.time()


def _read_users() -> dict:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not USERS_PATH.exists():
        return {"users": []}
    return json.loads(USERS_PATH.read_text(encoding="utf-8"))


def _write_users(payload: dict) -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    USERS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _hash_password(password: str, salt_hex: str | None = None) -> str:
    if salt_hex is None:
        raw_salt = secrets.token_bytes(16)
        salt_hex = raw_salt.hex()
    else:
        raw_salt = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), raw_salt, ITERATIONS)
    return f"{salt_hex}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    if not stored:
        return False
    if "$" not in stored:
        return hmac.compare_digest(stored, password)
    salt, digest = stored.split("$", 1)
    try:
        check = _hash_password(password, salt).split("$", 1)[1]
        if hmac.compare_digest(check, digest):
            return True
    except ValueError:
        pass
    legacy = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        ITERATIONS,
    ).hex()
    return hmac.compare_digest(legacy, digest)


def safe_parking_id(parking_id: str) -> str:
    text = (parking_id or "").strip()
    if not PARKING_ID_RE.match(text):
        raise ValueError("Aparkaleku ID baliogabea")
    return text


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_email(email: str) -> str:
    value = normalize_email(email)
    if not EMAIL_RE.match(value):
        raise ValueError("Korreo baliogabea")
    return value


def password_issues(password: str, username: str = "", email: str = "") -> list[str]:
    issues: list[str] = []
    if not password or len(password) < 12:
        issues.append("Gutxienez 12 karaktere")
    if password and (" " in password or password != password.strip()):
        issues.append("Zuriunerik gabe")
    if password and not re.search(r"[a-z]", password):
        issues.append("Minuskula bat")
    if password and not re.search(r"[A-Z]", password):
        issues.append("Maiuskula bat")
    if password and not re.search(r"[0-9]", password):
        issues.append("Zenbaki bat")
    if username and password and password.lower() == username.strip().lower():
        issues.append("Ez erabiltzaile-izena")
    local = normalize_email(email).split("@")[0] if email else ""
    if local and password and password.lower() == local:
        issues.append("Ez korreoaren izena")
    if password and password.lower() in WEAK_PASSWORDS:
        issues.append("Ohikoegia da")
    return issues


def validate_password(password: str, username: str = "", email: str = "") -> None:
    issues = password_issues(password, username, email)
    if issues:
        raise ValueError("Pasahitz ahula: " + ", ".join(issues).lower() + ".")


def public_user(user: dict) -> dict:
    email = normalize_email(str(user.get("email") or ""))
    invited = not bool(user.get("password_hash"))
    verified = bool(user.get("email_verified"))
    return {
        "username": user.get("username"),
        "name": user.get("name") or user.get("username"),
        "email": email,
        "role": user.get("role") or "user",
        "parking_ids": list(user.get("parking_ids") or []),
        "favorite_parking_ids": list(user.get("favorite_parking_ids") or []),
        "email_verified": verified,
        "must_change_password": bool(user.get("must_change_password")),
        "must_set_email": not bool(email),
        "must_verify_email": bool(email) and not verified and not invited,
        "invited": invited,
    }


def list_users() -> list[dict]:
    return [public_user(row) for row in _read_users().get("users", [])]


def find_user(username: str) -> dict | None:
    key = (username or "").strip().lower()
    for row in _read_users().get("users", []):
        if str(row.get("username", "")).lower() == key:
            return row
    return None


def find_user_by_email(email: str) -> dict | None:
    key = normalize_email(email)
    if not key:
        return None
    for row in _read_users().get("users", []):
        if normalize_email(str(row.get("email") or "")) == key:
            return row
    return None


def find_identity(identity: str) -> dict | None:
    text = (identity or "").strip()
    if "@" in text:
        return find_user_by_email(text)
    return find_user(text)


def create_user(
    username: str,
    password: str = "",
    name: str = "",
    role: str = "user",
    allow_weak: bool = False,
    email: str = "",
    invited: bool = False,
) -> dict:
    username = (username or "").strip().lower()
    if not USERNAME_RE.match(username):
        raise ValueError("Erabiltzailea: 3-32 karaktere, hizkiak, zenbakiak, . _ -")
    email_value = validate_email(email) if email else ""
    if email_value and find_user_by_email(email_value) is not None:
        raise ValueError("Korreo hau jada erregistratuta dago")
    if invited:
        password_hash = ""
    elif allow_weak:
        if not password or len(password) < 8:
            raise ValueError("Pasahitzak gutxienez 8 karaktere behar ditu")
        password_hash = _hash_password(password)
    else:
        validate_password(password, username, email_value)
        password_hash = _hash_password(password)
    if role not in {"admin", "owner", "user"}:
        raise ValueError("Rol ezezaguna")
    if find_user(username):
        raise ValueError("Erabiltzaile hau jada badago")
    data = _read_users()
    row = {
        "username": username,
        "name": name.strip() or username,
        "email": email_value,
        "email_verified": False,
        "role": role,
        "password_hash": password_hash,
        "parking_ids": [],
        "favorite_parking_ids": [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    data.setdefault("users", []).append(row)
    _write_users(data)
    return public_user(row)


def unique_username(seed: str) -> str:
    base = re.sub(r"[^a-z0-9._-]", "", (seed or "").strip().lower())
    base = base.strip(".-_") or "erabiltzaile"
    base = base[:24]
    if USERNAME_RE.match(base) and find_user(base) is None:
        return base
    for index in range(2, 1000):
        candidate = f"{base[:20]}{index}"
        if USERNAME_RE.match(candidate) and find_user(candidate) is None:
            return candidate
    return unique_username("erabiltzaile" + secrets.token_hex(3))


def register_account(email: str, password: str, name: str = "", username: str = "") -> tuple[dict, str]:
    email_value = validate_email(email)
    if find_user_by_email(email_value) is not None:
        raise ValueError("Korreo hau jada erregistratuta dago")
    handle = (username or "").strip().lower()
    if handle:
        if not USERNAME_RE.match(handle):
            raise ValueError("Erabiltzailea: 3-32 karaktere, hizkiak, zenbakiak, . _ -")
        if find_user(handle) is not None:
            raise ValueError("Erabiltzaile hau jada badago")
    else:
        handle = unique_username(email_value.split("@")[0])
    created = create_user(
        handle,
        password=password,
        name=name or handle,
        role="user",
        email=email_value,
    )
    raw = issue_account_token(created["username"], "verify", VERIFY_TTL, email_value)
    return created, raw


def set_user_parkings(username: str, parking_ids: list[str]) -> dict:
    data = _read_users()
    found = None
    for row in data.get("users", []):
        if str(row.get("username", "")).lower() == username.strip().lower():
            row["parking_ids"] = sorted(set(parking_ids))
            found = row
            break
    if found is None:
        raise ValueError("Erabiltzailea ez da aurkitu")
    _write_users(data)
    return public_user(found)


def assign_parking(username: str, parking_id: str, assigned: bool = True) -> dict:
    actor = {"username": "admin", "role": "admin"}
    return set_privilege(actor, parking_id, username, "user" if assigned else "none")


def _flag_weak_passwords() -> None:
    data = _read_users()
    changed = False
    for row in data.get("users", []):
        stored = row.get("password_hash") or ""
        if stored and _verify_password("admin123", stored):
            if not row.get("must_change_password"):
                row["must_change_password"] = True
                changed = True
        elif row.get("must_change_password") and stored and not _verify_password("admin123", stored):
            row["must_change_password"] = False
            changed = True
    if changed:
        _write_users(data)


def change_password(username: str, current_password: str, new_password: str) -> dict:
    user = find_user(username)
    if user is None or not _verify_password(current_password, user.get("password_hash") or ""):
        raise ValueError("Oraingo pasahitza okerra da")
    validate_password(new_password, username, str(user.get("email") or ""))
    if hmac.compare_digest(current_password, new_password):
        raise ValueError("Pasahitz berria ezin da oraingoa izan")
    data = _read_users()
    for row in data.get("users", []):
        if str(row.get("username", "")).lower() == username.strip().lower():
            row["password_hash"] = _hash_password(new_password)
            row["must_change_password"] = False
            break
    _write_users(data)
    logout_user(username)
    token = issue_session(username)
    return {"token": token, "user": public_user(find_user(username) or {})}


def issue_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        "username": username,
        "expires": _now() + SESSION_TTL,
    }
    return token


def logout_user(username: str) -> None:
    key = (username or "").strip().lower()
    dead = [token for token, session in _sessions.items() if session.get("username") == key]
    for token in dead:
        _sessions.pop(token, None)


def ensure_bootstrap(config: dict) -> None:
    if list_users():
        _flag_weak_passwords()
        return
    username = str(config.get("admin_user") or "admin").strip().lower()
    password = str(config.get("admin_password") or "admin123")
    create_user(username, password, name="Administratzailea", role="admin", allow_weak=True)
    if password.lower() in WEAK_PASSWORDS or password == "admin123":
        data = _read_users()
        for row in data.get("users", []):
            if row.get("username") == username:
                row["must_change_password"] = True
        _write_users(data)
    official = DATA_ROOT / "proba-ofiziala"
    if official.exists():
        save_config(
            "proba-ofiziala",
            {
                "name": load_config("proba-ofiziala").get("name") or "Proba ofiziala",
                "owner_id": username,
            },
        )
        assign_parking(username, "proba-ofiziala", True)


def login(username: str, password: str, client_id: str = "") -> dict:
    key = f"{client_id}|{(username or '').strip().lower()}"
    now = _now()
    recent = [stamp for stamp in _failures.get(key, []) if now - stamp < 300]
    if len(recent) >= 8:
        raise ValueError("Saiakera gehiegi. Itxaron minutu batzuk.")
    user = find_identity(username)
    if user is None or not user.get("password_hash") or not _verify_password(
        password, user.get("password_hash") or ""
    ):
        recent.append(now)
        _failures[key] = recent
        if user is not None and not user.get("password_hash"):
            raise ValueError("Korreoko gonbidapena osatu behar duzu pasahitza sortzeko")
        raise ValueError("Erabiltzailea edo pasahitza okerra")
    if user.get("email") and not user.get("email_verified"):
        raise ValueError("Berretsi korreoa saioa hasi aurretik")
    _failures.pop(key, None)
    token = issue_session(user["username"])
    return {"token": token, "user": public_user(user)}


def logout(token: str) -> None:
    _sessions.pop(token or "", None)


def user_from_token(token: str | None) -> dict | None:
    if not token:
        return None
    session = _sessions.get(token)
    if not session:
        return None
    if session["expires"] < _now():
        _sessions.pop(token, None)
        return None
    user = find_user(session["username"])
    return public_user(user) if user else None


PRIVILEGES = ("none", "user", "steward")


def parking_acl(parking_id: str) -> dict:
    return dict(load_config(parking_id).get("acl") or {})


def privilege_of(user: dict | None, parking_id: str) -> str:
    if not user:
        return "none"
    if user.get("role") == "admin":
        return "admin"
    username = user.get("username")
    config = load_config(parking_id)
    if config.get("owner_id") == username:
        return "owner"
    acl = parking_acl(parking_id)
    if username in acl:
        return str(acl.get(username) or "none")
    if username in (config.get("member_ids") or []):
        return "user"
    if parking_id in (user.get("parking_ids") or []):
        return "user"
    return "none"


def parking_visibility(parking_id: str) -> str:
    raw = str(load_config(parking_id).get("visibility") or "public").strip().lower()
    return "private" if raw == "private" else "public"


def is_public_parking(parking_id: str) -> bool:
    return parking_visibility(parking_id) == "public"


def is_assigned(user: dict | None, parking_id: str) -> bool:
    if not user:
        return False
    username = user.get("username")
    if not username:
        return False
    config = load_config(parking_id)
    if config.get("owner_id") == username:
        return True
    acl = parking_acl(parking_id)
    if str(acl.get(username) or "") in {"user", "steward"}:
        return True
    if username in (config.get("member_ids") or []):
        return True
    if parking_id in (user.get("parking_ids") or []):
        return True
    return False


def favorite_ids_of(user: dict | None) -> list[str]:
    if not user:
        return []
    found = find_user(user.get("username") or "")
    source = found if found is not None else user
    return [str(pid) for pid in (source.get("favorite_parking_ids") or [])]


def is_favorite(user: dict | None, parking_id: str) -> bool:
    return bool(parking_id) and parking_id in favorite_ids_of(user) and is_public_parking(parking_id)


def set_favorite(user: dict, parking_id: str, favorite: bool) -> dict:
    parking_id = safe_parking_id(parking_id)
    if not can_view(user, parking_id):
        raise ValueError("Ez duzu aparkaleku hau ikusteko baimenik")
    if favorite and not is_public_parking(parking_id):
        raise ValueError("Faboritoak aparkaleku publikoentzat dira")
    username = (user.get("username") or "").strip().lower()
    data = _read_users()
    found = None
    for row in data.get("users", []):
        if str(row.get("username", "")).lower() == username:
            found = row
            break
    if found is None:
        raise ValueError("Erabiltzailea ez da aurkitu")
    current = {str(pid) for pid in (found.get("favorite_parking_ids") or [])}
    if favorite:
        current.add(parking_id)
    else:
        current.discard(parking_id)
    found["favorite_parking_ids"] = sorted(current)
    _write_users(data)
    return public_user(found)


def can_view(user: dict | None, parking_id: str) -> bool:
    if not user:
        return False
    if is_public_parking(parking_id):
        return True
    return privilege_of(user, parking_id) in {"admin", "owner", "steward", "user"}


def can_edit(user: dict | None, parking_id: str) -> bool:
    return privilege_of(user, parking_id) in {"admin", "owner"}


def can_view_camera(user: dict | None, parking_id: str) -> bool:
    return can_edit(user, parking_id)


def can_manage_users(user: dict | None, parking_id: str) -> bool:
    return privilege_of(user, parking_id) in {"admin", "owner", "steward"}


def set_visibility(user: dict | None, parking_id: str, visibility: str) -> str:
    if not can_edit(user, parking_id):
        raise ValueError("Ikusgarritasuna jabeak edo adminak aldatzen du")
    value = "private" if str(visibility or "").strip().lower() == "private" else "public"
    save_config(parking_id, {"visibility": value})
    return value


def set_privilege(actor: dict, parking_id: str, username: str, privilege: str) -> dict:
    username = (username or "").strip().lower()
    privilege = (privilege or "none").strip().lower()
    if privilege not in PRIVILEGES:
        raise ValueError("Pribilegio ezezaguna")
    if find_user(username) is None:
        raise ValueError("Erabiltzailea ez da aurkitu")
    actor_priv = privilege_of(actor, parking_id)
    if actor_priv not in {"admin", "owner", "steward"}:
        raise ValueError("Ez duzu erabiltzaileak kudeatzeko baimenik")
    if actor_priv == "steward" and privilege == "steward":
        raise ValueError("Kudeatzaileak ezin du beste kudeatzailerik izendatu")
    acl = parking_acl(parking_id)
    if privilege == "none":
        acl.pop(username, None)
    else:
        acl[username] = privilege
    members = [name for name, role in acl.items() if role in {"user", "steward"}]
    save_config(parking_id, {"acl": acl, "member_ids": members})
    ids = [pid for pid, data in _user_parking_map(username).items()]
    if privilege == "none":
        ids = [pid for pid in ids if pid != parking_id]
    elif parking_id not in ids:
        ids.append(parking_id)
    return set_user_parkings(username, ids)


def _user_parking_map(username: str) -> dict:
    found = find_user(username) or {}
    return {pid: True for pid in (found.get("parking_ids") or [])}


def _read_tokens() -> dict:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not TOKENS_PATH.exists():
        return {"tokens": []}
    try:
        payload = json.loads(TOKENS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"tokens": []}
    return payload if isinstance(payload, dict) else {"tokens": []}


def _write_tokens(payload: dict) -> None:
    TOKENS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _purge_tokens(data: dict) -> dict:
    now = _now()
    data["tokens"] = [row for row in data.get("tokens", []) if float(row.get("expires") or 0) > now]
    return data


def issue_account_token(username: str, purpose: str, ttl: int, email: str = "") -> str:
    raw = secrets.token_urlsafe(32)
    data = _purge_tokens(_read_tokens())
    data.setdefault("tokens", []).append(
        {
            "id": secrets.token_hex(8),
            "username": username,
            "email": normalize_email(email),
            "purpose": purpose,
            "hash": _hash_token(raw),
            "expires": _now() + ttl,
        }
    )
    _write_tokens(data)
    return raw


def consume_account_token(raw: str, purpose: str) -> dict:
    token = (raw or "").strip()
    if len(token) < 16:
        raise ValueError("Esteka baliogabea edo iraungia")
    digest = _hash_token(token)
    data = _purge_tokens(_read_tokens())
    found = None
    kept = []
    for row in data.get("tokens", []):
        if found is None and row.get("purpose") == purpose and row.get("hash") == digest:
            found = row
            continue
        kept.append(row)
    if found is None:
        raise ValueError("Esteka baliogabea edo iraungia")
    data["tokens"] = kept
    _write_tokens(data)
    user = find_user(str(found.get("username") or ""))
    if user is None:
        raise ValueError("Erabiltzailea ez da aurkitu")
    return {"user": user, "email": found.get("email") or user.get("email") or ""}


def peek_account_token(raw: str, purpose: str) -> dict | None:
    token = (raw or "").strip()
    if len(token) < 16:
        return None
    digest = _hash_token(token)
    data = _purge_tokens(_read_tokens())
    for row in data.get("tokens", []):
        if row.get("purpose") == purpose and row.get("hash") == digest:
            user = find_user(str(row.get("username") or ""))
            if user:
                return public_user(user)
    return None


def _save_user_fields(username: str, **fields) -> dict:
    data = _read_users()
    found = None
    for row in data.get("users", []):
        if str(row.get("username", "")).lower() == username.strip().lower():
            row.update(fields)
            found = row
            break
    if found is None:
        raise ValueError("Erabiltzailea ez da aurkitu")
    _write_users(data)
    return public_user(found)


def invite_user(username: str, email: str, name: str = "", role: str = "user") -> tuple[dict, str]:
    created = create_user(username, name=name, role=role, email=email, invited=True)
    raw = issue_account_token(created["username"], "invite", INVITE_TTL, email)
    return created, raw


def request_password_reset(identity: str, email: str = "", allow_bind: bool = False) -> dict | None:
    user = find_identity(identity)
    bind_email = normalize_email(email)
    if user is None and bind_email:
        user = find_user_by_email(bind_email)
    if user is None:
        return None
    target = normalize_email(str(user.get("email") or ""))
    if not target:
        if not allow_bind or not bind_email:
            return None
        if find_user_by_email(bind_email) and find_user_by_email(bind_email).get("username") != user.get("username"):
            return None
        _save_user_fields(user["username"], email=validate_email(bind_email), email_verified=False)
        target = bind_email
        user = find_user(user["username"])
    raw = issue_account_token(user["username"], "reset", RESET_TTL, target)
    return {"user": public_user(user), "email": target, "token": raw}


def complete_password_reset(raw_token: str, new_password: str) -> dict:
    payload = consume_account_token(raw_token, "reset")
    user = payload["user"]
    email = payload.get("email") or user.get("email") or ""
    validate_password(new_password, user["username"], email)
    logout_user(user["username"])
    fields = {
        "password_hash": _hash_password(new_password),
        "must_change_password": False,
        "email_verified": True if email else bool(user.get("email_verified")),
    }
    if email:
        fields["email"] = normalize_email(email)
    public = _save_user_fields(user["username"], **fields)
    token = issue_session(user["username"])
    return {"token": token, "user": public}


def complete_invite(raw_token: str, new_password: str) -> dict:
    payload = consume_account_token(raw_token, "invite")
    user = payload["user"]
    email = payload.get("email") or user.get("email") or ""
    validate_password(new_password, user["username"], email)
    public = _save_user_fields(
        user["username"],
        password_hash=_hash_password(new_password),
        must_change_password=False,
        email_verified=True,
        email=normalize_email(email),
    )
    token = issue_session(user["username"])
    return {"token": token, "user": public}


def request_email_verification(username: str) -> dict | None:
    user = find_user(username)
    email = normalize_email(str((user or {}).get("email") or ""))
    if user is None or not email:
        return None
    raw = issue_account_token(username, "verify", VERIFY_TTL, email)
    return {"user": public_user(user), "email": email, "token": raw}


def complete_email_verification(raw_token: str) -> dict:
    payload = consume_account_token(raw_token, "verify")
    user = payload["user"]
    email = payload.get("email") or user.get("email") or ""
    public = _save_user_fields(
        user["username"],
        email=normalize_email(email),
        email_verified=True,
    )
    token = issue_session(user["username"])
    return {"token": token, "user": public}


def set_account_email(username: str, email: str) -> dict:
    value = validate_email(email)
    other = find_user_by_email(value)
    if other is not None and other.get("username") != username:
        raise ValueError("Korreo hau jada erregistratuta dago")
    public = _save_user_fields(username, email=value, email_verified=False)
    raw = issue_account_token(username, "verify", VERIFY_TTL, value)
    return {"user": public, "email": value, "token": raw}
