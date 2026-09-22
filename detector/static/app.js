const TOKEN_KEY = "parkingai_token";
const USER_KEY = "parkingai_user";

function token() {
  return "";
}
function setToken(_value) {
  localStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(TOKEN_KEY);
}
function currentUser() {
  try { return JSON.parse(sessionStorage.getItem(USER_KEY) || "null"); }
  catch { return null; }
}
function setCurrentUser(user) {
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(TOKEN_KEY);
  if (user) sessionStorage.setItem(USER_KEY, JSON.stringify(user));
  else sessionStorage.removeItem(USER_KEY);
}
function authHeaders(json) {
  const headers = {};
  if (json !== false) headers["Content-Type"] = "application/json";
  return headers;
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}
async function api(path, options = {}) {
  const opts = { credentials: "include", ...options };
  opts.headers = { ...authHeaders(!!opts.body && !(opts.body instanceof FormData)), ...(options.headers || {}) };
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
  if (res.status === 403 && data.must_change_password && !path.startsWith("/auth/")) {
    if (location.pathname !== "/" && location.pathname !== "") location.href = "/";
    throw new Error("Pasahitza aldatu behar duzu");
  }
  if (res.status === 401 && !path.startsWith("/auth/")) {
    setToken("");
    setCurrentUser(null);
    const here = (location.pathname || "/").replace(/\/+$/, "") || "/";
    if (here !== "/" && here !== "/aparkalekuak" && here !== "/berreskuratu") location.href = "/";
  }
  if (!res.ok) {
    const detail = data.detail || data.error || text || res.status;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}
function requireAuth() {
  return true;
}
function canEditParking(parking) {
  const user = currentUser();
  if (!user) return false;
  if (user.role === "admin") return true;
  if (parking && parking.can_edit) return true;
  if (parking && parking.owner_id === user.username) return true;
  return user.role === "owner";
}
function canViewCamera(parking) {
  if (parking && parking.can_view_camera === true) return true;
  return canEditParking(parking);
}
function withToken(url) {
  return url;
}
async function authImage(img, path) {
  if (!img || img._loading) return;
  img._loading = true;
  try {
    const res = await fetch(path, { credentials: "include", cache: "no-store" });
    if (!res.ok) {
      if (img._blobUrl) URL.revokeObjectURL(img._blobUrl);
      img._blobUrl = "";
      img.removeAttribute("src");
      return;
    }
    const blob = await res.blob();
    if (img._blobUrl) URL.revokeObjectURL(img._blobUrl);
    img._blobUrl = URL.createObjectURL(blob);
    img.src = img._blobUrl;
  } finally {
    img._loading = false;
  }
}
