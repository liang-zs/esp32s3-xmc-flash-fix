"""Optional password for the web console (WEB_TOKEN).

The console can drive the iPhone, and anyone on the same Wi-Fi can open it. With WEB_TOKEN set, a browser on
another machine first gets a login page; the right password sets a cookie (30 days), then the console works as
before. Requests from this Mac itself (loopback: tools/regress.py, a local browser) need nothing. Scripts on other
machines can send "Authorization: Bearer <WEB_TOKEN>". Without WEB_TOKEN nothing changes.
"""

import hashlib
import hmac
import time
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE = "hidagent_auth"
COOKIE_DAYS = 30
LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
MAX_FAILS, LOCKOUT_S = 5, 60  # wrong passwords in a row from one address, then that address waits a minute

LOGIN_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESP32 iPhone Agent</title>
<style>
  :root {{ --bg: #fff; --fg: #111; --muted: #777; --line: #e6e6e6; --mark: #ff3b30; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg: #0d0d0d; --fg: #eee; --muted: #888; --line: #262626; }} }}
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: var(--bg); color: var(--fg);
         font: 16px/1.5 ui-sans-serif, system-ui, -apple-system, "PingFang SC", sans-serif; }}
  form {{ width: min(320px, calc(100vw - 32px)); display: flex; flex-direction: column; gap: 12px; }}
  h1 {{ font-size: 20px; margin: 0; }}
  p {{ margin: 0; color: var(--muted); font-size: 14px; }}
  input, button {{ font: inherit; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--line);
                  background: var(--bg); color: var(--fg); }}
  button {{ background: var(--fg); color: var(--bg); border-color: var(--fg); font-weight: 600; }}
  .err {{ color: var(--mark); }}
</style></head>
<body><form method="post" action="/login">
  <h1>ESP32 iPhone Agent</h1>
  <p>输入访问口令 · Enter the password</p>
  <input type="password" name="token" autocomplete="current-password" autofocus required>
  <button type="submit">进入 · Open</button>
  <p class="err">{error}</p>
</form></body></html>"""


def cookie_value(token: str) -> str:
    """What the cookie holds: derived from the password, never the password itself."""
    return hmac.new(token.encode(), b"hidagent web console", hashlib.sha256).hexdigest()


def install(app: FastAPI, token: str) -> None:
    """Guard every page and API call of `app` with `token`; no-op when the token is empty."""
    if not token:
        return
    expected = cookie_value(token).encode()
    fails: dict[str, tuple[int, float]] = {}  # address -> (wrong tries in a row, locked until)

    def allowed(request: Request) -> bool:
        if (request.client.host if request.client else "") in LOOPBACK:
            return True
        if hmac.compare_digest(request.cookies.get(COOKIE, "").encode(), expected):
            return True
        auth = request.headers.get("authorization", "")
        return auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].encode(), token.encode())

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path == "/login" or allowed(request):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "login required"}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page():
        return LOGIN_HTML.format(error="")

    @app.post("/login")
    async def login(request: Request):
        who = request.client.host if request.client else ""
        count, until = fails.get(who, (0, 0.0))
        if until > time.time():
            return HTMLResponse(LOGIN_HTML.format(error="错误太多次，请 1 分钟后再试 · Too many tries, wait a minute"),
                                status_code=429)
        given = parse_qs((await request.body()).decode(errors="replace")).get("token", [""])[0]
        if not hmac.compare_digest(given.encode(), token.encode()):
            count += 1
            fails[who] = (0, time.time() + LOCKOUT_S) if count >= MAX_FAILS else (count, 0.0)
            return HTMLResponse(LOGIN_HTML.format(error="口令不对 · Wrong password"), status_code=401)
        fails.pop(who, None)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(COOKIE, expected.decode(), max_age=COOKIE_DAYS * 86400, httponly=True, samesite="strict")
        return response
