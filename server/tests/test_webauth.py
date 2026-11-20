from fastapi import FastAPI
from fastapi.testclient import TestClient

from hidagent import webauth


def make_app(token: str) -> FastAPI:
    app = FastAPI()
    webauth.install(app, token)

    @app.get("/")
    def index():
        return {"page": "console"}

    @app.get("/api/status")
    def status():
        return {"ok": True}

    return app


def test_no_token_changes_nothing():
    client = TestClient(make_app(""))
    assert client.get("/api/status").status_code == 200
    assert client.get("/login").status_code == 404


def test_other_machines_must_log_in_first():
    client = TestClient(make_app("s3cret"))  # TestClient's address is "testclient", not loopback
    assert client.get("/api/status").status_code == 401
    page = client.get("/", follow_redirects=False)
    assert page.status_code == 303 and page.headers["location"] == "/login"
    assert "password" in client.get("/login").text


def test_right_password_sets_a_cookie_that_opens_everything():
    client = TestClient(make_app("s3cret"))
    r = client.post("/login", data={"token": "s3cret"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert "s3cret" not in r.headers["set-cookie"]  # the cookie never holds the password itself
    assert client.get("/api/status").json() == {"ok": True}
    assert client.get("/").json() == {"page": "console"}


def test_wrong_password_and_lockout():
    client = TestClient(make_app("s3cret"))
    for _ in range(webauth.MAX_FAILS):
        assert client.post("/login", data={"token": "nope"}).status_code == 401
    assert client.post("/login", data={"token": "s3cret"}).status_code == 429  # locked for a minute
    assert client.get("/api/status").status_code == 401


def test_bearer_header_for_scripts():
    client = TestClient(make_app("s3cret"))
    assert client.get("/api/status", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/api/status", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_this_mac_itself_needs_no_password():
    client = TestClient(make_app("s3cret"), client=("127.0.0.1", 50000))
    assert client.get("/api/status").status_code == 200
