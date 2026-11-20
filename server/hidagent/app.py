"""Web API + control page. Run with `python -m hidagent`."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import webauth
from .agent import Agent
from .debug import Trace
from .device import Device, DeviceError
from .llm import LLM, price_at
from .phone import Phone
from .screen import Screen

DATA_DIR = Path(os.environ.get("HIDAGENT_DATA", Path.home() / ".hidagent"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEBUG = os.environ.get("HIDAGENT_DEBUG") == "1"

device = Device(key=os.environ.get("LINK_KEY", ""))
screen = Screen(os.environ.get("AIRPLAY_NAME", "iPhone Agent"))
phone = Phone(device, DATA_DIR / "calibration.json", screen)
agent = Agent(phone, screen, LLM.from_env("ACTOR"), LLM.from_env("PLANNER"),
              Trace(DATA_DIR / "traces" if DEBUG else None),
              runs_file=DATA_DIR / "runs.jsonl",
              price=lambda when: price_at(when, os.environ.get("LLM_PRICE", ""), os.environ.get("LLM_PRICE_IDLE", ""),
                                          os.environ.get("LLM_PEAK_HOURS", "9-12,14-18")),
              grid=os.environ.get("ACTOR_GRID") == "1",
              at_double_tap_home=os.environ.get("AT_DOUBLE_TAP_HOME") == "1")


@asynccontextmanager
async def lifespan(_):
    tcp_port = int(os.environ.get("DEVICE_PORT", 7300))
    server = await device.serve("0.0.0.0", tcp_port)
    # without LINK_KEY the board does not trust beacons (anyone can send one), so there is nothing to announce
    found = asyncio.create_task(device.beacon(tcp_port)) if device.key else None
    screen.start()
    awake = asyncio.create_task(device.keep_awake(screen=screen))
    # board on a USB serial port (a path, or "auto"): no network needed between the Mac and the board
    serial = os.environ.get("DEVICE_SERIAL")
    cable = asyncio.create_task(device.serial_loop(serial)) if serial else None
    yield
    awake.cancel()
    if cable:
        cable.cancel()
    if found:
        found.cancel()
    device.close()
    screen.stop()
    server.close()


app = FastAPI(lifespan=lifespan)
webauth.install(app, os.environ.get("WEB_TOKEN", ""))  # optional console password
class Static(StaticFiles):
    """Static files that browsers re-check on every load, so a new version shows up on refresh."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", Static(directory=Path(__file__).parent / "static"), name="static")


class Point(BaseModel):
    x: float
    y: float
    aim: bool = False  # tap: find the pointer on screen and correct before clicking, like the agent does


class Swipe(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Scroll(Point):
    lines: int


class Text(BaseModel):
    text: str


async def act(coro):
    try:
        await coro
    except (DeviceError, RuntimeError) as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/status")
def status():
    return {"device": device.connected, "board": device.info(), "screen": screen.live, "calibrated": phone.calibrated,
            "debug": DEBUG, "trace": str(agent.trace.dir or ""), "aspect": round(screen.aspect, 4),
            "models": {"planner": agent.planner.model, "actor": agent.actor.model},
            "agent": {"running": agent.running, "task": agent.task,
                      "state": agent.task_ctx.snapshot(), "steps": agent.steps,
                      "progress": {"completed": agent.completed, "plan_step": agent.plan_step},
                      "usage": agent.usage}}


@app.get("/api/runs")
def runs(last: int = 50):
    """Recent run summaries plus totals: success rate, average time and cost per run."""
    path = DATA_DIR / "runs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    rows = rows[-last:]
    ok = [r for r in rows if r.get("result") == "done"]
    costs = [r["cost"] for r in rows if r.get("cost") is not None]
    return {"runs": rows, "total": len(rows), "success": len(ok),
            "success_rate": round(len(ok) / len(rows), 3) if rows else None,
            "avg_seconds_ok": round(sum(r["seconds"] for r in ok) / len(ok), 1) if ok else None,
            "avg_tokens": round(sum(r.get("prompt", 0) + r.get("completion", 0) for r in rows) / len(rows))
            if rows else None,
            "total_cost": round(sum(costs), 4) if costs else None,
            "think": _think_totals(rows)}


def _think_totals(rows: list[dict]) -> dict:
    """Actor calls that thought instead of answering fast, by reason, over these runs: count and seconds."""
    out: dict = {}
    for r in rows:
        for k, v in r.items():
            if k.startswith("think_") and isinstance(v, (int, float)):
                out[k[6:]] = round(out.get(k[6:], 0) + v, 2)
    return out


@app.get("/api/frame.jpg")
def frame():
    if not screen.live:
        raise HTTPException(404, "no live screen: start screen mirroring on the phone")
    return Response(screen.frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/steps/{index}.jpg")
def step_shot(index: int):
    if index not in agent.shots:
        raise HTTPException(404)
    return Response(agent.shots[index], media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/tap")
async def tap(p: Point):
    return await act(phone.tap(p.x, p.y, aim=p.aim))


@app.post("/api/swipe")
async def swipe(s: Swipe):
    return await act(phone.swipe(s.x1, s.y1, s.x2, s.y2))


@app.post("/api/scroll")
async def scroll(s: Scroll):
    return await act(phone.scroll(s.x, s.y, s.lines))


@app.post("/api/type")
async def type_text(t: Text):
    return await act(phone.type(t.text))


@app.post("/api/key")
async def key(t: Text):
    return await act(phone.key(t.text))


@app.post("/api/home")
async def home():
    return await act(phone.go_home())


@app.post("/api/back")
async def back():
    return await act(phone.go_back())


@app.post("/api/raw")
async def raw(t: Text):
    """Debug only: send one protocol line to the board as-is."""
    if not DEBUG:
        raise HTTPException(404)
    return await act(device.send(t.text))


@app.post("/api/calibrate/start")
async def calibrate_start():
    """Calibrate the pointer. {"auto": true}: done by itself; otherwise the user clicks the pointer next."""
    try:
        return {"ok": True, "auto": await phone.start_calibration(), "span": phone.span}
    except (DeviceError, RuntimeError) as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/calibrate/finish")
def calibrate_finish(p: Point):
    phone.finish_calibration(p.x, p.y)
    return {"ok": True, "span": phone.span}


@app.post("/api/agent")
async def agent_start(t: Text):
    try:
        agent.start(t.text)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True}


@app.post("/api/agent/home")
async def agent_home():
    """The agent's own home action (AssistiveTouch), e.g. to start a regression run from the Home Screen."""
    try:
        taps = await agent.go_home()
    except (DeviceError, RuntimeError, ValueError) as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "taps": taps}


@app.post("/api/agent/stop")
async def agent_stop():
    """Emergency stop: cancel the agent and release the mouse button."""
    agent.stop()
    await asyncio.sleep(0.1)  # let a gesture in flight finish its own release
    return await act(phone.reset()) if device.connected else {"ok": True}


@app.post("/api/agent/clear")
def agent_clear():
    """Forget the last run (steps, screenshots, plan) so the page starts clean."""
    try:
        agent.clear()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True}

class Locate(BaseModel):
    x: float
    y: float
    fix: bool = True
    aim: bool = False


@app.post("/api/debug/locate")
async def debug_locate(p: Locate):
    """Debug only: move the pointer to (x, y) without clicking and report where it really landed."""
    if not DEBUG:
        raise HTTPException(404)
    if agent.running:
        raise HTTPException(409, "stop agent before testing")
    try:
        found = await phone.locate(p.x, p.y, p.fix, p.aim)
    except (DeviceError, RuntimeError) as e:
        raise HTTPException(409, str(e)) from e
    return {"target": [p.x, p.y], "found": list(found) if found else None}


@app.post("/api/debug/pointer-frames")
async def debug_pointer_frames():
    """测试鼠标移动后，AirPlay 镜像截图的更新时间。"""

    if not DEBUG:
        raise HTTPException(404, "debug mode disabled")

    if agent.running:
        raise HTTPException(409, "stop agent before testing")

    if not device.connected:
        raise HTTPException(409, "ESP32-S3 not connected")

    if not screen.live:
        raise HTTPException(409, "iPhone screen not connected")

    try:
        await phone.debug_pointer_frames()
    except (DeviceError, RuntimeError) as e:
        raise HTTPException(409, str(e)) from e

    return {"ok": True, "message": "FRAME-TEST completed"}
