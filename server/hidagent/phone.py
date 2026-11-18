"""Touch-like gestures on the phone, built from relative mouse moves.

The pointer is relative, so the server tracks where it is (in mouse units) and moves
from there; every few gestures it slams it into the top-left corner (`home`) again to
cancel drift. Coordinates are fractions of the screen (0..1, origin top-left), so they
work for any frame size.
"""

import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path

from . import vision
from .device import Device

log = logging.getLogger(__name__)

CALIBRATION_MOVE = 200  # mouse units moved diagonally for a manual (one point) calibration; stays on screen
CALIBRATION_STOPS = (150, 350)  # automatic calibration finds the pointer at these distances from the corner
REHOME_EVERY = 10       # gestures between re-homing
HOLD = 0.25             # seconds to rest before lifting, so iOS adds no momentum (the swipe stops where it ends)
IME_SWITCH = 0.4        # seconds for iOS to switch keyboards / show pinyin candidates
TAP_GAP = 0.15          # seconds to stay still after a click, so iOS sees a finished tap, not the start of a drag
NUDGE = 12              # mouse units the pointer is wiggled by to find it on screen
AIM_TOLERANCE = 0.008   # screen fraction; a bigger miss is corrected before pressing
AIM_ROUNDS = 3          # find-and-correct rounds before pressing anyway (each correction is a move that can miss too)
SPOT_WAIT = 0.55        # seconds after a wiggle before its frame is used (AirPlay shows it ~0.2-0.35 s later)
MAX_MISS = 0.06         # bigger misses are still corrected, but not learned from (they may be misreadings)


class Phone:
    def __init__(self, device: Device, calibration_file: Path, screen=None):
        self.device, self._file, self.screen = device, calibration_file, screen
        # mouse units needed to cross the full screen width / height
        saved = json.loads(calibration_file.read_text()) if calibration_file.exists() else None
        self.span = {"w": saved["w"], "h": saved["h"]} if saved else None
        self._pos = None  # pointer position in mouse units; None = unknown
        self._moves = 0
        self._link = 0  # device.links when the pointer was last homed
        # where the pointer really landed (seen on screen) for the screen fraction we aimed at, per axis;
        # a straight-line fit through them corrects every later move, so taps get better as the agent works
        self._seen = {axis: deque(map(tuple, (saved or {}).get("seen", {}).get(axis, [])), maxlen=12)
                      for axis in ("x", "y")}

    @property
    def calibrated(self) -> bool:
        return self.span is not None

    async def start_calibration(self) -> bool:
        """Find the pointer on screen at two known distances from the corner and fit the mapping from both.

        Two points give the scale and the offset, which one point cannot (iOS pointer motion is not exactly
        proportional). Returns True when it calibrated itself; otherwise the user clicks the pointer
        (finish_calibration).
        """
        self._pos = None
        await self.device.send("home")
        seen = []
        for units in CALIBRATION_STOPS:
            step = units - (seen[-1][0] if seen else 0)
            await self.device.send(f"move {step} {step}")
            found = await self._find_pointer() if self.screen is not None and self.screen.live else None
            if not found:
                break
            seen.append((units, *found))


        if len(seen) < 2:
            log.info("calibration: pointer not found on screen, waiting for a click")
            if not seen:
                await self.device.send("home")
                await self.device.send(f"move {CALIBRATION_MOVE} {CALIBRATION_MOVE}")
            return False
        (u1, x1, y1), (u2, x2, y2) = seen
        self.span = {"w": round((u2 - u1) / (x2 - x1)), "h": round((u2 - u1) / (y2 - y1))}
        for samples in self._seen.values():
            samples.clear()
        for units, x, y in seen:  # where each stop landed, so moves are corrected for the offset too
            self._seen["x"].append((units / self.span["w"], x))
            self._seen["y"].append((units / self.span["h"], y))
        self._save()
        log.info("calibration: pointer at %s, span %s", [(round(x, 3), round(y, 3)) for _, x, y in seen], self.span)
        return True

    async def _find_pointer(self) -> tuple[float, float] | None:
        """Wiggle the pointer and find it anywhere on screen (the span may still be unknown)."""
        await self.screen.settled(quiet=0.2, least=0.3, most=1.5)
        found = await self._spot(NUDGE, 0.025, None, wait=1.0)
        return found if found and 0.03 < found[0] < 0.97 and 0.03 < found[1] < 0.97 else None

    def finish_calibration(self, x: float, y: float) -> None:
        """x, y: where the pointer ended up after start_calibration() (one point, picked by the user)."""
        self.span = {"w": round(CALIBRATION_MOVE / x), "h": round(CALIBRATION_MOVE / y)}
        for samples in self._seen.values():
            samples.clear()
        self._save()

    def _save(self) -> None:
        """Calibration plus the observed landings, so the correction survives a restart."""
        self._file.write_text(json.dumps({**self.span, "seen": {k: list(v) for k, v in self._seen.items()}}))

    async def point(self, x: float, y: float, aim: bool = False) -> None:
        if not self.calibrated:
            raise RuntimeError("pointer not calibrated")
        aim = aim and self.screen is not None
        if aim:  # start from the corner: iOS pointer motion depends on the path (acceleration), so a precise
            self._moves = REHOME_EVERY  # move is only repeatable, and its error learnable, from a known start
        await self._move_to(self._dx(x), self._dy(y))
        if aim:
            await self._aim(x, y)

    async def locate(self, x: float, y: float, fix: bool = True, aim: bool = False) -> tuple[float, float] | None:
        """Debug: move to (x, y) without clicking and return where the pointer really is.
        fix=False skips the learned correction, to measure the raw calibration; aim=True measures after the
        closed-loop correction a precise tap does."""
        if not self.calibrated or self.screen is None:
            raise RuntimeError("pointer not calibrated or no screen")
        if aim:
            await self.point(x, y, aim=True)
            nudge = NUDGE if x < 0.9 else -NUDGE
            return await self._spot(nudge, nudge / self.span["w"], (x, y))
        tx = round(min(max(self._fix(x, "x") if fix else x, 0), 1) * self.span["w"])
        ty = round(min(max(self._fix(y, "y") if fix else y, 0), 1) * self.span["h"])
        self._moves = REHOME_EVERY  # from the corner, like a precise tap
        await self._move_to(tx, ty)
        nudge = NUDGE if x < 0.9 else -NUDGE
        return await self._spot(nudge, nudge / self.span["w"], (x, y))

    async def _spot(self, nudge: int, shift: float, near: tuple[float, float] | None,
                    wait: float = 0.6) -> tuple[float, float] | None:
        """Wiggle the pointer right and back and find it from the two frames after each move.

        Both moves just happened, so the pointer shows in both frames (iOS hides it once it rests, so a frame
        taken before the wiggle may have none). Each frame is taken once the screen settles after a move: on a
        screen with an animation a new frame arrives every second anyway, and that one may predate the move.
        near=None searches the whole screen.
        """
        frames, marks = [], []
        for step in (nudge, -nudge):
            sent = time.time()
            await self.device.send(f"move {step} 0")
            if not await self.screen.fresh(sent, wait):
                return None
            # a 12-unit wiggle is too small to count as a content change, so give the frame showing it time to come
            await self.screen.settled(quiet=0.15, least=SPOT_WAIT, most=0.8)
            marks.append(f"{time.time() - sent:.2f}")
            frames.append(self.screen.frame)
        log.debug("spot: wiggle took %s s", " + ".join(marks))
        after, back = frames
        if near is not None:
            found = vision.pointer(back, after, near, shift)
            if found:
                return found
        return vision.pointer(back, after, (0.5, 0.5), shift, reach=(0.5, 0.5))

    async def _aim(self, x: float, y: float) -> None:
        """Closed loop: find the real pointer on screen and fix a miss until it is on target (AIM_ROUNDS at most).

        The first landing, straight from the corner, is learned so later moves start closer.
        """
        nudge = NUDGE if x < 0.9 else -NUDGE  # stay clear of the screen edge
        for round_ in range(AIM_ROUNDS):
            found = await self._spot(nudge, nudge / self.span["w"], (x, y))  # near the target first, then anywhere
            if not found:
                log.warning("aim (%.3f, %.3f) round %d: pointer not found", x, y, round_ + 1)
                return
            fx, fy = found
            log.info("aim (%.3f, %.3f) round %d: pointer at (%.3f, %.3f), error (%+.3f, %+.3f)",
                     x, y, round_ + 1, fx, fy, fx - x, fy - y)
            if round_ == 0 and max(abs(fx - x), abs(fy - y)) <= MAX_MISS:  # learn normal drift, not a misreading
                self._seen["x"].append((self._fix(x, "x"), fx))
                self._seen["y"].append((self._fix(y, "y"), fy))
                self._save()
            if abs(fx - x) <= AIM_TOLERANCE and abs(fy - y) <= AIM_TOLERANCE:
                return
            dx, dy = round((x - fx) * self.span["w"]), round((y - fy) * self.span["h"])
            await self.device.send(f"move {dx} {dy}")
            self._pos = (self._pos[0] + dx, self._pos[1] + dy)

    def _fix(self, t: float, axis: str) -> float:
        """The fraction to aim at so the pointer lands on t, from the line through the observed landings."""
        pts = self._seen[axis]
        if len(pts) < 2:
            return t
        n = len(pts)
        mc, mf = sum(c for c, _ in pts) / n, sum(f for _, f in pts) / n
        var = sum((c - mc) ** 2 for c, _ in pts)
        a = sum((c - mc) * (f - mf) for c, f in pts) / var if var > 1e-4 else 1.0  # landed = a * aimed + b
        a = a if 0.85 < a < 1.15 else 1.0
        b = mf - a * mc
        return (t - b) / a if abs(b) < 0.1 else t

    async def _move_to(self, tx: int, ty: int, rehome: bool = True) -> None:
        if self._link != self.device.links:  # new link: a dropped gesture may have left the button down
            await self.device.send("release")
            self._pos, self._link = None, self.device.links
        if self._pos is None or (rehome and self._moves >= REHOME_EVERY):
            await self.device.send("home")
            self._pos, self._moves = (0, 0), 0
        x, y = self._pos
        self._pos = None  # unknown until the move succeeds
        await self.device.send(f"move {tx - x} {ty - y}")
        self._pos, self._moves = (tx, ty), self._moves + 1

    async def reset(self) -> None:
        """Let go of the button and re-home the pointer, whatever state a gesture left."""
        self._pos = None
        await self.device.send("release")
        await self.device.send("home")
        self._pos, self._moves = (0, 0), 0

    async def tap(self, x: float, y: float, aim: bool = False) -> None:
        """aim=True finds the pointer on screen first (slower); worth it when a tap just missed."""
        await self.point(x, y, aim)
        await self.device.send("click")
        await asyncio.sleep(TAP_GAP)

    async def swipe(self, x1: float, y1: float, x2: float, y2: float, hold: bool = True) -> None:
        await self.drag([(x1, y1), (x2, y2)], hold)

    async def drag(self, points: list[tuple[float, float]], hold: bool = True) -> None:
        """Press at the first point, move through the rest, release: one continuous stroke.

        hold=True rests at the end first, so lists and pickers stop exactly there instead of flinging on.
        System gestures (home, back) need the flick, so they pass hold=False.
        """
        await self.point(*points[0])
        await self.device.send("press")
        try:
            for x, y in points[1:]:
                await self._move_to(self._dx(x), self._dy(y), rehome=False)  # never home mid-drag
            if hold:
                await asyncio.sleep(HOLD)
        finally:
            await self.device.send("release")

    async def scroll(self, x: float, y: float, lines: int) -> None:
        await self.point(x, y)
        await self.device.send(f"scroll {lines} 0")

    async def type(self, text: str) -> None:
        if not text.isascii():  # the board's keyboard sends ASCII keys; anything else turns into garbage
            raise RuntimeError("type sends ASCII only; enter Chinese with pinyin()")
        await self.device.send(f"type {text}")

    async def pinyin(self, letters: str) -> None:
        """Enter Chinese through the iPhone's Pinyin keyboard: switch to it (ctrl+space), type the letters,
        take the first candidate (space), switch back. Assumes the English keyboard was active, and leaves it so.
        iPhone 11, 2026-11-17: "changanguniang" + space gave 长安姑娘 in bilibili search."""
        await self.device.send("key ctrl+space")
        await asyncio.sleep(IME_SWITCH)
        try:
            await self.device.send(f"type {letters}")
            await asyncio.sleep(IME_SWITCH)  # let the candidate bar catch up before picking
            await self.device.send("key space")
        finally:
            await asyncio.sleep(IME_SWITCH)
            await self.device.send("key ctrl+space")

    async def clear_field(self) -> None:
        """Empty the focused text field: select all and delete, then cmd+backspace (delete to the start of the
        line) for when select-all did not take. iPhone 11, 2026-11-18: Spotlight reopens with the last query and
        the agent spent up to 10 rounds on cmd+a / the x button, as the selection is hard to see."""
        for combo in ("cmd+a", "backspace", "cmd+backspace"):
            await self.device.send(f"key {combo}")

    async def key(self, combo: str) -> None:
        await self.device.send(f"key {combo}")

    async def go_home(self) -> None:
        await self.swipe(0.5, 0.995, 0.5, 0.6, hold=False)  # resting here would open the app switcher

    async def go_back(self) -> None:
        await self.swipe(0.005, 0.45, 0.6, 0.45, hold=False)

    def _dx(self, x: float) -> int:
        return round(min(max(self._fix(x, "x"), 0), 1) * self.span["w"])

    def _dy(self, y: float) -> int:
        return round(min(max(self._fix(y, "y"), 0), 1) * self.span["h"])
