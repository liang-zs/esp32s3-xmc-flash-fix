"""Going to the Home Screen through the AssistiveTouch button, and remembering where that button sits."""

import asyncio
import json
import logging
import math
import time
from pathlib import Path

from . import vision
from .errors import LandscapeError
from .prompts import AT_BUTTON_Q, AT_BUTTON_Q2, AT_HOME_Q, AT_KEYBOARD

log = logging.getLogger(__name__)

AT_AVOID = 70  # a coordinate tap this close to the AssistiveTouch button (in screen-width units) is refused
POINTER_PARK = (0.5, 0.5)  # where the pointer waits while the AssistiveTouch button is looked for: never at an edge


class HomeMixin:
    """Part of Agent (see agent.py): uses its phone, screen, actor, usage, trace and grid."""

    async def go_home(self) -> list[tuple[int, int]]:
        """The home action outside a task (e.g. before a regression run starts)."""
        if self.running:
            raise RuntimeError("agent is running")
        if self.screen.aspect > 1:
            raise LandscapeError("the phone screen is landscape (a full-screen video?): turn it back to portrait")
        return await self._home()

    async def _home(self) -> list[tuple[int, int]]:
        """Home Screen in one action: tap the AssistiveTouch button, then Home in its menu, each located on the
        current frame. iPhone 11, 2026-11-18: going home took the actor 2-4 rounds (button, look, Home, look,
        misses in between); cmd+h does nothing on iPhone. Offline: button found 8/8, menu Home 8/8 (the button
        moves between runs, so it is never a fixed position). No button: the old bottom-edge swipe."""
        known = self._known_at_button() if self.at_double_tap_home else None
        if known and await self._button_at(known):
            # fast path (iPhone 11: the button stayed at 750 947 for days, keyboard up or not): no pointer parking,
            # no two locate questions. A double tap that does not reach the Home Screen falls back to the full search.
            try:
                return await self._home_by_double_tap(known)
            except ValueError as e:
                log.warning("[HOME] remembered spot %s did not work (%s); looking for the button", known, e)
        button = await self._locate_at_button()
        if button is None:  # the menu may be open already (it hides the button)
            item = await self._locate(AT_HOME_Q)
            if item is not None:
                await self._tap_until_changed(item)
                log.info("[HOME] AssistiveTouch menu was open; Home at %s", item)
                return [item]
            # no bottom-edge swipe: it is unreliable here, and with the keyboard up iOS's slide-to-type read it as
            # the word "by" (iPhone 11, 2026-11-19: "by by by" in Safari's search field after failed homes)
            raise ValueError("home: no AssistiveTouch button found on screen")
        if self.at_double_tap_home:
            return await self._home_by_double_tap(button)
        item = None
        for attempt in range(2):  # iPhone 11, 2026-11-18: the menu did not show up after the first tap twice
            await self.phone.tap(button[0] / 1000, button[1] / 1000, aim=True)
            for wait in (0.5, 1.0):  # it may still be opening
                await self.screen.settled(quiet=0.2, least=wait, most=1.5)
                item = await self._locate(AT_HOME_Q)
                if item is not None:
                    break
            if item is not None:
                break
            log.warning("[HOME] AssistiveTouch menu did not open after tap %d at %s", attempt + 1, button)
        if item is None:
            raise ValueError("home: the AssistiveTouch menu did not open (no Home item on screen)")
        await self._tap_until_changed(item)
        log.info("[HOME] AssistiveTouch at %s, Home at %s", button, item)
        self._remember_at_button(button)
        return [button, item]

    async def _locate_at_button(self) -> tuple[int, int] | None:
        """The AssistiveTouch button. The mouse pointer is parked in the middle of the screen first (a move, no
        click): the button always sits at an edge, so an answer there is the pointer. Both wordings are asked in
        parallel and the first answer away from the pointer wins. iPhone 11, 2026-11-18: right after a restart the
        pointer's place was unknown and the locator's (500, 600), the pointer, was double-tapped twice."""
        await self.phone.point(*POINTER_PARK)
        self._pointer = POINTER_PARK
        await self.screen.settled(quiet=0.2, least=0.4, most=1.0)
        found = await asyncio.gather(self._locate(AT_BUTTON_Q2 + AT_KEYBOARD), self._locate(AT_BUTTON_Q + AT_KEYBOARD))
        park = (POINTER_PARK[0] * 1000, POINTER_PARK[1] * 1000)
        for spot in found:
            if spot is not None and math.dist(spot, park) > 120:
                return spot
        known = self._known_at_button()
        log.warning("[HOME] AssistiveTouch button not found (answers %s; the pointer is parked at %s); %s", found,
                    park, f"using where it was last time, {known}" if known else "no earlier position")
        return known

    def _at_file(self) -> Path | None:
        return self.runs_file.parent / "assistivetouch.json" if self.runs_file else None

    def _known_at_button(self) -> tuple[int, int] | None:
        """Where the button was when home last worked (it stays put: iPhone 11, 750 945-950 for days, keyboard or
        not). Used when locating fails, e.g. the button over the keyboard or over Safari's toolbar."""
        f = self._at_file()
        try:
            x, y = json.loads(f.read_text())["button"] if f and f.exists() else (None, None)
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return (int(x), int(y)) if x is not None else None

    def _on_at_button(self, acts: list[dict]) -> tuple[int, int, int, int] | None:
        """A coordinate tap that would land on the AssistiveTouch button (at its remembered spot): (x, y, bx, by).
        iPhone 11, 2026-11-20: "tap 690 940" for Freeform's pen tool opened the AssistiveTouch menu instead."""
        button = self._known_at_button()
        if not button:
            return None
        aspect = self.screen.aspect or 0.46
        for a in acts:
            if a.get("action") == "tap":
                # distance in width units: y counts in units of the height, which is 1/aspect times longer
                dx, dy = a["x"] - button[0], (a["y"] - button[1]) / aspect
                if math.hypot(dx, dy) < AT_AVOID:
                    return int(a["x"]), int(a["y"]), *button
        return None

    def _remember_at_button(self, spot: tuple[int, int]) -> None:
        f = self._at_file()
        if f:
            try:
                f.write_text(json.dumps({"button": list(spot), "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
            except OSError as e:
                log.warning("could not save %s: %s", f, e)

    async def _home_by_double_tap(self, button: tuple[int, int]) -> list[tuple[int, int]]:
        """The phone's AssistiveTouch Double-Tap action is Home: double-tap the button in place (one located spot,
        no menu). Taken as a single tap, it opens the menu (Single-Tap stays Open Menu): then tap Home there."""
        spot = (button[0] / 1000, button[1] / 1000)
        for attempt in range(2):
            before = self.screen.frame
            await self.phone.double_tap(*spot, aim=True)
            await self.screen.settled(quiet=0.3, least=0.8, most=2.0)  # iOS waits out the double-tap timeout
            # both questions at once: did the menu open (taken as a single tap), or is this the Home Screen
            item, home = await asyncio.gather(self._locate(AT_HOME_Q), self._is_home_screen())
            if item is not None:  # counted as a single tap: the menu opened
                log.warning("[HOME] double tap opened the menu; tapping its Home at %s", item)
                await self._tap_until_changed(item)
                self._remember_at_button(button)
                return [button, item]
            # "the screen changed" is no proof: a playing video changes all the time (iPhone 11, 2026-11-19)
            if home and vision.changed(before, self.screen.frame, ignore=(spot,)):
                log.info("[HOME] double tap on AssistiveTouch at %s", button)
                self._remember_at_button(button)
                return [button]
            log.warning("[HOME] double tap %d at %s changed nothing", attempt + 1, button)
        raise ValueError("home: double-tapping AssistiveTouch changed nothing")

    async def _tap_until_changed(self, spot: tuple[int, int], tries: int = 2) -> bool:
        """Tap, wait for the screen, and tap once more if nothing changed. iPhone 11, 2026-11-18: the first tap
        on the AssistiveTouch menu's Home (located right) did nothing, probably while the menu still opened."""
        for _ in range(tries):
            before = self.screen.frame
            await self.phone.tap(spot[0] / 1000, spot[1] / 1000, aim=True)
            await self.screen.settled(quiet=0.3, least=0.6, most=2.0)
            if vision.changed(before, self.screen.frame, ignore=((spot[0] / 1000, spot[1] / 1000),)):
                return True
            log.warning("[HOME] tap at %s changed nothing; tapping again", spot)
        return False
