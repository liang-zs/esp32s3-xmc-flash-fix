"""Typing through on-screen keyboards: switch to Pinyin and back by looking, and pressing keypad keys by label."""

import asyncio
import logging
import time

from . import vision
from .errors import NotFound
from .locate import last_line
from .prompts import KEYBOARD_Q, ONE_WORD_PROMPT

log = logging.getLogger(__name__)

KEYBOARD_SWITCH = 1.2  # seconds after ctrl+space before looking: by hand, 1.5 s worked and 0.4 s did not
PORTRAIT_WAIT = 2.0  # a landscape frame mid-task can be a mirroring glitch: wait this long for portrait to return


class KeyboardMixin:
    """Part of Agent (see agent.py): uses its phone, screen, actor, usage, trace and grid."""

    async def _press_key(self, label: str, x: int, y: int, aim: bool = True) -> bool:
        """Tap one keypad key; True when the screen changed (a digit shows up on the display)."""
        before, started = self.screen.frame, time.time()
        await self.phone.tap(x / 1000, y / 1000, aim=aim)
        await self.screen.settled(quiet=0.15, least=0.3, most=1.0)
        hit = vision.changed(before, self.screen.frame)
        log.info("[PRESS] %s at (%d, %d): %s in %.2f s", label, x, y, "screen changed" if hit else "NO CHANGE",
                 time.time() - started)
        return hit

    async def _keyboard(self, recheck: bool = True) -> str:
        """Which on-screen keyboard shows: pinyin, english, other or none (one fast question on the frame).
        bilibili-search, 2026-11-20: the question got a 1920x1080 frame (a mirroring glitch while bilibili showed a
        landscape overlay), answered none, and the blind fallback typed Latin 'changanguniang'. So: look at a
        portrait frame only, and ask once more before believing 'none'."""
        waited = 0.0
        while self.screen.aspect > 1 and waited < PORTRAIT_WAIT:
            await asyncio.sleep(0.2)
            waited += 0.2
        frame = self.screen.frame
        reply = await self._quick(ONE_WORD_PROMPT, KEYBOARD_Q, frame)
        last = last_line(reply)
        kind = ("pinyin" if any(w in last for w in ("pinyin", "空格", "选定")) else "english" if "english" in last
                else "none" if "none" in last else "other")
        self.trace.write(frame, kind="locate", question=KEYBOARD_Q, reply=reply[-60:], spot=None, keyboard=kind)
        if kind == "none" and recheck:
            await asyncio.sleep(0.5)
            return await self._keyboard(recheck=False)
        return kind

    async def _switch_keyboard(self, want: str, tries: int = 3) -> str:
        """ctrl+space until the on-screen keyboard is `want`; returns the keyboard it ended on."""
        kind = await self._keyboard()
        for _ in range(tries):
            if kind in (want, "none"):
                break
            await self.phone.key("ctrl+space")
            await asyncio.sleep(KEYBOARD_SWITCH)  # the input-source picker shows for a moment
            await self.screen.settled(quiet=0.3, least=0.3, most=1.5)
            kind = await self._keyboard()
        return kind

    async def _pinyin(self, letters: str) -> list[str]:
        """Chinese text: make sure the Simplified Pinyin keyboard is on (look, don't assume), type the letters, take
        the first candidate with space, and go back to English. iPhone 11, 2026-11-19: the blind version (ctrl+space,
        0.4 s, type) typed Latin 'ceshibiji' into Notes: the switch had not finished, or the keyboard was already
        Pinyin and ctrl+space switched it away. Done by hand with 1.5 s waits it gave 测试."""
        seen = [await self._switch_keyboard("pinyin")]
        if seen[0] == "none":  # no on-screen keyboard to look at: the old blind sequence
            await self.phone.pinyin(letters)
            return seen
        if seen[0] != "pinyin":
            raise ValueError(f"pinyin: could not switch to the Simplified Pinyin keyboard (it shows {seen[0]})")
        await self.phone.type(letters)
        await asyncio.sleep(0.8)  # let the candidate bar fill before taking the first one
        await self.phone.key("space")
        await asyncio.sleep(0.5)
        seen.append(await self._switch_keyboard("english"))
        if seen[-1] != "english":
            log.warning("[PINYIN] text entered, but the keyboard did not go back to English (%s)", seen[-1])
        return seen

    async def _locate_keys(self, labels: list[str]) -> list[tuple[int, int]]:
        """Positions (0-1000) of keypad keys named by label, one fast question per distinct label, in parallel.
        iPhone 11 Calculator, 2026-11-18: asked one key at a time the actor found 16/16 keys, but writing a
        7-tap sequence of coordinates itself it got 0/16 sequences right (1 2 8 came out as 1 2 3)."""
        distinct = list(dict.fromkeys(labels))
        spots = await asyncio.gather(*(self._locate(f"Where is the center of the key labeled '{k}'?", by_text=True)
                                       for k in distinct))
        missing = [k for k, p in zip(distinct, spots, strict=True) if p is None]
        if missing:
            raise NotFound(f"press: could not find key(s) {' '.join(missing)} on screen")
        found = dict(zip(distinct, spots, strict=True))
        log.info("[PRESS] %s", {k: found[k] for k in labels})
        return [found[k] for k in labels]
