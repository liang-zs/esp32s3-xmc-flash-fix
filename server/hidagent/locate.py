"""One quick question about the current frame: where is a thing, is the button here, is this the Home Screen."""

import re
import time

from . import vision
from .prompts import AT_BUTTON_AT_Q, EXACT_TEXT, GRID_NOTE, IS_HOME_PROMPT, IS_HOME_Q, LOCATE_PROMPT


def last_line(reply: str) -> str:
    """The answer line of a reply (models sometimes reason first): its last non-empty line, lowercased."""
    lines = [x.strip() for x in reply.strip().splitlines() if x.strip()]
    return lines[-1].lower() if lines else ""


class LocateMixin:
    """Part of Agent (see agent.py): uses its screen, actor, usage, trace and grid."""

    async def _quick(self, system: str, question: str, image: bytes | None, timing: str = "") -> str:
        """One short question to the actor model, on its fast settings when FAST_BODY is configured. Counts the call
        and its tokens in this run's usage (and its seconds under `timing`)."""
        started = time.time()
        reply = await (self.actor.chat(system, question, image, max_tokens=300, fast=True)
                       if getattr(self.actor, "fast_extra", None) else self.actor.chat(system, question, image))
        self.usage["calls"] = self.usage.get("calls", 0) + 1
        if timing:
            self.usage[timing] = round(self.usage.get(timing, 0) + time.time() - started, 2)
        for k, v in (getattr(self.actor, "last_usage", {}) or {}).items():
            self.usage[k] = self.usage.get(k, 0) + v
        return reply

    async def _locate(self, question: str, by_text: bool = False) -> tuple[int, int] | None:
        """One fast question about the current frame: where is <thing>? (0-1000), None if it is not there.
        by_text: the thing is named by its visible text, which must match exactly."""
        frame = self.screen.frame
        image = vision.grid(frame) if self.grid else frame
        system = LOCATE_PROMPT + (EXACT_TEXT if by_text else "") + (GRID_NOTE if self.grid else "")
        text = question + " If there is none, answer: none. Otherwise answer with one line: X Y"
        reply = await self._quick(system, text, image, timing="locate_s")
        last = last_line(reply)
        nums = re.findall(r"\d+(?:\.\d+)?", last)
        spot = None
        if "none" not in last and len(nums) >= 2:
            x, y = round(float(nums[-2])), round(float(nums[-1]))
            spot = (x, y) if 0 <= x <= 1000 and 0 <= y <= 1000 else None
        # the frame it looked at: iPhone 11, 2026-11-18 a tapon went to 'rednote' although the actor's screenshot
        # showed Notes; the layout had likely changed since (Spotlight results load late)
        self.trace.write(frame, kind="locate", question=question, reply=reply[-120:], spot=spot)
        return spot

    async def _button_at(self, spot: tuple[int, int]) -> bool:
        """Is the AssistiveTouch button at `spot`? One yes/no question. Offline: yes 8/8 where it was (keyboard up,
        Safari toolbar), no 3/3 on full-screen video frames (where a double tap 'likes' a bilibili video)."""
        frame = self.screen.frame
        image = vision.grid(frame) if self.grid else frame
        q = AT_BUTTON_AT_Q.format(x=spot[0], y=spot[1])
        reply = await self._quick(IS_HOME_PROMPT + (GRID_NOTE if self.grid else ""), q, image)
        yes = last_line(reply).startswith("yes")
        self.trace.write(frame, kind="locate", question=q, reply=reply[-40:], spot=list(spot) if yes else None)
        return yes

    async def _is_home_screen(self) -> bool:
        """One fast yes/no question about the current frame: is this the Home Screen?"""
        frame = self.screen.frame
        reply = await self._quick(IS_HOME_PROMPT, IS_HOME_Q, frame)
        yes = last_line(reply).startswith("yes")
        self.trace.write(frame, kind="locate", question=IS_HOME_Q, reply=reply[-80:], spot=None, home=yes)
        return yes
