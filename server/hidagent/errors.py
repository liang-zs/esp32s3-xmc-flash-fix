"""Exceptions the agent loop catches."""


class GiveUp(Exception):
    """The task is about to fail; _attempt asks the user (outside the step's time limit) before it does."""

    def __init__(self, reason: str, shot: bytes | None = None):
        super().__init__(reason)
        self.reason, self.shot = reason, shot


class BlindError(RuntimeError):
    """The screen feed is gone; acting now would be guessing."""


class LandscapeError(BlindError):
    """The phone shows a landscape screen (e.g. a full-screen video). Pointer calibration, the AssistiveTouch spot
    and every coordinate assume portrait; iPhone 11, 2026-11-19: a bilibili video went full screen in landscape,
    the AssistiveTouch button was hidden and two tasks double-tapped the video (likely 'liking' it) until they
    failed. Stop and ask for portrait instead."""


class NotFound(ValueError):
    """tapon / press found nothing by that name on the screen. Mild: the actor is told and looks again (another
    label, coordinates for an icon without text, scroll or wait) instead of a whole new plan. 2026-11-18..21: all 23
    not-found tapons led to a replan; offline, a looser second question found three toolbar icons but also pointed
    at the AssistiveTouch button and a Safari suggestion for absent things, so the locator itself stays strict."""
