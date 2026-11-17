import io

import pytest
from PIL import Image, ImageDraw

from hidagent.vision import changed, pointer

W, H = 498, 1080  # a mirrored iPhone frame


def screen(pointer_at=None, button=False) -> bytes:
    """A synthetic home screen: gradient background, optional button and pointer dot."""
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    for y in range(H):
        draw.line([(0, y), (W, y)], fill=(40, 60 + y // 10, 120))
    if button:
        draw.rectangle([150, 400, 350, 600], fill=(250, 250, 250))
    if pointer_at:
        x, y = pointer_at
        draw.ellipse([x - 12, y - 12, x + 12, y + 12], fill=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return buf.getvalue()


def test_pointer_move_is_not_a_screen_change():
    assert not changed(screen((260, 856)), screen((272, 856)), ignore=((260 / W, 856 / H), (272 / W, 856 / H)))


def test_new_content_is_a_screen_change():
    assert changed(screen((260, 856)), screen((260, 856), button=True))


def test_missing_frame_counts_as_changed():
    assert changed(None, screen())


def test_pointer_is_found_from_a_nudge():
    shift = 12 / W
    found = pointer(screen((262, 856)), screen((274, 856)), near=(0.5, 0.8), shift=shift)
    assert found is not None
    assert found[0] * W == pytest.approx(262, abs=3)
    assert found[1] * H == pytest.approx(856, abs=3)


def test_pointer_not_found_when_nothing_moved():
    frame = screen((262, 856))
    assert pointer(frame, frame, near=(0.5, 0.8), shift=12 / W) is None


def checkmark_screen(checked: bool, pointer_at=(100, 500)) -> bytes:
    img = Image.open(io.BytesIO(screen(pointer_at))).convert("RGB")
    if checked:
        ImageDraw.Draw(img).line([(430, 495), (438, 505), (452, 485)], fill=(255, 149, 0), width=4)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return buf.getvalue()


def test_small_checkmark_counts_as_a_change():
    assert changed(checkmark_screen(False), checkmark_screen(True))


def test_masked_pointer_move_is_not_a_change():
    before, after = checkmark_screen(False, (100, 500)), checkmark_screen(False, (300, 500))
    assert not changed(before, after, ignore=((100 / W, 500 / H), (300 / W, 500 / H)))


def test_pointer_far_from_the_target_is_found_with_a_full_screen_search():
    before, after = screen((354, 136)), screen((366, 136))
    assert pointer(before, after, near=(0.9, 0.1), shift=12 / W) is None  # outside the window near the target
    found = pointer(before, after, near=(0.5, 0.5), shift=12 / W, reach=(0.5, 0.5))
    assert found[0] * W == pytest.approx(354, abs=3) and found[1] * H == pytest.approx(136, abs=3)


def clock_screen(pointer_at=None, hand=0) -> bytes:
    """Home screen with an animated icon: a second hand that ticks on its own (the Clock app icon)."""
    import math

    img = Image.open(io.BytesIO(screen(pointer_at))).convert("RGB")
    draw = ImageDraw.Draw(img)
    cx, cy, r = 190, 260, 35
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(245, 245, 245))
    a = math.radians(hand * 6 - 90)
    draw.line([(cx, cy), (cx + r * math.cos(a), cy + r * math.sin(a))], fill=(230, 80, 20), width=3)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return buf.getvalue()


def test_animated_icon_is_not_mistaken_for_the_pointer():
    # iPhone 11 run, 2026-11-17: the Clock icon's second hand was read as the pointer, so aim "corrected" taps
    shift = 12 / W
    after, back = clock_screen((412, 700), 1), clock_screen((400, 700), 2)  # the hand ticked between the two
    found = pointer(back, after, (0.5, 0.5), shift, reach=(0.5, 0.5))
    assert found == pytest.approx((400 / W, 700 / H), abs=0.01)


def test_animation_alone_is_not_a_screen_change():
    # seconds: the model looks at 0, we act at 2, the result is checked at 5
    seen, idle, after = clock_screen(hand=0), clock_screen(hand=2), clock_screen(hand=5)
    assert changed(seen, after)  # the second hand moved
    assert not changed(seen, after, idle=idle)  # ... but it was moving before we acted, too


def test_real_change_still_counts_next_to_an_animation():
    seen, idle = clock_screen(hand=0), clock_screen(hand=2)
    img = Image.open(io.BytesIO(clock_screen(hand=5))).convert("RGB")
    ImageDraw.Draw(img).rectangle([150, 700, 350, 800], fill=(250, 250, 250))  # e.g. a sheet slid up
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    assert changed(seen, buf.getvalue(), idle=idle)


def test_animation_alone_is_not_a_pointer():
    assert pointer(clock_screen(hand=0), clock_screen(hand=1), (190 / W, 260 / H), 12 / W) is None


def test_pointer_hidden_before_the_wiggle_is_still_found_close():
    # iOS hides a resting pointer: only one of the two frames may show it; the error is at most shift/2
    found = pointer(screen(), screen((274, 856)), near=(0.5, 0.8), shift=12 / W)
    assert found is not None and abs(found[0] * W - 262) <= 7 and abs(found[1] * H - 856) <= 3


def test_stream_noise_is_not_a_screen_change():
    # iPhone 11, 2026-11-17: AirPlay streams ~8 fps of near-identical frames, so "no new frame for a while"
    # never happened and every settle ran to its limit; settled() now looks at content.
    from PIL import Image, ImageDraw

    from hidagent.screen import _differs, _thumb

    def jpeg(dot=None, speck=False):
        im = Image.new("RGB", (400, 800), (40, 40, 40))
        d = ImageDraw.Draw(im)
        if speck:
            d.point([(10, 10), (300, 500)], fill=(90, 90, 90))  # a couple of noisy pixels
        if dot:
            d.ellipse([dot[0] - 12, dot[1] - 12, dot[0] + 12, dot[1] + 12], fill=(200, 200, 200))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=75)
        return buf.getvalue()

    base = _thumb(jpeg())
    assert not _differs(base, _thumb(jpeg(speck=True)))
    assert _differs(base, _thumb(jpeg(dot=(200, 400))))
    assert _differs(None, base)
