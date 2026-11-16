"""Small image checks on screen frames: did the screen change, and where is the pointer."""

import io
import logging

from PIL import Image, ImageChops, ImageFilter


log = logging.getLogger(__name__)

def _gray(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("L")


def _noise(a: Image.Image, b: Image.Image, level: int, grow: int = 9) -> bytes:
    """Pixels that changed on their own between two frames (an animated icon such as the Clock's second hand),
    grown a little so the moving part's whole path is covered; nonzero = ignore."""
    moved = ImageChops.difference(a, b).point(lambda v: 255 if v > level else 0)
    return moved.filter(ImageFilter.MaxFilter(grow)).tobytes()


def changed(before: bytes | None, after: bytes | None, ignore: tuple = (), threshold: float = 1.0,
            idle: bytes | None = None) -> bool:
    """True when the screen content changed, including small things such as a checkmark.

    `ignore` lists screen points (fractions) where the pointer was or went: those spots are masked, so pointer
    moves never count. The status bar (clock) is ignored too. `idle` is a frame taken after `before` but
    before acting: whatever changed between those two moves by itself (animations) is ignored as well.
    """
    if not before or not after:
        return True
    a, b = _gray(before), _gray(after)
    if a.size != b.size:
        return True
    small = ImageChops.difference(a.resize((32, 64), Image.BILINEAR), b.resize((32, 64), Image.BILINEAR)).tobytes()
    if sum(small) / len(small) > threshold:  # a big change: new screen, scrolled list, keyboard
        return True
    w, h = a.size[0] // 2, a.size[1] // 2  # fine pass for small changes, at half resolution
    diff = ImageChops.difference(a.resize((w, h), Image.BILINEAR), b.resize((w, h), Image.BILINEAR)).tobytes()
    masks = [(px * w, py * h) for px, py in ignore]
    still = None
    if idle:
        c = _gray(idle)
        if c.size == a.size:
            still = _noise(a.resize((w, h), Image.BILINEAR), c.resize((w, h), Image.BILINEAR), 25, grow=21)
    hits = 0
    for i, v in enumerate(diff):
        if v > 25 and not (still and still[i]):
            x, y = i % w, i // w
            if y > 0.06 * h and all((x - mx) ** 2 + (y - my) ** 2 > (0.05 * w) ** 2 for mx, my in masks):
                hits += 1
    return hits >= 6


POINTER_SIZE = (0.025, 0.075)  # pointer dot diameter as a fraction of the screen width (iPhone: about 0.05)


def pointer(before: bytes, after: bytes, near: tuple[float, float], shift: float,
            reach: tuple[float, float] = (0.12, 0.08)) -> tuple[float, float] | None:
    """Where the pointer is in `before`, given it moved right by `shift` (fraction of width) in `after`.

    The changed pixels are grouped into blobs; the pointer is the one shaped like a dot that moved sideways:
    pointer-sized, about as tall as the dot, at most `shift` wider, and filled. That rules out other things
    that change on their own: the Clock icon's second hand (thin lines), the AssistiveTouch button fading
    (too big). The blob's centroid sits shift/2 to the right of the first spot. Of several dots, the one
    nearest `near`; None if there is none within `reach`.
    Both frames should show the pointer (iOS hides it when it rests): pass the frame after moving back
    as `before` rather than one taken while the pointer sat still.
    """
    a, b = _gray(before), _gray(after)
    if a.size != b.size:
        return None
    w, h = a.size
    rx, ry = reach
    left, top = max(0, int((near[0] - rx) * w)), max(int(0.05 * h), int((near[1] - ry) * h))  # skip the status bar
    box = (left, top, min(w, int((near[0] + rx) * w)), min(h, int((near[1] + ry) * h)))
    bw, bh = box[2] - box[0], box[3] - box[1]
    if bw <= 0 or bh <= 0:
        return None
    mask = ImageChops.difference(a.crop(box), b.crop(box)).point(lambda v: 255 if v > 30 else 0)
    hit = mask.tobytes()
    count = sum(1 for v in hit if v)
    if not 15 <= count <= len(hit) // 4:  # nothing moved, or much more than a pointer (a new screen)
        return None
    grown = mask.filter(ImageFilter.MaxFilter(5)).tobytes()  # join the two halves of a moved dot
    lo, hi = POINTER_SIZE[0] * w, POINTER_SIZE[1] * w
    slack, seen, dots = abs(shift) * w + 0.03 * w, bytearray(len(grown)), []
    for i in range(len(grown)):
        if not grown[i] or seen[i]:
            continue
        seen[i], stack, pts = 1, [i], []
        while stack:  # flood fill one blob
            j = stack.pop()
            if hit[j]:
                pts.append(j)
            x, y = j % bw, j // bw
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                k = ny * bw + nx
                if 0 <= nx < bw and 0 <= ny < bh and grown[k] and not seen[k]:
                    seen[k] = 1
                    stack.append(k)
        if len(pts) < 15:
            continue
        xs, ys = [j % bw for j in pts], [j // bw for j in pts]
        dw, dh = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        fill = len(pts) / (dw * dh)
        if lo <= dh <= hi and dh * 0.7 <= dw <= dh + slack and fill >= 0.3:
            cx, cy = sum(xs) / len(pts) + left, sum(ys) / len(pts) + top
            dots.append((cx / w - shift / 2, cy / h))
        else:
            log.debug("[POINTER] blob rejected: %dx%d px, fill %.2f", dw, dh, fill)
    if not dots:
        return None
    return min(dots, key=lambda d: (d[0] - near[0]) ** 2 + (d[1] - near[1]) ** 2)
