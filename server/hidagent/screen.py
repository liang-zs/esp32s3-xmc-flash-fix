"""The phone's screen, received over AirPlay mirroring by UxPlay, as JPEG frames."""

import asyncio
import io
import logging
import os
import subprocess
import threading
import time

from PIL import Image, ImageChops

log = logging.getLogger(__name__)

# AirPlay keeps streaming ~8 fps even when nothing moves, and each frame carries a little video noise
# (iPhone 11, 2026-11-17: at most 4 pixels of a quarter-size frame differ by more than 20 when idle; a pointer
# move changes 38+). A frame counts as a real change only above this many pixels.
CHANGE_PIXELS = 8


def _thumb(jpeg: bytes) -> Image.Image | None:
    """Quarter-size grayscale, decoded cheaply (the JPEG decoder scales while decoding)."""
    try:
        im = Image.open(io.BytesIO(jpeg))
        im.draft("L", (im.width // 4, im.height // 4))
        return im.convert("L")
    except OSError:
        return None


def _differs(a: Image.Image | None, b: Image.Image | None) -> bool:
    if a is None or b is None or a.size != b.size:
        return True
    return sum(1 for v in ImageChops.difference(a, b).tobytes() if v > 20) >= CHANGE_PIXELS


class Screen:
    """Runs UxPlay with a GStreamer sink that writes JPEGs into a pipe; keeps the latest one."""

    def __init__(self, name: str = "iPhone Agent", fps: int = 8, quality: int = 75):
        self.name, self.fps, self.quality = name, fps, quality
        self.frame: bytes | None = None
        self.updated = 0.0      # last frame received
        self.changed_at = 0.0   # last frame whose content really changed (not just stream noise)
        self._thumb = None
        self._proc = None
        self._clients_at, self._clients_n = 0.0, None
        self._stopping = False

    def start(self) -> None:
        threading.Thread(target=self._supervise, daemon=True).start()

    def stop(self) -> None:
        self._stopping = True
        if self._proc:
            self._proc.terminate()

    def _supervise(self) -> None:
        """Keep UxPlay running: it exits on some disconnects, so restart it every time."""
        while not self._stopping:
            read_fd, write_fd = os.pipe()
            sink = (f"videorate drop-only=true ! video/x-raw,framerate={self.fps}/1 ! "
                    f"jpegenc quality={self.quality} ! fdsink fd={write_fd}")
            self._proc = subprocess.Popen(
                ["uxplay", "-n", self.name, "-nh", "-a", "-reset", "0", "-vs", sink],
                pass_fds=(write_fd,))  # UxPlay logs to stdout: keep them in ours
            os.close(write_fd)
            log.info("AirPlay receiver '%s' started (pid %d)", self.name, self._proc.pid)
            self._read(read_fd)  # returns when UxPlay closes the pipe
            code = self._proc.wait()
            self.frame = None
            log.warning("AirPlay receiver exited (code %s), restarting", code)
            time.sleep(1)

    @property
    def live(self) -> bool:
        running = self._proc is not None and self._proc.poll() is None
        if not running or self.frame is None:
            return False
        clients = self._clients()
        if clients is not None:  # Linux: is a phone connected to UxPlay right now?
            return clients > 0
        return time.time() - self.updated < 90  # elsewhere: AirPlay sends no frames while the screen is still

    def _clients(self) -> int | None:
        """Established TCP connections of the UxPlay process (a mirroring phone keeps some open).

        Read from /proc, cached for a second; None where /proc is not available (e.g. macOS).
        """
        if time.time() - self._clients_at < 1:
            return self._clients_n
        try:
            fds = os.listdir(f"/proc/{self._proc.pid}/fd")
            inodes = set()
            for fd in fds:
                link = os.readlink(f"/proc/{self._proc.pid}/fd/{fd}")
                if link.startswith("socket:["):
                    inodes.add(link[8:-1])
            n = 0
            for table in ("/proc/net/tcp", "/proc/net/tcp6"):
                with open(table) as f:
                    for row in f.readlines()[1:]:
                        cols = row.split()
                        n += cols[3] == "01" and cols[9] in inodes  # 01 = ESTABLISHED
        except OSError:
            n = None
        self._clients_at, self._clients_n = time.time(), n
        return n

    @property
    def aspect(self) -> float:
        """Frame width / height, read from the JPEG header (portrait iPhone as a fallback)."""
        f = self.frame or b""
        i = f.find(b"\xff\xc0")  # baseline start-of-frame: ... height(2) width(2)
        if i < 0 or len(f) < i + 9:
            return 9 / 19.5
        return int.from_bytes(f[i + 7:i + 9], "big") / int.from_bytes(f[i + 5:i + 7], "big")

    async def settled(self, quiet: float = 0.4, least: float = 0.5, most: float = 4) -> None:
        """Wait until the screen content stops changing for `quiet` seconds (`least` .. `most`).

        Judged by content, not by frame arrival: the stream never goes quiet, so waiting for a gap between
        frames always ran to `most` (4 s after every action, 0.8 s per pointer check). `least` gives a tap time
        to start its animation, so a screen that has not begun to change is not taken as settled."""
        started = time.time()
        await asyncio.sleep(least)
        while time.time() - self.changed_at < quiet and time.time() - started < most:
            await asyncio.sleep(0.05)

    async def fresh(self, since: float, timeout: float = 0.8) -> bytes | None:
        """The first frame received after `since`, or None if none arrives in time."""
        deadline = time.time() + timeout
        while self.updated <= since:
            if time.time() > deadline:
                return None
            await asyncio.sleep(0.02)
        return self.frame

    def _read(self, fd: int) -> None:
        buf = b""
        with os.fdopen(fd, "rb", buffering=0) as pipe:
            while chunk := pipe.read(1 << 16):
                buf += chunk
                while (start := buf.find(b"\xff\xd8")) >= 0 and (end := buf.find(b"\xff\xd9", start)) >= 0:
                    frame = buf[start:end + 2]
                    thumb = _thumb(frame)
                    now = time.time()
                    if _differs(self._thumb, thumb):
                        self.changed_at = now
                    self._thumb = thumb
                    self.frame, self.updated = frame, now
                    buf = buf[end + 2:]
