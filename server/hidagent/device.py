"""Link to the HID bridge board (see firmware/hid_bridge): the board dials in over Wi-Fi (TCP), or sits on a
USB serial port. Same line protocol either way."""

import asyncio
import glob
import logging
import os
import termios
import time
import tty

log = logging.getLogger(__name__)

BAUD = 115200
# USB-UART chips on ESP32-S3 boards: CP210x, CH34x (Apple's driver or WCH's), FTDI
SERIAL_GLOBS = ["/dev/cu.usbserial*", "/dev/cu.wchusbserial*", "/dev/cu.SLAB_USBtoUART*", "/dev/cu.usbmodem*"]
PROBE_S = 3.0  # opening the port may reset the board: boot log, then "hello", well within this


class DeviceError(RuntimeError):
    pass


class SerialLink:
    """A tty as an asyncio line stream: readline() like a StreamReader, write/drain/close like a StreamWriter."""

    def __init__(self, path: str, baud: int = BAUD):
        self.path = path
        fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            tty.setraw(fd)
            attrs = termios.tcgetattr(fd)
            attrs[2] = (attrs[2] | termios.CLOCAL | termios.CREAD) & ~termios.HUPCL  # closing must not reset it
            attrs[4] = attrs[5] = getattr(termios, f"B{baud}")
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            termios.tcflush(fd, termios.TCIOFLUSH)
        except (OSError, termios.error):
            os.close(fd)
            raise
        self._fd, self._out, self._closed = fd, bytearray(), False
        self.reader = asyncio.StreamReader()
        self._loop = asyncio.get_running_loop()
        self._loop.add_reader(fd, self._readable)

    def _readable(self) -> None:
        try:
            data = os.read(self._fd, 4096)
        except BlockingIOError:
            return
        except OSError:  # unplugged
            data = b""
        if data:
            self.reader.feed_data(data)
        else:
            self.close()

    async def readline(self) -> bytes:
        return await self.reader.readline()

    def write(self, data: bytes) -> None:
        self._out += data

    async def drain(self) -> None:
        while self._out:
            if self._closed:
                raise ConnectionResetError("serial port closed")
            try:
                n = os.write(self._fd, self._out)
            except BlockingIOError:
                await asyncio.sleep(0.005)
                continue
            del self._out[:n]

    def is_closing(self) -> bool:
        return self._closed

    def get_extra_info(self, _name, default=None):
        return default

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.remove_reader(self._fd)
        os.close(self._fd)
        self.reader.feed_eof()


def serial_candidates(spec: str) -> list[str]:
    """DEVICE_SERIAL: a port path, or "auto" for every USB-UART port present."""
    if spec != "auto":
        return [spec] if os.path.exists(spec) else []
    return sorted({p for g in SERIAL_GLOBS for p in glob.glob(g)})


class Device:
    """Accepts one board connection and sends it commands, one at a time."""

    def __init__(self):
        self._reader = self._writer = None
        self._lock = asyncio.Lock()
        self._last = self._beat = 0.0
        self.links = 0  # bumped on every (re)connect, so gestures know to release and re-home
        # for /api/status: lets a dead link be told apart from a board that never connected
        self.peer, self.hello, self.since = "", "", 0.0  # current link: board address, greeting, connect time
        self.lost_at, self.lost_reason = 0.0, ""          # last time a link died, and why
        self.serial = False  # current link is the USB serial port (preferred: no network needed)

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def serve(self, host: str, port: int):
        return await asyncio.start_server(self._accept, host, port)

    async def _accept(self, reader, writer):
        if self.connected and self.serial:  # the board is on the cable; it stops dialing Wi-Fi once it notices
            writer.close()
            return
        # newer firmware greets with "hello <last reset reason> <uptime ms>": read it before any command
        try:
            hello = (await asyncio.wait_for(reader.readline(), 1)).decode().strip()
        except (OSError, asyncio.TimeoutError):
            hello = ""
        peer = writer.get_extra_info("peername")
        self._attach(reader, writer, hello, f"{peer[0]}:{peer[1]}" if peer else "", serial=False)

    def _attach(self, reader, writer, hello: str, peer: str, serial: bool) -> None:
        if self.connected:
            self._writer.close()
            self._lost("replaced by a new connection")
        self._reader, self._writer, self.serial = reader, writer, serial
        self.links += 1
        self._last = self._beat = self.since = time.time()  # fresh link: nothing to keep awake yet
        self.peer, self.hello = peer, hello
        log.info("board connected over %s %s %s", "serial" if serial else "wifi", peer, hello)

    async def serial_loop(self, spec: str, every: float = 2.0) -> None:
        """Keep a board on a USB serial port linked: find it, and find it again after a replug."""
        while True:
            if not (self.connected and self.serial):
                for path in serial_candidates(spec):
                    if await self._probe(path):
                        break
            await asyncio.sleep(every)

    async def _probe(self, path: str) -> bool:
        """Open a port and attach it if a board answers "hello"; anything else (boot log, silence) is skipped."""
        try:
            link = SerialLink(path)
        except (OSError, termios.error) as e:
            log.debug("serial %s: %r", path, e)
            return False
        deadline = time.time() + PROBE_S
        hello = ""
        try:
            while not hello and time.time() < deadline:
                link.write(b"hello\n")
                await link.drain()
                try:
                    hello = await asyncio.wait_for(self._read_hello(link), min(1.0, deadline - time.time()))
                except asyncio.TimeoutError:
                    pass
        except (OSError, ValueError) as e:
            log.debug("serial %s: %r", path, e)
        if not hello:
            link.close()
            return False
        async with self._lock:  # never cut off a command in flight on the old link
            self._attach(link, link, hello, f"serial {path}", serial=True)
        return True

    @staticmethod
    async def _read_hello(link) -> str:
        while line := await link.readline():
            text = line.decode(errors="replace").strip()
            if text.startswith("hello "):
                return text
        raise ConnectionResetError("serial port closed")

    async def _reply(self) -> str:
        """The board's answer to the command just sent. Over serial other lines can turn up: the ROM boot log, and
        the "hello" the board sends when it starts. A "hello" means it restarted, so gestures must re-home."""
        while True:
            line = await self._reader.readline()
            if not line:
                raise ConnectionResetError("board closed the link")
            text = line.decode(errors="replace").strip()
            if text == "ok" or text.startswith("err"):
                return text
            if text.startswith("hello "):
                self.hello, self.since = text, time.time()
                self.links += 1
                log.info("board restarted: %s", text)

    def _lost(self, reason: str) -> None:
        self.lost_at, self.lost_reason = time.time(), reason
        log.info("board link lost: %s", reason)

    def info(self) -> dict:
        """Link details for the status page; times are seconds ago (None = never)."""
        ago = lambda t: round(time.time() - t, 1) if t else None  # noqa: E731
        return {"connected": self.connected, "links": self.links, "peer": self.peer, "hello": self.hello,
                "link": ("serial" if self.serial else "wifi") if self.connected else None,
                "connected_for": ago(self.since) if self.connected else None,
                "lost_ago": ago(self.lost_at), "lost_reason": self.lost_reason}

    async def wait_connected(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while not self.connected and time.time() < deadline:
            await asyncio.sleep(0.2)
        return self.connected

    def close(self) -> None:
        if self.connected:
            self._writer.close()

    @staticmethod
    def _timeout(command: str) -> float:
        """Seconds the board may take: Wi-Fi replies need up to ~2 s at p99, moves are paced
        (10 units per 8 ms), typing is per character."""
        name, _, arg = command.partition(" ")
        if name == "move":
            return 5 + max(abs(int(n)) for n in arg.split()) / 10 * 0.012
        if name == "type":
            return 5 + 0.1 * len(arg)
        if name == "ping":
            return 8  # an idle check: tolerate a Wi-Fi hiccup rather than drop a healthy board
        return 5

    async def send(self, command: str) -> None:
        if not self.connected:
            raise DeviceError("board not connected")
        timeout = self._timeout(command)
        async with self._lock:
            started = time.time()
            try:
                self._writer.write(command.encode() + b"\n")
                await self._writer.drain()
                reply = await asyncio.wait_for(self._reply(), timeout)
            except (OSError, ValueError, asyncio.TimeoutError) as e:
                self._writer.close()  # the board reconnects on its own within a few seconds
                self._lost(f"'{command[:30]}': {e!r}")
                raise DeviceError(f"board link lost during '{command[:30]}': {e!r}") from e
        self._beat = time.time()  # last reply of any kind
        if command != "ping":
            self._last = self._beat  # last real command: the phone's idle time
            log.debug("%s -> %s (%.0f ms)", command, reply, (time.time() - started) * 1000)
        if reply != "ok":
            raise DeviceError(reply or "board disconnected")

    async def keep_awake(self, idle: float = 25, heartbeat: float = 2, screen=None) -> None:
        """While idle: ping every few seconds, so a board that was unplugged or hung shows as gone within
        seconds (not at the next command), and nudge the pointer now and then so the phone does not auto-lock."""
        while True:
            await asyncio.sleep(heartbeat)
            if not self.connected or time.time() - self._beat < heartbeat:
                continue
            try:
                # nudge only a phone that is awake and mirroring: moving a sleeping phone's pointer can stall the
                # board on its USB link, and there is nothing to keep awake anyway
                if time.time() - self._last > idle and (screen is None or screen.live):
                    await self.send("move 1 0")
                    await self.send("move -1 0")
                else:
                    await self.send("ping")
            except DeviceError as e:
                log.info("board gone: %s", e)
