"""Link to the HID bridge board (see firmware/hid_bridge): the board dials in over Wi-Fi (TCP), or sits on a
USB serial port. Same line protocol either way."""

import asyncio
import fcntl
import glob
import hashlib
import hmac
import logging
import os
import re
import secrets
import socket
import subprocess
import termios
import time
import tty

log = logging.getLogger(__name__)

BAUD = 115200
# USB-UART chips on ESP32-S3 boards: CP210x, CH34x (Apple's driver or WCH's), FTDI
SERIAL_GLOBS = ["/dev/cu.usbserial*", "/dev/cu.wchusbserial*", "/dev/cu.SLAB_USBtoUART*", "/dev/cu.usbmodem*"]
PROBE_S = 3.0  # opening the port may reset the board: boot log, then "hello", well within this
QUIET_PORT_S = 300.0  # a port where no board answered is left alone this long (it may be esptool's, mid-flash)
DISCOVERY_PORT = 7301  # the board listens here for "hidagent <tcp port>" beacons


def broadcast_addrs(ifconfig: str) -> list[str]:
    """Directed broadcast address of every IPv4 interface (`ifconfig` output), plus the limited broadcast.
    macOS sends 255.255.255.255 out of one interface only, so each network gets its own as well."""
    found = re.findall(r"inet (?!127\.)\S+ netmask \S+ broadcast (\S+)", ifconfig)
    return list(dict.fromkeys(found + ["255.255.255.255"]))


def link_auth(key: str, message: str) -> str:
    """Hex HMAC-SHA256(key, message): how each end proves it knows LINK_KEY."""
    return hmac.new(key.encode(), message.encode(), hashlib.sha256).hexdigest()


def _ip(addr) -> str:
    host = addr[0] if addr else ""
    return host.removeprefix("::ffff:")


class DeviceError(RuntimeError):
    pass


class SerialLink:
    """A tty as an asyncio line stream: readline() like a StreamReader, write/drain/close like a StreamWriter."""

    def __init__(self, path: str, baud: int = BAUD):
        self.path = path
        fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # in use (esptool, a serial monitor): hands off
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

    def __init__(self, key: str = ""):
        self.key = key  # LINK_KEY: shared with the firmware, answers the board's hello nonce
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
        if not await self._auth(reader, writer, hello):
            writer.close()
            return
        peer = writer.get_extra_info("peername")
        self._attach(reader, writer, hello, f"{peer[0]}:{peer[1]}" if peer else "", serial=False)

    async def _auth(self, reader, writer, hello: str) -> bool:
        """With LINK_KEY both ends prove they know it, each over the other's nonce and the address it sees the
        other at (so a relay in the middle fails):
            board  -> hello <reason> <uptime> <nonce_b>
            server -> auth <HMAC(key, "server|" + nonce_b + "|" + board IP)> <nonce_s>
            board  -> ok <HMAC(key, "board|" + nonce_s + "|" + server IP)>
        Without the board's proof anyone on the network could pose as the board and take its link over."""
        words = hello.split()
        offered = len(words) >= 4 and words[0] == "hello"
        if not self.key:
            if offered:
                log.warning("board asks for link auth but LINK_KEY is not set; turning it away")
            return not offered
        if not offered:
            log.warning("a client on the board port did not offer link auth (old firmware, or not the board); "
                        "turning it away: %s", hello[:60])
            return False
        board_ip, server_ip = _ip(writer.get_extra_info("peername")), _ip(writer.get_extra_info("sockname"))
        mine = secrets.token_hex(8)
        try:
            writer.write(f"auth {link_auth(self.key, f'server|{words[3]}|{board_ip}')} {mine}\n".encode())
            await writer.drain()
            reply = (await asyncio.wait_for(reader.readline(), 3)).decode(errors="replace").strip()
        except (OSError, asyncio.TimeoutError) as e:
            reply = repr(e)
        status, _, proof = reply.partition(" ")
        expected = link_auth(self.key, f"board|{mine}|{server_ip}")
        if status != "ok" or not hmac.compare_digest(proof.encode(), expected.encode()):
            log.warning("link auth failed from %s (%s): LINK_KEY differs, or this is not the board",
                        board_ip, reply[:40])
            return False
        return True

    async def beacon(self, tcp_port: int, every: float = 1.0, port: int = DISCOVERY_PORT, addrs=None) -> None:
        """While no board is linked, tell the network where to dial: "hidagent <tcp port>" to every broadcast
        address. The board then needs no IP written in, so it works on any network it can join."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)
        message, targets, refreshed = f"hidagent {tcp_port}".encode(), [], 0.0
        try:
            while True:
                if not self.connected:
                    if time.time() - refreshed > 10:  # networks come and go (Wi-Fi switch, hotspot)
                        targets = addrs() if addrs else await asyncio.to_thread(_local_broadcasts)
                        refreshed = time.time()
                    for addr in targets:
                        try:
                            sock.sendto(message, (addr, port))
                        except OSError:
                            pass  # that network just went away
                await asyncio.sleep(every)
        finally:
            sock.close()

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
        """Keep a board on a USB serial port linked: find it, and find it again after a replug. A port where no
        board answered is not probed again for a while: probing writes to it, which breaks a flash in progress."""
        quiet: dict[str, float] = {}
        while True:
            if not (self.connected and self.serial):
                now = time.time()
                for path in serial_candidates(spec):
                    if quiet.get(path, 0) > now:
                        continue
                    if await self._probe(path):
                        quiet.pop(path, None)
                        break
                    quiet[path] = now + QUIET_PORT_S
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


def _local_broadcasts() -> list[str]:
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    return broadcast_addrs(out)
