import asyncio

import pytest

from hidagent.device import Device, DeviceError


def test_timeout_grows_with_move_distance_and_text_length():
    assert Device._timeout("click") == 5
    assert Device._timeout("move 1000 -20") > Device._timeout("move 10 10") >= 5
    assert Device._timeout("type " + "x" * 40) > Device._timeout("type hi")


def test_send_without_board_fails_fast():
    with pytest.raises(DeviceError, match="not connected"):
        asyncio.run(Device().send("ping"))


def test_round_trip_with_a_fake_board():
    async def scenario():
        device = Device()
        server = await device.serve("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)

        async def board():  # answers every command like the firmware does
            while line := await reader.readline():
                writer.write(b"ok\n" if line.strip() != b"bad" else b"err unknown command\n")
                await writer.drain()

        task = asyncio.create_task(board())
        assert await device.wait_connected(2)
        await device.send("ping")
        with pytest.raises(DeviceError, match="unknown command"):
            await device.send("bad")
        assert device.links == 1
        task.cancel()
        writer.close()
        server.close()

    asyncio.run(scenario())


def test_info_reports_the_link_and_why_it_died():
    async def scenario():
        device = Device()
        assert device.info()["lost_ago"] is None and not device.info()["connected"]
        server = await device.serve("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"hello POWERON 1234\n")
        await writer.drain()
        assert await device.wait_connected(2)
        info = device.info()
        assert info["hello"] == "hello POWERON 1234" and info["peer"].startswith("127.0.0.1:")
        writer.close()  # the board goes away; the next command finds out
        with pytest.raises(DeviceError):
            await device.send("ping")
        info = device.info()
        assert not info["connected"] and info["lost_ago"] is not None and "ping" in info["lost_reason"]
        server.close()

    asyncio.run(scenario())


class FakeSerialBoard:
    """The board end of a pty: answers lines like the firmware does over its UART."""

    def __init__(self, junk_first: bytes = b""):
        import os
        self.master, self.slave = os.openpty()
        self.path = os.ttyname(self.slave)
        self.junk_first = junk_first  # the ROM boot log a reset on open would print
        self.restart_on = None        # command that makes the board "restart" before answering
        self._buf = b""
        asyncio.get_running_loop().add_reader(self.master, self._readable)

    def _send(self, data: bytes):
        import os
        os.write(self.master, data)

    def _readable(self):
        import os
        self._buf += os.read(self.master, 1024)
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if self.junk_first:
                self._send(self.junk_first)
                self.junk_first = b""
            if line == self.restart_on:
                self._send(b"ESP-ROM:esp32s3-20210327\r\n\xff\xferst:0x1 (POWERON)\r\nhello power-on 12\r\n")
                self.restart_on = None
            self._send(b"hello power-on 5\n" if line == b"hello" else b"ok\n")

    def unplug(self):
        import os
        asyncio.get_running_loop().remove_reader(self.master)
        os.close(self.master)

    def close(self):
        import os
        os.close(self.slave)


def test_serial_board_is_found_past_its_boot_log_and_answers():
    async def scenario():
        board = FakeSerialBoard(junk_first=b"ESP-ROM:esp32s3\r\nbuild: Mar 27 2021\r\n\x00\xff garbage\r\n")
        device = Device()
        assert await device._probe(board.path)
        info = device.info()
        assert info["link"] == "serial" and info["peer"] == f"serial {board.path}"
        assert info["hello"] == "hello power-on 5" and device.links == 1
        await device.send("move 10 0")
        device.close()
        board.close()

    asyncio.run(scenario())


def test_a_board_restart_mid_command_is_noticed_and_the_reply_still_read():
    async def scenario():
        board = FakeSerialBoard()
        device = Device()
        assert await device._probe(board.path)
        board.restart_on = b"click"
        await device.send("click")  # boot log and "hello" come before the "ok"
        assert device.links == 2 and device.hello == "hello power-on 12"
        device.close()
        board.close()

    asyncio.run(scenario())


def test_wifi_dial_in_is_turned_away_while_the_serial_link_is_live():
    async def scenario():
        board = FakeSerialBoard()
        device = Device()
        assert await device._probe(board.path)
        server = await device.serve("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"hello power-on 1\n")
        try:  # closed on us
            assert await reader.read() == b""
        except ConnectionResetError:
            pass
        assert device.serial and device.links == 1
        await device.send("ping")
        writer.close()
        server.close()
        device.close()
        board.close()

    asyncio.run(scenario())


def test_unplugged_serial_board_shows_as_gone():
    async def scenario():
        board = FakeSerialBoard()
        device = Device()
        assert await device._probe(board.path)
        board.unplug()
        board.close()
        with pytest.raises(DeviceError):
            await device.send("ping")
        assert not device.connected and device.info()["lost_reason"]

    asyncio.run(scenario())


def test_probe_skips_a_port_where_no_board_answers():
    async def scenario():
        import os
        master, slave = os.openpty()  # nothing answers on the other end
        device = Device()
        import hidagent.device as d
        old, d.PROBE_S = d.PROBE_S, 0.5
        try:
            assert not await device._probe(os.ttyname(slave))
        finally:
            d.PROBE_S = old
        assert not device.connected
        os.close(master)
        os.close(slave)

    asyncio.run(scenario())


def test_serial_candidates(monkeypatch):
    from hidagent import device as d
    assert d.serial_candidates("/dev/does-not-exist") == []
    monkeypatch.setattr(d.glob, "glob", lambda g: ["/dev/cu.usbserial-10"] if "usbserial" in g else [])
    assert d.serial_candidates("auto") == ["/dev/cu.usbserial-10"]
