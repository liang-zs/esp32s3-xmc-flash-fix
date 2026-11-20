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


IFCONFIG = """lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
	inet 127.0.0.1 netmask 0xff000000
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	inet 192.168.0.4 netmask 0xffffff00 broadcast 192.168.0.255
en7: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	inet 172.20.10.2 netmask 0xfffffff0 broadcast 172.20.10.15
"""


def test_broadcast_addresses_cover_every_network():
    from hidagent.device import broadcast_addrs
    assert broadcast_addrs(IFCONFIG) == ["192.168.0.255", "172.20.10.15", "255.255.255.255"]
    assert broadcast_addrs("") == ["255.255.255.255"]


async def _dial(port: int, hello: bytes):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(hello)
    await writer.drain()
    return reader, writer


def test_board_with_a_nonce_is_linked_once_the_server_proves_the_key():
    from hidagent.device import link_auth

    async def scenario():
        device = Device(key="k3y")
        server = await device.serve("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await _dial(port, b"hello power-on 99 abc123\n")
        line = (await asyncio.wait_for(reader.readline(), 2)).decode().strip()
        assert line == "auth " + link_auth("k3y", "abc123")
        writer.write(b"ok\n")
        await writer.drain()
        assert await device.wait_connected(2)
        assert device.hello == "hello power-on 99 abc123"
        writer.close()
        server.close()

    asyncio.run(scenario())


def test_board_that_refuses_our_auth_is_not_linked():
    async def scenario():
        device = Device(key="wrong")
        server = await device.serve("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await _dial(port, b"hello power-on 99 abc123\n")
        await asyncio.wait_for(reader.readline(), 2)  # our auth line
        writer.write(b"err auth\n")
        await writer.drain()
        assert not await device.wait_connected(0.5)
        writer.close()
        server.close()

    asyncio.run(scenario())


def test_board_asking_for_auth_without_a_key_here_is_turned_away():
    async def scenario():
        device = Device()
        server = await device.serve("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await _dial(port, b"hello power-on 99 abc123\n")
        try:
            assert await asyncio.wait_for(reader.read(), 2) == b""  # hung up without an auth line
        except ConnectionResetError:
            pass
        assert not device.connected
        writer.close()
        server.close()

    asyncio.run(scenario())


def test_beacon_announces_the_tcp_port_until_a_board_links():
    async def scenario():
        got = asyncio.Queue()

        class Listener(asyncio.DatagramProtocol):
            def datagram_received(self, data, addr):
                got.put_nowait(data)

        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(Listener, local_addr=("127.0.0.1", 0))
        port = transport.get_extra_info("sockname")[1]
        device = Device(key="k")
        task = asyncio.create_task(device.beacon(7300, every=0.05, port=port, addrs=lambda: ["127.0.0.1"]))
        assert await asyncio.wait_for(got.get(), 2) == b"hidagent 7300"
        device._writer = type("W", (), {"is_closing": lambda self: False})()  # a board is linked now
        await asyncio.sleep(0.15)
        while not got.empty():
            got.get_nowait()
        await asyncio.sleep(0.2)
        assert got.empty()  # quiet while linked
        task.cancel()
        transport.close()

    asyncio.run(scenario())


def test_serial_loop_leaves_a_silent_port_alone(monkeypatch):
    from hidagent import device as d

    async def scenario():
        device = Device()
        probed = []

        async def probe(path):
            probed.append(path)
            return False

        monkeypatch.setattr(device, "_probe", probe)
        monkeypatch.setattr(d, "serial_candidates", lambda spec: ["/dev/cu.usbmodem1"])
        task = asyncio.create_task(device.serial_loop("auto", every=0.01))
        await asyncio.sleep(0.1)
        task.cancel()
        assert probed == ["/dev/cu.usbmodem1"]  # once, then quiet for QUIET_PORT_S

    asyncio.run(scenario())


def test_a_port_held_by_another_program_is_not_opened():
    import fcntl
    import os

    from hidagent.device import SerialLink

    async def scenario():
        master, slave = os.openpty()
        path = os.ttyname(slave)
        holder = os.open(path, os.O_RDWR | os.O_NOCTTY)
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)  # like esptool's exclusive open
        with pytest.raises(OSError):
            SerialLink(path)
        os.close(holder)
        link = SerialLink(path)  # free again
        link.close()
        os.close(master)
        os.close(slave)

    asyncio.run(scenario())
