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
