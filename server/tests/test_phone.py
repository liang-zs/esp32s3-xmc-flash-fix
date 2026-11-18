import asyncio
import json

from hidagent.phone import AIM_ROUNDS, Phone


class Device:
    links = 0

    def __init__(self):
        self.sent = []

    async def send(self, line):
        self.sent.append(line)


def phone(tmp_path, landings):
    (tmp_path / "c.json").write_text(json.dumps({"w": 427, "h": 939}))
    p = Phone(Device(), tmp_path / "c.json", screen=object())
    spots = iter(landings)

    async def spot(nudge, shift, near, wait=0.6):
        return next(spots)

    p._spot = spot
    return p


def corrections(p):
    return [s for s in p.device.sent if s.startswith("move") and s != "move 0 0"][1:]  # after the first approach


def test_aim_corrects_until_on_target(tmp_path):
    # iPhone 11, 2026-11-17: one correction move can itself land short, so check again before clicking
    p = phone(tmp_path, [(0.45, 0.58), (0.49, 0.597), (0.5, 0.6)])
    asyncio.run(p.tap(0.5, 0.6, aim=True))
    assert len(corrections(p)) == 2 and p.device.sent[-1] == "click"
    assert len(p._seen["y"]) == 1  # only the straight-from-the-corner landing is learned


def test_aim_stops_after_a_few_rounds(tmp_path):
    p = phone(tmp_path, [(0.45, 0.58)] * 10)
    asyncio.run(p.tap(0.5, 0.6, aim=True))
    assert len(corrections(p)) == AIM_ROUNDS and p.device.sent[-1] == "click"


def test_precise_tap_starts_from_the_corner(tmp_path):
    p = phone(tmp_path, [(0.2, 0.2), (0.5, 0.6)])
    asyncio.run(p.tap(0.2, 0.2, aim=True))
    asyncio.run(p.tap(0.5, 0.6, aim=True))
    assert p.device.sent.count("home") == 2


def test_pinyin_switches_keyboard_and_back(tmp_path, monkeypatch):
    import hidagent.phone as phone_module
    monkeypatch.setattr(phone_module, "IME_SWITCH", 0)
    p = phone(tmp_path, [])
    asyncio.run(p.pinyin("changanguniang"))
    assert p.device.sent == ["key ctrl+space", "type changanguniang", "key space", "key ctrl+space"]


def test_type_refuses_non_ascii(tmp_path):
    import pytest
    p = phone(tmp_path, [])
    with pytest.raises(RuntimeError, match="ASCII"):
        asyncio.run(p.type("长安姑娘"))
    assert p.device.sent == []


def test_clear_field_selects_deletes_and_deletes_to_line_start(tmp_path):
    p = phone(tmp_path, [])
    asyncio.run(p.clear_field())
    assert p.device.sent == ["key cmd+a", "key backspace", "key cmd+backspace"]
