import math

import pytest

from hidagent.agent import circle, parse_actions, parse_command, parse_plan


def test_parse_single_action():
    acts = parse_actions("THINK: The Search pill is visible, so I tap it.\nDO: tap 500 814")
    assert acts == [{"thought": "The Search pill is visible, so I tap it.", "thought_zh": "",
                     "action": "tap", "x": 500, "y": 814}]


def test_parse_several_actions_keeps_thought_on_first():
    acts = parse_actions("THINK: Type and search.\nDO: type Whiteboard\nDO: key enter")
    assert [a["action"] for a in acts] == ["type", "key"]
    assert acts[0]["text"] == "Whiteboard" and acts[1]["combo"] == "enter"
    assert "thought" not in acts[1]


def test_parse_caps_at_eight_actions():  # a keypad expression such as 128 x 37 = is seven taps
    reply = "THINK: draw\n" + "\n".join(f"DO: tap {i} {i}" for i in range(10))
    assert len(parse_actions(reply)) == 8


def test_parse_path_and_circle():
    assert parse_command("path 100 200 300 400 500 600")["points"] == [[100, 200], [300, 400], [500, 600]]
    assert parse_command("circle 500 450 200") == {"action": "circle", "x": 500, "y": 450, "r": 200}


@pytest.mark.parametrize("line", ["fly 1 2", "tap 500", "path 1 2 3"])
def test_parse_rejects_bad_commands(line):
    with pytest.raises(ValueError):
        parse_command(line)


def test_parse_requires_a_do_line():
    with pytest.raises(ValueError):
        parse_actions("THINK: I am not sure what to do.")


def test_parse_plan_with_and_without_languages():
    assert parse_plan("EN:\n1. Tap Search\nZH:\n1. 点搜索") == ("1. Tap Search", "1. 点搜索")
    assert parse_plan("1. Tap Search") == ("1. Tap Search", "1. Tap Search")


def test_circle_is_closed_and_round_on_a_tall_screen():
    aspect = 0.46  # width / height of a portrait iPhone frame
    pts = circle(0.5, 0.5, 0.2, aspect)
    assert pts[0] == pytest.approx(pts[-1])
    for x, y in pts:  # equal radius in pixels: y offsets shrink by the aspect ratio
        assert math.hypot(x - 0.5, (y - 0.5) / aspect) == pytest.approx(0.2)


def test_actor_may_not_repeat_a_move_that_changed_nothing():
    import asyncio

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Screen:
        live, frame, aspect = True, b"jpeg", 0.46

    class Actor:
        model = "fake"
        def __init__(self):
            self.prompts = []
        async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
            self.prompts.append(text)
            return "THINK: again\nDO: swipe 382 253 382 150" if len(self.prompts) == 1 else "THINK: other\nDO: tap 1 2"

    actor = Actor()
    agent = Agent(phone=None, screen=Screen(), actor=actor, planner=actor, trace=Trace(None))
    agent.steps = [{"action": "swipe", "x1": 382, "y1": 253, "x2": 382, "y2": 150, "changed": False}]
    acts = asyncio.run(agent._next_actions())
    assert acts[0]["action"] == "tap"
    assert "changed nothing" in actor.prompts[1]


def test_repeating_a_move_that_keeps_changing_the_screen_is_not_stuck():
    from hidagent.agent import Agent
    from hidagent.debug import Trace

    agent = Agent(phone=None, screen=None, actor=None, planner=None, trace=Trace(None))
    swipe = {"action": "swipe", "x1": 500, "y1": 800, "x2": 500, "y2": 400}
    agent.steps = [{**swipe, "changed": True}, {**swipe, "changed": True}]
    assert not agent._repeating(dict(swipe))
    agent.steps[-1]["changed"] = False  # reached the end of the list
    assert agent._repeating(dict(swipe))


def test_precise_tap_flag():
    assert parse_command("tap 120 640 precise") == {"action": "tap", "x": 120, "y": 640, "precise": True}
    assert "precise" not in parse_command("tap 120 640")


def test_chinese_goes_through_pinyin_not_type():
    # iPhone 11, 2026-11-17: "type 长安姑娘" came out as "9j;5952f53m" in bilibili search
    import pytest

    from hidagent.agent import parse_command
    with pytest.raises(ValueError, match="pinyin"):
        parse_command("type 长安姑娘")
    assert parse_command("pinyin chang'an gu niang") == {"action": "pinyin", "text": "changanguniang"}
    assert parse_command("pinyin nü") == {"action": "pinyin", "text": "nv"}
    with pytest.raises(ValueError, match="letters only"):
        parse_command("pinyin 长安")


def test_actor_answers_fast_on_routine_steps_and_thinks_after_a_miss():
    import asyncio

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Screen:
        live, frame, aspect = True, b"jpeg", 0.46

    class Actor:
        model, fast_extra = "fake", {"thinking": {"type": "disabled"}}
        def __init__(self):
            self.fast = []
        async def chat(self, system, text, image=None, max_tokens=600, temperature=0, fast=False):
            self.fast.append(fast)
            return "THINK: ok\nDO: tap 1 2"

    actor = Actor()
    agent = Agent(phone=None, screen=Screen(), actor=actor, planner=actor, trace=Trace(None))
    agent.steps = [{"action": "tap", "x": 9, "y": 9, "changed": True}]
    asyncio.run(agent._next_actions())
    agent.steps = [{"action": "tap", "x": 9, "y": 9, "changed": False}]  # that tap missed: look harder
    asyncio.run(agent._next_actions())
    agent.steps = [{"action": "tap", "x": 9, "y": 9, "changed": True}]
    agent.task_ctx.rounds_on_step = 3  # the screen changes, but the plan step does not advance
    asyncio.run(agent._next_actions())
    assert actor.fast == [True, False, False] and agent.usage["fast_calls"] == 1
    assert agent.usage["think_missed"] == 1 and agent.usage["think_stuck"] == 1
    assert "think_rejected" not in agent.usage


def test_think_reason_order():
    from hidagent.agent import Agent
    from hidagent.debug import Trace

    agent = Agent(phone=None, screen=None, actor=None, planner=None, trace=Trace(None))
    assert agent._think_reason() == ""
    agent.task_ctx.rounds_on_step = 3
    assert agent._think_reason() == "stuck"
    agent.steps = [{"action": "tap", "x": 1, "y": 1, "changed": False}]
    assert agent._think_reason() == "missed"
    assert agent._think_reason("rejected") == "rejected"


def test_clear_action_parses():
    assert parse_command("clear") == {"action": "clear"}


def test_grid_goes_to_the_actor_only_and_the_trace_keeps_the_clean_frame():
    import asyncio
    import io

    from PIL import Image

    from hidagent.agent import Agent
    from hidagent.debug import Trace
    buf = io.BytesIO()
    Image.new("RGB", (100, 200)).save(buf, "JPEG")
    clean = buf.getvalue()

    class Screen:
        live, frame, aspect = True, clean, 0.46

    class LLM:
        model = "fake"
        def __init__(self):
            self.seen = []
        async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
            self.seen.append((system, image))
            return "DONE: no\nTHINK: ok\nDO: tap 1 2"

    actor, planner = LLM(), LLM()
    agent = Agent(phone=None, screen=Screen(), actor=actor, planner=planner, trace=Trace(None), grid=True)
    _, _, frame = asyncio.run(agent._ask("actor", actor, "SYS", "go"))
    asyncio.run(agent._ask("planner", planner, "SYS", "go"))
    assert frame == clean and actor.seen[0][1] != clean and "grid lines" in actor.seen[0][0]
    assert planner.seen[0] == ("SYS", clean)


def test_press_parses_key_labels():
    assert parse_command("press 1 2 8 × 3 7 =") == {"action": "press", "keys": "1 2 8 × 3 7 ="}
    import pytest
    with pytest.raises(ValueError, match="1-12 key labels"):
        parse_command("press")


def test_press_locates_each_distinct_key_once_and_taps_in_order():
    import asyncio
    import re

    from hidagent.agent import Agent
    from hidagent.debug import Trace
    where = {"1": (140, 756), "2": (380, 756), "8": (380, 537), "×": (854, 537), "=": (854, 866)}

    class Screen:
        live, frame, aspect = True, None, 0.46  # no frame: vision.changed counts every tap as a change
        async def settled(self, **kw):
            pass

    class Actor:
        model, fast_extra, last_usage = "fake", {"thinking": {"type": "disabled"}}, {}
        def __init__(self):
            self.asked = []
        async def chat(self, system, text, image=None, max_tokens=600, temperature=0, fast=False):
            label = re.search(r"labeled '(.+?)'", text).group(1)
            self.asked.append(label)
            return "{} {}".format(*where[label])

    class Phone:
        def __init__(self):
            self.taps = []
        async def tap(self, x, y, aim=False):
            self.taps.append((round(x * 1000), round(y * 1000)))

    actor, phone = Actor(), Phone()
    agent = Agent(phone=phone, screen=Screen(), actor=actor, planner=actor, trace=Trace(None))
    act = {"action": "press", "keys": "1 2 8 × 2 ="}
    asyncio.run(agent._do(act))
    assert sorted(actor.asked) == sorted(set("1 2 8 × =".split()))  # "2" asked once
    assert phone.taps == [where[k] for k in "1 2 8 × 2 =".split()] == act["taps"]
    assert act["registered"] == [True] * 6


def test_press_taps_a_digit_again_when_the_screen_did_not_change_but_never_an_operator():
    import asyncio

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Screen:
        live, frame, aspect = True, b"jpeg", 0.46
        async def settled(self, **kw):
            pass

    class Phone:
        def __init__(self):
            self.taps = []
        async def tap(self, x, y, aim=False):
            self.taps.append(round(x * 1000))

    lost = {"first 1", "first x"}  # the first tap on 1 and on x change nothing
    seen = []

    async def press_key(label, x, y):
        await agent.phone.tap(x / 1000, y / 1000, aim=True)
        key = ("first " if label not in seen else "again ") + label
        seen.append(label)
        return key not in lost

    agent = Agent(phone=Phone(), screen=Screen(), actor=None, planner=None, trace=Trace(None))
    agent._press_key = press_key

    async def locate(labels):
        return [({"1": 100, "2": 200, "x": 300}[k], 500) for k in labels]
    agent._locate_keys = locate
    act = {"action": "press", "keys": "1 2 x"}
    asyncio.run(agent._do(act))
    assert agent.phone.taps == [100, 100, 200, 300]  # 1 tapped twice, x once
    assert act["retapped"] == ["1"] and act["registered"] == [True, True, False]


def _home_agent(answers):
    """An agent whose locate questions are answered from `answers` (button, menu Home) and whose phone logs."""
    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Screen:
        live, frame, aspect = True, None, 0.46  # no frame: every tap counts as a change
        async def settled(self, **kw):
            pass

    class Actor:
        model, fast_extra, last_usage = "fake", {"thinking": {"type": "disabled"}}, {}
        async def chat(self, system, text, image=None, max_tokens=600, temperature=0, fast=False):
            return answers["home" if "Home item" in text else "button"]

    class Phone:
        def __init__(self):
            self.did, self.moves = [], []
        async def tap(self, x, y, aim=False):
            self.did.append(("tap", round(x * 1000), round(y * 1000)))
        async def go_home(self):
            self.did.append(("swipe home",))
        async def point(self, x, y, aim=False):  # moves without a click, kept apart from taps
            self.moves.append((round(x * 1000), round(y * 1000)))
        async def double_tap(self, x, y, aim=False):
            self.did.append(("double", round(x * 1000), round(y * 1000)))

    return Agent(phone=Phone(), screen=Screen(), actor=Actor(), planner=None, trace=Trace(None))


def test_home_taps_assistivetouch_then_its_home_item():
    import asyncio
    agent = _home_agent({"button": "880 770", "home": "570 855"})
    act = {"action": "home"}
    asyncio.run(agent._do(act))
    assert agent.phone.did == [("tap", 880, 770), ("tap", 570, 855)] and act["taps"] == [(880, 770), (570, 855)]


def test_home_without_assistivetouch_falls_back_to_the_swipe():
    import asyncio
    agent = _home_agent({"button": "none", "home": "none"})
    asyncio.run(agent._do({"action": "home"}))
    assert agent.phone.did == [("swipe home",)]


def test_home_fails_when_the_menu_does_not_open():
    import asyncio

    import pytest
    agent = _home_agent({"button": "880 770", "home": "none"})
    with pytest.raises(ValueError, match="menu did not open"):
        asyncio.run(agent._do({"action": "home"}))
    assert agent.phone.did == [("tap", 880, 770), ("tap", 880, 770)]  # tried the button twice


def test_home_with_the_menu_already_open_taps_its_home_item():
    import asyncio
    agent = _home_agent({"button": "none", "home": "570 855"})
    asyncio.run(agent._do({"action": "home"}))
    assert agent.phone.did == [("tap", 570, 855)]


def test_home_taps_the_menu_home_again_when_the_first_tap_changed_nothing(monkeypatch):
    import asyncio

    import hidagent.agent as agent_module
    agent = _home_agent({"button": "880 770", "home": "570 855"})
    results = iter([False, True])  # the first Home tap does nothing, the second goes home
    monkeypatch.setattr(agent_module.vision, "changed", lambda *a, **k: next(results))
    asyncio.run(agent._do({"action": "home"}))
    assert agent.phone.did == [("tap", 880, 770), ("tap", 570, 855), ("tap", 570, 855)]


def test_tapon_locates_the_described_thing_and_taps_it():
    import asyncio

    import pytest
    pill = "Search pill above the Dock"
    assert parse_command(f"tapon {pill}") == {"action": "tapon", "what": pill}
    agent = _home_agent({"button": "500 835", "home": "none"})  # every non-menu question gets "500 835"
    act = {"action": "tapon", "what": "Search pill above the Dock"}
    asyncio.run(agent._do(act))
    assert agent.phone.did == [("tap", 500, 835)] and act["taps"] == [(500, 835)]
    agent = _home_agent({"button": "none", "home": "none"})
    with pytest.raises(ValueError, match="could not find"):
        asyncio.run(agent._do({"action": "tapon", "what": "Purple unicorn button"}))


def test_home_taps_the_button_again_when_the_menu_does_not_show_up():
    import asyncio
    agent = _home_agent({"button": "880 770", "home": "none"})
    answers = iter(["880 770", "880 770", "none", "none", "570 855"])  # button (two wordings); menu missing twice

    async def chat(system, text, image=None, max_tokens=600, temperature=0, fast=False):
        return next(answers)
    agent.actor.chat = chat
    asyncio.run(agent._do({"action": "home"}))
    assert agent.phone.did == [("tap", 880, 770), ("tap", 880, 770), ("tap", 570, 855)]


def test_home_by_double_tap_needs_one_located_spot(monkeypatch):
    import asyncio

    import hidagent.agent as agent_module
    agent = _home_agent({"button": "750 945", "home": "none"})
    agent.at_double_tap_home = True
    monkeypatch.setattr(agent_module.vision, "changed", lambda *a, **k: True)
    act = {"action": "home"}
    asyncio.run(agent._do(act))
    assert agent.phone.did == [("double", 750, 945)] and act["taps"] == [(750, 945)]


def test_home_by_double_tap_taken_as_single_tap_uses_the_menu(monkeypatch):
    import asyncio

    import hidagent.agent as agent_module
    agent = _home_agent({"button": "750 945", "home": "570 855"})  # the menu is there after the double tap
    agent.at_double_tap_home = True
    monkeypatch.setattr(agent_module.vision, "changed", lambda *a, **k: True)
    asyncio.run(agent._do({"action": "home"}))
    assert agent.phone.did == [("double", 750, 945), ("tap", 570, 855)]


def test_home_by_double_tap_tries_twice_then_fails(monkeypatch):
    import asyncio

    import pytest

    import hidagent.agent as agent_module
    agent = _home_agent({"button": "750 945", "home": "none"})
    agent.at_double_tap_home = True
    monkeypatch.setattr(agent_module.vision, "changed", lambda *a, **k: False)
    with pytest.raises(ValueError, match="changed nothing"):
        asyncio.run(agent._do({"action": "home"}))
    assert agent.phone.did == [("double", 750, 945), ("double", 750, 945)]


def test_the_pointer_is_parked_in_the_middle_and_an_answer_there_is_the_pointer():
    import asyncio

    from hidagent.agent import AT_BUTTON_Q2
    agent = _home_agent({"button": "none", "home": "none"})

    async def chat(system, text, image=None, max_tokens=600, temperature=0, fast=False):
        if "Home item" in text:
            return "none"
        return "750 950" if AT_BUTTON_Q2[:60] in text else "505 520"  # the old wording answered the pointer
    agent.actor.chat = chat
    assert asyncio.run(agent._locate_at_button()) == (750, 950)
    assert agent.phone.moves == [(500, 500)] and agent.phone.did == []

    async def pointer_only(system, text, image=None, max_tokens=600, temperature=0, fast=False):
        return "none" if "Home item" in text else "505 520"
    agent.actor.chat = pointer_only
    assert asyncio.run(agent._locate_at_button()) is None
