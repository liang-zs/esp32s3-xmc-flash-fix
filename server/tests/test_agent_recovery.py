"""无变化恢复接入 Agent：RETRY 放行、REPLAN 重规划、ABORT 记为失败。"""

import asyncio

from hidagent.agent import Agent
from hidagent.task_state import TaskState


class Trace:
    def __init__(self):
        self.events = []

    def start(self, task):
        pass

    def write(self, *args, **kwargs):
        self.events.append(kwargs)


class Screen:
    live = True
    frame = b"jpeg"
    aspect = 0.46


class LLM:
    model = "fake"

    async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
        return "1. look\n2. tap"


class PlanLLM:
    model = "fake"

    async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
        return "1. look\n2. tap"


class TapLLM:
    model = "fake"

    async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
        return "THINK: again\nDO: tap 500 500"


def make_agent() -> Agent:
    agent = Agent(phone=None, screen=Screen(), actor=LLM(), planner=LLM(), trace=Trace())
    agent.steps = [{"t": 0.0, "action": "tap", "changed": False}]
    return agent


def test_first_miss_handles_back_to_the_actor():
    agent = make_agent()
    assert asyncio.run(agent._handle_no_change("tap")) is False
    assert agent.steps[-1]["recovery"] == "retry"
    assert agent.task_ctx.state is TaskState.RECOVER


def test_second_miss_replans():
    agent = make_agent()
    asyncio.run(agent._handle_no_change("tap"))
    assert asyncio.run(agent._handle_no_change("tap")) is False
    assert agent.steps[-1]["action"] == "replan (stuck)"
    assert agent.steps[-1]["plan"] == "1. look\n2. tap"


def test_third_miss_fails_the_task():
    agent = make_agent()
    asyncio.run(agent._handle_no_change("tap"))
    asyncio.run(agent._handle_no_change("tap"))
    assert asyncio.run(agent._handle_no_change("tap")) is True
    assert agent.steps[-1]["action"] == "fail"
    assert agent.task_ctx.state is TaskState.FAILED
    assert "屏幕无变化" in agent.steps[-1]["reason"]


def test_state_machine_exists_before_start():


    agent = Agent(phone=None, screen=Screen(), actor=LLM(), planner=LLM(), trace=Trace())
    assert agent.task_ctx.state is TaskState.IDLE
    assert agent.task_ctx.snapshot()["state"] == "idle"


def test_loop_fails_after_three_silent_steps(monkeypatch):
    """整条循环：无变化 → RETRY → 拒绝重复动作 → REPLAN → 三次无变化判失败。"""
    from hidagent import agent as agent_module

    monkeypatch.setattr(agent_module.vision, "changed", lambda *a, **k: False)

    class Phone:
        async def tap(self, x, y, aim=False):
            return None

    class QuietScreen(Screen):
        async def settled(self):
            return None

    agent = Agent(phone=Phone(), screen=QuietScreen(), actor=TapLLM(), planner=PlanLLM(),
                  trace=Trace(), max_steps=10)

    async def run():
        agent.start("打开 Notes")
        await agent._job

    asyncio.run(run())
    assert agent.task_ctx.state is TaskState.FAILED
    assert agent.steps[-1]["action"] == "fail"
    assert "屏幕无变化" in agent.steps[-1]["reason"]
    assert agent.task_ctx.no_progress >= 3


def test_phone_usb_asleep_twice_stops_with_a_hint():
    """板子报 usb asleep（和 iPhone 的 USB 不通）：第二次就停，给出重插提示，不再反复重新规划。"""
    from hidagent.device import DeviceError

    class Device:
        connected = True

        async def wait_connected(self, timeout):
            return True

    class AsleepPhone:
        device = Device()

        async def tap(self, x, y, aim=False):
            raise DeviceError("err phone usb asleep (unlock the phone)")

    class QuietScreen(Screen):
        async def settled(self):
            return None

    agent = Agent(phone=AsleepPhone(), screen=QuietScreen(), actor=TapLLM(), planner=PlanLLM(),
                  trace=Trace(), max_steps=20)

    async def run():
        agent.start("打开哔哩哔哩")
        await agent._job

    asyncio.run(run())
    actions = [s["action"] for s in agent.steps]
    assert actions.count("hiccup") == 2 and "replan (stuck)" not in actions
    assert agent.steps[-1]["action"] == "fail" and "重插" in agent.steps[-1]["reason"]


class PromptPlanLLM:
    """Planner that remembers what it was asked."""
    model = "fake"

    def __init__(self):
        self.prompts = []

    async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
        self.prompts.append(text)
        return "1. look\n2. tap"


def _silent_agent(monkeypatch, planner=None):
    """Every tap changes nothing on screen, so the task heads for 'no change' three times."""
    from hidagent import agent as agent_module

    monkeypatch.setattr(agent_module.vision, "changed", lambda *a, **k: False)

    class Phone:
        async def tap(self, x, y, aim=False):
            return None

    class QuietScreen(Screen):
        async def settled(self):
            return None

    return Agent(phone=Phone(), screen=QuietScreen(), actor=TapLLM(), planner=planner or PlanLLM(),
                 trace=Trace(), max_steps=40)


async def _until_asked(agent):
    for _ in range(200):
        if agent.help:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the agent never asked")


def test_paused_task_stops_when_the_user_says_stop(monkeypatch):
    agent = _silent_agent(monkeypatch)

    async def run():
        agent.start("打开白板画一个笑脸", ask_help=True)
        await _until_asked(agent)
        assert "屏幕无变化" in agent.help["reason"] and agent.running
        agent.decide("stop")
        await agent._job

    asyncio.run(run())
    actions = [s["action"] for s in agent.steps]
    assert actions[-3:] == ["ask", "user", "fail"] and "屏幕无变化" in agent.steps[-1]["reason"]
    assert agent.help is None


def test_replan_with_a_hint_reaches_both_models_then_it_can_ask_again(monkeypatch):
    planner = PromptPlanLLM()
    agent = _silent_agent(monkeypatch, planner)

    async def run():
        agent.start("打开白板画一个笑脸", ask_help=True)
        await _until_asked(agent)
        agent.decide("replan", "笔是左下角圆圈里的图标")
        await asyncio.sleep(0.05)
        await _until_asked(agent)  # still silent: it asks again instead of failing
        agent.decide("stop")
        await agent._job

    asyncio.run(run())
    assert agent.user_hints == ["笔是左下角圆圈里的图标"]
    assert any("笔是左下角圆圈里的图标" in p for p in planner.prompts)
    users = [s for s in agent.steps if s["action"] == "user"]
    assert [u["choice"] for u in users] == ["replan", "stop"] and users[0]["hint"] == "笔是左下角圆圈里的图标"
    assert agent.steps[-1]["action"] == "fail"


def test_nobody_answers_so_it_stops_by_itself(monkeypatch):
    from hidagent import agent as agent_module

    monkeypatch.setattr(agent_module, "HELP_WAIT_S", 0.05)
    agent = _silent_agent(monkeypatch)

    async def run():
        agent.start("打开白板画一个笑脸", ask_help=True)
        await agent._job

    asyncio.run(run())
    assert agent.steps[-1]["action"] == "fail" and "无人回应" in agent.steps[-1]["reason"]


def test_at_most_three_questions_per_task(monkeypatch):
    agent = _silent_agent(monkeypatch)

    async def run():
        agent.start("打开白板画一个笑脸", ask_help=True)
        for _ in range(3):
            await _until_asked(agent)
            agent.decide("continue")
            await asyncio.sleep(0.02)
        await agent._job  # the fourth time it just fails

    asyncio.run(run())
    assert sum(s["action"] == "ask" for s in agent.steps) == 3
    assert agent.steps[-1]["action"] == "fail"


def test_decide_only_while_paused_and_only_known_choices():
    import pytest

    agent = make_agent()
    with pytest.raises(RuntimeError):
        agent.decide("replan")
    with pytest.raises(ValueError):
        agent.decide("maybe")
