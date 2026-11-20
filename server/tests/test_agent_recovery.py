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
