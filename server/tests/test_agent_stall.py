"""Item 3 gaps (V2.2.1 regression: "一直卡在点击 settings，没有停止"): the screen keeps changing, so the
no-change escalation never fires, but the plan step never finishes. A per-step round budget replans and then
gives up; a whole-task time limit stops a run that is merely slow."""

import asyncio

from hidagent.agent import Agent
from hidagent.task_state import Recovery, TaskContext, TaskState


class Trace:
    def start(self, task):
        pass

    def write(self, *args, **kwargs):
        pass


class Screen:
    live = True
    frame = b"jpeg"
    aspect = 0.46


class LLM:
    model = "fake"

    async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
        return "1. Tap Settings\n2. Tap General"


def test_rounds_without_progress_replan_then_abort():
    ctx = TaskContext("打开 Settings", max_rounds_per_step=3, max_stalls=2)
    assert [ctx.record_round(False, "Tap Settings") for _ in range(3)] == [Recovery.NONE] * 2 + [Recovery.REPLAN]
    assert ctx.state is TaskState.RECOVER and ctx.rounds_on_step == 0
    assert [ctx.record_round(False) for _ in range(3)][-1] is Recovery.ABORT


def test_finishing_a_step_resets_the_budget():
    ctx = TaskContext("打开 Settings", max_rounds_per_step=3)
    ctx.record_round(False)
    ctx.record_round(False)
    assert ctx.record_round(True) is Recovery.NONE
    assert ctx.rounds_on_step == ctx.stalls == 0
    assert ctx.snapshot()["rounds_on_step"] == 0


class WanderLLM:
    """Taps around on screens that do change, but always says it is still on step 1."""
    model = "fake"

    async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
        return "STEP: 1\nTHINK: try\nDO: tap 500 500"


class Phone:
    class device:
        connected = False

    async def tap(self, x, y, aim=False):
        return None


class BusyScreen(Screen):
    async def settled(self):
        return None


def run(agent):
    async def go():
        agent.start("打开 Settings")
        await agent._job
    asyncio.run(go())


def test_changing_screen_without_progress_stops(monkeypatch):
    from hidagent import agent as agent_module
    monkeypatch.setattr(agent_module.vision, "changed", lambda *a, **k: True)  # every tap "does something"
    a = Agent(phone=Phone(), screen=BusyScreen(), actor=WanderLLM(), planner=LLM(), trace=Trace(), max_steps=40)
    run(a)
    actions = [s["action"] for s in a.steps]
    assert "stalled" in actions and actions.count("replan (stuck)") == 1
    assert actions[-1] == "fail" and "卡在步骤" in a.steps[-1]["reason"]
    assert actions.count("tap") < 20  # stopped well before the 40-step limit


def test_time_limit_ends_a_slow_run(monkeypatch):
    from hidagent import agent as agent_module
    monkeypatch.setattr(agent_module.vision, "changed", lambda *a, **k: True)

    class SlowPhone(Phone):
        async def tap(self, x, y, aim=False):
            await asyncio.sleep(1)  # a gesture that hangs

    a = Agent(phone=SlowPhone(), screen=BusyScreen(), actor=WanderLLM(), planner=LLM(), trace=Trace(),
              max_seconds=0.3)
    run(a)
    assert a.steps[-1]["action"] == "fail" and "time limit" in a.steps[-1]["reason"]
    assert a.task_ctx.state is TaskState.FAILED


def test_a_replan_gives_the_new_plan_a_full_budget():
    # iPhone 11, 2026-11-17: a replan after an empty reply was followed at once by a stall replan
    a = Agent(phone=Phone(), screen=BusyScreen(), actor=WanderLLM(), planner=LLM(), trace=Trace())
    a.task_ctx.rounds_on_step = 7
    asyncio.run(a._replan(stuck=True))
    assert a.task_ctx.rounds_on_step == 0


def test_each_run_leaves_a_summary_line(tmp_path, monkeypatch):
    import json

    from hidagent import agent as agent_module
    monkeypatch.setattr(agent_module.vision, "changed", lambda *a, **k: True)

    class DoneLLM:
        model = "fake"
        last_usage = {"prompt": 10, "completion": 5, "reasoning": 2, "cache_hit": 0}

        async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
            return "STEP: 2\nTHINK: ok\nDO: done opened"

    a = Agent(phone=Phone(), screen=BusyScreen(), actor=DoneLLM(), planner=DoneLLM(), trace=Trace(),
              runs_file=tmp_path / "runs.jsonl", price="1,2,3")
    run(a)
    row = json.loads((tmp_path / "runs.jsonl").read_text())
    assert row["result"] == "done" and row["task"] == "打开 Settings" and row["calls"] == 2
    assert row["prompt"] == 20 and row["completion"] == 10 and row["cost"] == round((20 * 2 + 10 * 3) / 1e6, 6)
