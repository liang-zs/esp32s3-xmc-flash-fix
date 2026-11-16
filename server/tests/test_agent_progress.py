"""Plan progress: once a plan step is done the agent must not go back to it (iPhone 11 run, 2026-11-17:
after Settings → General → About → Home the actor saw the Home Screen and reopened Settings)."""

import asyncio

from hidagent.agent import Agent, parse_actions, plan_steps
from hidagent.debug import Trace

PLAN = "1. Tap Settings.\n2. Tap General.\n3. Tap About.\n4. Return to the Home Screen.\n5. Open Notes."


def agent(plan: str = PLAN) -> Agent:
    a = Agent(phone=None, screen=None, actor=None, planner=None, trace=Trace(None))
    a.plan = plan
    return a


def test_plan_steps():
    assert plan_steps(PLAN)[3] == "Return to the Home Screen."
    assert plan_steps("1) a\n  2. b\nnot a step") == ["a", "b"]


def test_step_line_is_parsed_and_not_part_of_the_action():
    acts = parse_actions("STEP: 4\nTHINK: home\nDO: tap 894 226 precise")
    assert acts[0]["step"] == 4
    assert parse_actions("THINK: x\nDO: wait")[0].get("step") is None


def test_progress_only_moves_forward():
    a = agent()
    a._advance(5)
    assert a.completed == plan_steps(PLAN)[:4] and a.plan_step == 5
    a._advance(1)  # Home Screen looks like the start again: no going back
    assert a.plan_step == 5 and len(a.completed) == 4
    a._advance(None)
    assert a.plan_step == 5


def test_progress_marks_and_completed_list_reach_the_actor():
    a = agent()
    a._advance(3)
    text = a._progress()
    assert "1. [done] Tap Settings." in text and "3. [now] Tap About." in text and "5. [todo] Open Notes." in text
    assert "NEVER redo" in text


def test_completed_steps_survive_a_replan():
    class Screen:
        live, frame, aspect = True, b"jpeg", 0.46

    class Planner:
        model = "fake"
        def __init__(self):
            self.prompts = []
        async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
            self.prompts.append(text)
            return "1. Open Notes."

    planner = Planner()
    a = Agent(phone=None, screen=Screen(), actor=planner, planner=planner, trace=Trace(None))
    a.plan = PLAN
    a._advance(5)
    asyncio.run(a._replan(stuck=True))
    assert "- Tap About." in planner.prompts[0] and "do NOT plan these again" in planner.prompts[0]
    assert a.plan_step == 1 and len(a.completed) == 4
    assert "- Return to the Home Screen." in a._progress()
