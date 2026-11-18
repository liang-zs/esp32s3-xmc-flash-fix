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


def go_to(a: Agent, step: int) -> None:
    for _ in range(step - 1):  # the plan moves one step per round
        a._advance(None, done=True)


def test_progress_only_moves_forward():
    a = agent()
    go_to(a, 5)
    assert a.completed == plan_steps(PLAN)[:4] and a.plan_step == 5
    a._advance(1)  # Home Screen looks like the start again: no going back
    assert a.plan_step == 5 and len(a.completed) == 4
    a._advance(None)
    assert a.plan_step == 5


def test_progress_marks_and_completed_list_reach_the_actor():
    a = agent()
    go_to(a, 3)
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
    go_to(a, 5)
    asyncio.run(a._replan(stuck=True))
    assert "- Tap About." in planner.prompts[0] and "do NOT plan these again" in planner.prompts[0]
    assert a.plan_step == 1 and len(a.completed) == 4
    assert "- Return to the Home Screen." in a._progress()


def test_done_line_is_parsed():
    assert parse_actions("DONE: yes\nTHINK: Settings is open\nDO: tap 1 2")[0]["done"] is True
    assert parse_actions("DONE: No\nTHINK: x\nDO: wait")[0]["done"] is False
    assert "done" not in parse_actions("THINK: x\nDO: wait")[0]


def test_done_moves_one_step_and_no_keeps_it():
    a = agent()
    a._advance(None, done=False)
    assert a.plan_step == 1 and a.completed == []
    a._advance(None, done=True)
    assert a.plan_step == 2 and a.completed == ["Tap Settings."]
    a._advance(4, done=False)  # the yes/no wins over a step number in the same reply
    assert a.plan_step == 2


def test_a_claimed_jump_moves_only_one_step():
    # iPhone 11, 2026-11-17: the actor claimed step 4 while still on step 1 and skipped two steps
    a = agent()
    a._advance(4)
    assert a.plan_step == 2 and a.completed == ["Tap Settings."]


def test_last_step_done_marks_everything_done():
    a = agent()
    go_to(a, 5)
    a._advance(None, done=True)
    a._advance(None, done=True)  # nothing past the end
    assert a.plan_step == 6 and len(a.completed) == 5
    assert "[now]" not in a._progress()


def test_done_through_moves_to_the_step_after_the_last_finished_one():
    # iPhone 11, 2026-11-18: Settings reopened straight on About; one step per round sent the agent Home and
    # back into Settings. "DONE: yes through 3" now covers steps 1-3 at once.
    acts = parse_actions("DONE: yes through 3\nTHINK: Settings opened on About\nDO: home")
    assert acts[0]["done"] is True and acts[0]["done_through"] == 3
    a = agent()
    a._advance(None, done=True, through=3)
    assert a.plan_step == 4 and a.completed == plan_steps(PLAN)[:3]
    a._advance(None, done=True, through=2)  # an older step number: just one more step
    assert a.plan_step == 5
    a._advance(None, done=True, through=99)  # past the end: everything done, no crash
    assert a.plan_step == 6 and len(a.completed) == 5
    assert "done_through" not in parse_actions("DONE: no through 3\nTHINK: x\nDO: wait")[0]
