import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("regress", Path(__file__).parents[1] / "tools" / "regress.py")
regress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regress)


def test_judge_flags_a_too_quick_pass_as_suspect():
    calc = {"id": "calculator", "expect": ["4736", "4,736"], "min_actions": 5}
    assert regress.judge(calc, "done", "128 × 37 = 4,736", 7) == "PASS"
    # iPhone 11, 2026-11-18: Calculator still showed 4,736 and the agent read it off in 4 actions, 16 s
    assert regress.judge(calc, "done", "128 × 37 = 4736", 4) == "SUSPECT"
    assert regress.judge(calc, "done", "4,700", 9) == "FAIL"
    assert regress.judge(calc, "fail", "", 40) == "FAIL"
    assert regress.judge({"id": "x"}, "done", "", 1) == "PASS"


def test_a_task_with_teardown_runs_it_after_the_task_and_does_not_score_it(monkeypatch):
    import httpx
    posted, runs = [], [{"result": "done", "actions": 6, "seconds": 40},
                        {"result": "done", "actions": 4, "seconds": 20}]

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST":
            posted.append((path, request.content.decode()))
            return httpx.Response(200, json={"ok": True})
        if path == "/api/status":
            return httpx.Response(200, json={"aspect": 0.46, "agent": {"running": False,
                                                                       "steps": [{"answer": "已输入"}]}})
        if path == "/api/runs":
            return httpx.Response(200, json={"runs": [runs.pop(0)]})
        return httpx.Response(404)

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle)) as c:
        monkeypatch.setattr(regress.time, "sleep", lambda s: None)
        row = regress.run_one(c, {"id": "notes-chinese", "task": "写笔记", "teardown": "删掉那条笔记"}, wait=0)
    started = [body for path, body in posted if path == "/api/agent"]
    assert len(started) == 2 and "写笔记" in started[0] and "删掉那条笔记" in started[1]
    assert row["verdict"] == "PASS" and row["actions"] == 6 and row["teardown"] == "done"


def test_a_task_with_a_help_hint_asks_and_gets_replan_with_that_hint(monkeypatch):
    import httpx
    posted, polls = [], iter([False, True, True, False])  # the aspect check, paused, running, finished
    helps = iter([{"reason": "3 errors in a row", "since": 0}, None])

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST":
            posted.append((path, json.loads(request.content) if request.content else None))
            return httpx.Response(200, json={"ok": True})
        if path == "/api/status":
            running = next(polls, False)
            return httpx.Response(200, json={"aspect": 0.46, "agent": {
                "running": running, "help": next(helps, None) if running else None, "steps": [{"answer": "画好了"}]}})
        if path == "/api/runs":
            return httpx.Response(200, json={"runs": [{"result": "done", "actions": 9, "seconds": 80, "helps": 1}]})
        return httpx.Response(404)

    task = {"id": "freeform-smiley", "task": "画一个笑脸", "min_actions": 6, "help_hint": "画笔在左下角"}
    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle)) as c:
        monkeypatch.setattr(regress.time, "sleep", lambda s: None)
        row = regress.run_one(c, task, wait=0)
    started = [body for path, body in posted if path == "/api/agent"]
    assert started == [{"text": "画一个笑脸", "ask_help": True}]
    answers = [body for path, body in posted if path == "/api/agent/help"]
    assert answers == [{"choice": "replan", "hint": "画笔在左下角"}]
    assert row["verdict"] == "PASS" and row["helps"] == 1


def test_tasks_without_a_hint_never_ask():
    tasks = json.loads((Path(__file__).parents[1] / "tools" / "tasks.json").read_text())
    assert [t["id"] for t in tasks if t.get("help_hint")] == ["freeform-smiley"]
    assert tasks[-1]["id"] == "bilibili-search"  # it can end in a landscape video: keep it last
