"""Run the task set in tools/tasks.json on the phone through a running server and report per task.

    uv run python tools/regress.py                      # every task once
    uv run python tools/regress.py -n 3 calculator ...  # chosen tasks, three times each

Each run starts from the Home Screen (the agent's AssistiveTouch home). A task passes when the agent finishes
with `done` and, when the task lists `expect`, its answer contains one of those strings; with `min_actions`, a
pass in fewer actions is SUSPECT (e.g. an answer left on screen by the previous run). A task with `help_hint`
runs the way the web console does (it pauses and asks before giving up); this script answers "replan" with that
hint, so the ask-for-help path is covered too, and the row counts the helps. Results are appended to
~/.hidagent/regress.jsonl with the git commit, so versions can be compared later.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
OUT = Path.home() / ".hidagent" / "regress.jsonl"


def status(c: httpx.Client) -> dict:
    return c.get("/api/status").json()


def judge(task: dict, result: str, answer: str, actions: int) -> str:
    """PASS, FAIL or SUSPECT (done and right, but in fewer actions than the task needs: e.g. Calculator still
    showing the answer from the previous run, read off without calculating)."""
    if result != "done" or (task.get("expect") and not any(e in answer for e in task["expect"])):
        return "FAIL"
    return "SUSPECT" if actions < task.get("min_actions", 0) else "PASS"


def go_home(c: httpx.Client) -> None:
    """Start from the Home Screen: the agent's AssistiveTouch home, else the bottom-edge swipe."""
    if c.post("/api/agent/home", timeout=60).status_code != 200:
        c.post("/api/home")
    time.sleep(2)


class Landscape(RuntimeError):
    pass


def run_agent(c: httpx.Client, text: str, wait: float = 3, help_hint: str = "") -> tuple[dict, dict]:
    """Start the agent on `text` from the Home Screen and wait; the run summary and the final status.
    With `help_hint` the task may pause and ask; each time it does, answer replan with that hint."""
    c.post("/api/agent/clear")
    if status(c)["aspect"] > 1:  # e.g. the previous task left a full-screen video: everything assumes portrait
        raise Landscape("the phone screen is landscape (a full-screen video?): turn it back to portrait")
    go_home(c)
    r = c.post("/api/agent", json={"text": text, "ask_help": bool(help_hint)})
    r.raise_for_status()
    time.sleep(wait)
    while (s := status(c))["agent"]["running"]:
        if help_hint and s["agent"].get("help"):
            print(f"  asked for help ({s['agent']['help'].get('reason', '')[:80]}): replan with the hint", flush=True)
            c.post("/api/agent/help", json={"choice": "replan", "hint": help_hint})
        time.sleep(wait)
    return c.get("/api/runs", params={"last": 1}).json()["runs"][-1], s


def run_one(c: httpx.Client, task: dict, wait: float = 3) -> dict:
    run, s = run_agent(c, task["task"], wait, task.get("help_hint", ""))
    last = s["agent"]["steps"][-1] if s["agent"]["steps"] else {}
    answer = str(last.get("answer", ""))
    verdict = judge(task, run["result"], answer, run["actions"])
    row = {"id": task["id"], "ok": verdict == "PASS", "verdict": verdict, "result": run["result"],
           "answer": answer[:200],
           "reason": run.get("reason", ""), "seconds": run["seconds"], "actions": run["actions"],
           "calls": run.get("calls"), "fast_calls": run.get("fast_calls"), "llm_s": run.get("llm_s"),
           "cost": run.get("cost"), "helps": run.get("helps", 0),
           "think": {k: v for k, v in run.items() if k.startswith("think_")}}
    if task.get("teardown"):  # undo what the task left on the phone (a test note, a playing video); not scored
        try:
            done, _ = run_agent(c, task["teardown"], wait)
            row["teardown"] = done["result"] + (f": {done.get('reason', '')}" if done.get("reason") else "")
        except Landscape as e:
            row["teardown"] = f"skipped: {e}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="task ids (default: all)")
    ap.add_argument("-n", type=int, default=1, help="runs per task")
    ap.add_argument("--url", default="http://localhost:8890")
    args = ap.parse_args()
    tasks = json.loads((HERE / "tasks.json").read_text())
    if args.ids:
        tasks = [t for t in tasks if t["id"] in args.ids]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with httpx.Client(base_url=args.url, timeout=30) as c:
        s = status(c)
        if not (s["board"]["connected"] and s["screen"]):
            raise SystemExit(f"phone not ready: board {s['board']['connected']}, screen {s['screen']}")
        rows = []
        stop = False
        for task in tasks:
            for _ in range(args.n):
                try:
                    result = run_one(c, task)
                except Landscape as e:
                    print(f"STOPPED before {task['id']}: {e}")
                    stop = True
                    break
                row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit, **result}
                rows.append(row)
                OUT.parent.mkdir(exist_ok=True)
                with OUT.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{row['verdict']:7s} {row['id']:16s} {row['seconds']:6.1f}s "
                      f"{row['actions']:3d} acts  {row['result']:7s} {row['answer'] or row['reason']}"[:200]
                      + (f"  [helps: {row['helps']}]" if row.get("helps") else "")
                      + (f"  [teardown: {row['teardown']}]" if "teardown" in row else ""), flush=True)
            if stop:
                break
    ok = [r for r in rows if r["ok"]]
    if not rows:
        return
    print(f"\n{len(ok)}/{len(rows)} passed, {commit}; average {sum(r['seconds'] for r in rows) / len(rows):.0f} s, "
          f"cost {sum(r['cost'] or 0 for r in rows):.3f}")


if __name__ == "__main__":
    main()
