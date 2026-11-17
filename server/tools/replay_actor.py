"""Replay recorded actor calls offline, thinking on and off, against the actions of the original run.

    uv run --env-file .env python tools/replay_actor.py ~/.hidagent/traces/<run> [<run> ...]

Same screenshot, same prompt, current ACTOR_PROMPT. "match" means the same action (taps within 4% of
the screen); a mismatch is not always wrong (any point on a row hits it), so read the listed ones.
Use it before switching models, prompts or ACTOR_FAST_BODY; the phone run is still the final check."""
import asyncio, json, math, sys, time
from pathlib import Path
from hidagent.agent import ACTOR_PROMPT, parse_actions
from hidagent.llm import LLM

MODES = {"think": False, "fast": True}

def first(reply):
    try: return parse_actions(reply)[0]
    except Exception as e: return {"action": "INVALID", "err": str(e)[:80]}

def same(a, b):
    if a["action"] != b["action"]: return False
    if a["action"] == "tap": return math.dist((a["x"], a["y"]), (b["x"], b["y"])) <= 40
    if a["action"] == "swipe": return all(abs(a[k]-b[k]) <= 80 for k in ("x1","y1","x2","y2"))
    if a["action"] in ("type", "pinyin"): return a["text"].lower() == b["text"].lower()
    if a["action"] == "key": return a["combo"] == b["combo"]
    return True

async def main(dirs):
    calls = []
    for d in dirs:
        for line in open(Path(d) / "steps.jsonl"):
            r = json.loads(line)
            if r.get("kind") == "llm" and r["role"] == "actor":
                calls.append((d, r["n"], r["prompt"], r["reply"], (Path(d) / f"{r['n']:03d}.jpg").read_bytes()))
    sem = asyncio.Semaphore(4)
    async def one(mode, c):
        async with sem:
            llm = LLM.from_env("ACTOR")
            t = time.time(); reply = await llm.chat(ACTOR_PROMPT, c[2], c[4], fast=MODES[mode])
            return mode, c, reply, time.time() - t, llm.last_usage
    res = await asyncio.gather(*(one(m, c) for m in MODES for c in calls))
    for mode in MODES:
        rows = [r for r in res if r[0] == mode]
        ok = 0
        for _, c, reply, s, u in rows:
            ref, got = first(c[3]), first(reply)
            m = same(ref, got); ok += m
            if not m: print(f"  {mode} {Path(c[0]).name}#{c[1]}: recorded {ref} | got {got}")
        secs = sorted(r[3] for r in rows)
        print(f"{mode}: {ok}/{len(rows)} match, mean {sum(secs)/len(secs):.2f}s, median {secs[len(secs)//2]:.2f}s, "
              f"max {secs[-1]:.2f}s, reasoning {sum(r[4].get('reasoning',0) for r in rows)}")
asyncio.run(main(sys.argv[1:]))
