#!/usr/bin/env python3
"""Install chat-verified iPhone Agent prompt and precise click improvements.

Target: onehumanbeing/esp32-iphone-agent legacy branch, server/hidagent/agent.py
Run from any directory: python3 /path/to/apply_agent_improvements.py
Creates a timestamped backup. Does not touch ESP32 firmware or secrets.
"""
from __future__ import annotations

import ast
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT = pathlib.Path.home() / "esp32-iphone-agent/server/hidagent/agent.py"
TARGET = pathlib.Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else DEFAULT
ACTOR_TEXT = (HERE / "actor_prompt.txt").read_text(encoding="utf-8").strip()
PLANNER_ADDON = "\n\nIPHONE 18 PRO TASK PLANNING RULES:\n- The device's UI language is English; translate Chinese system app names to English (备忘录=Notes, 设置=Settings) before searching.\n- To navigate away from an open app, use the visible AssistiveTouch floating button (single tap is configured to go Home). The bottom-edge Home swipe and software \"home\" action are unreliable on this device.\n- Prefer Home Screen -> Spotlight -> focus search field -> type English app name -> inspect result -> precise tap -> verify opening.\n- Each stage is a separate visual action and screenshot check; never assume an unverified step worked.\n- If Spotlight is already open, repair/focus its existing search instead of reopening it.\n"

if not TARGET.is_file():
    raise SystemExit(f"Cannot find target: {TARGET}\nPass the path to your local server/hidagent/agent.py.")
original = TARGET.read_text(encoding="utf-8")
tree = ast.parse(original, filename=str(TARGET))
assignments = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ("ACTOR_PROMPT", "PLANNER_PROMPT"):
                assignments[target.id] = node.value
if set(assignments) != {"ACTOR_PROMPT", "PLANNER_PROMPT"}:
    raise SystemExit("Could not locate both prompt assignments. No files changed.")

lines = original.splitlines(keepends=True)
offsets = [0]
for line in lines:
    offsets.append(offsets[-1] + len(line))
def span(node):
    return offsets[node.lineno - 1] + node.col_offset, offsets[node.end_lineno - 1] + node.end_col_offset
# Python AST column offsets are UTF-8 byte offsets. Legacy prompt definition is ASCII
# before the replaced literals; if your local source differs, fail safely.
if any(len(line.encode("utf-8")) != len(line) for line in lines[:min(n.lineno for n in assignments.values()) - 1]):
    raise SystemExit("Non-ASCII lines before prompt assignments: use manual integration instead.")

replacements = []
for key, node in assignments.items():
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise SystemExit(f"{key} is not a literal string; no changes made.")
    if key == "ACTOR_PROMPT":
        revised = ACTOR_TEXT
    else:
        revised = node.value
        if "IPHONE 18 PRO TASK PLANNING RULES:" not in revised:
            revised += PLANNER_ADDON
    replacements.append((*span(node), repr(revised)))

old = 'aim=a.get("precise", False) or self._missed()'
if old in original:
    a = original.index(old)
    replacements.append((a, a + len(old), "aim=True"))
elif "aim=True" in original:
    print("Precise clicking already enabled.")
else:
    raise SystemExit("Unknown tap implementation; no changes made. Check agent.py manually.")

modified = original
for start, end, content in sorted(replacements, reverse=True):
    modified = modified[:start] + content + modified[end:]
compile(modified, str(TARGET), "exec")
if modified == original:
    print("Nothing to change.")
    sys.exit(0)
backup = TARGET.with_name(TARGET.name + ".backup." + time.strftime("%Y%m%d-%H%M%S"))
backup.write_text(original, encoding="utf-8")
TARGET.write_text(modified, encoding="utf-8")
print(f"Patched: {TARGET}\nBackup:  {backup}")
print("Run: python3 -m py_compile " + str(TARGET))
print("Restart: cd ~/esp32-iphone-agent/server && uv run --env-file .env python -m hidagent")
