"""The actor's reply format: DO lines -> action dicts, plan text -> steps, and the geometry of an action."""

import math
import re
import unicodedata

META = ("t", "thought", "thought_zh", "think_s", "act_s", "shot", "_shot", "changed",
        "recovery", "step", "done", "done_through")  # not part of the action

# action name -> argument names; numeric arguments are 0-1000 screen coordinates
ACTIONS = {"tap": ("x", "y"), "swipe": ("x1", "y1", "x2", "y2"), "circle": ("x", "y", "r"), "path": ("points",),
           "type": ("text",), "pinyin": ("text",), "key": ("combo",), "done": ("answer",), "fail": ("reason",),
           "home": (), "back": (), "wait": (), "replan": (), "clear": (), "press": ("keys",), "tapon": ("what",)}
TEXT_ARGS = ("text", "combo", "answer", "reason", "keys", "what")


def parse_command(line: str) -> dict:
    """One DO line, e.g. "tap 152 362" -> {"action": "tap", "x": 152, "y": 362}."""
    name, _, rest = line.strip().partition(" ")
    name = name.lower()
    if name not in ACTIONS:
        raise ValueError(f"unknown action: {line[:100]}")
    keys = ACTIONS[name]
    if keys and keys[0] in TEXT_ARGS:
        text = rest.strip().strip('"')
        if name == "type":  # full-width ！？，Ａ (a Chinese keyboard's habit) -> their ASCII forms; CJK stays as is
            text = "".join(unicodedata.normalize("NFKC", c) if 0xFF01 <= ord(c) <= 0xFF5E or c == "\u3000" else c
                           for c in text)
        if name == "type" and not text.isascii():  # the board types ASCII keys only: Chinese came out as garbage
            raise ValueError(f"type sends ASCII only, it cannot enter {text[:20]!r}. For Chinese use the pinyin "
                             "action with the pinyin letters, e.g. 长安姑娘 -> DO: pinyin changanguniang")
        if name == "press":
            labels = text.split()
            if not 1 <= len(labels) <= 12:
                raise ValueError(f"press takes 1-12 key labels separated by spaces: {rest[:60]!r}")
        if name == "pinyin":
            text = text.lower().replace("ü", "v").replace("u:", "v")
            text = re.sub(r"[\s'’]", "", text)  # a space would commit a candidate halfway
            if not re.fullmatch(r"[a-z]+", text):
                raise ValueError(f"pinyin takes letters only (no tones or Chinese characters): {rest[:60]!r}")
        return {"action": name, keys[0]: text}
    numbers = [int(float(n)) for n in re.findall(r"-?\d+(?:\.\d+)?", rest)]
    if keys == ("points",):
        if len(numbers) < 4 or len(numbers) % 2:
            raise ValueError(f"path needs 2+ x y pairs: {line[:100]}")
        return {"action": name, "points": [numbers[i:i + 2] for i in range(0, len(numbers), 2)][:20]}
    if len(numbers) != len(keys):
        raise ValueError(f"{name} needs {len(keys)} numbers: {line[:100]}")
    action = {"action": name, **dict(zip(keys, numbers, strict=True))}
    if name == "tap" and "precise" in rest.lower():
        action["precise"] = True
    return action


def parse_actions(text: str) -> list[dict]:
    """Parse a THINK / DO… reply (an optional ZH line is kept); the thought goes on the first action."""
    thought = re.search(r"^\s*THINK\s*[:：]\s*(.*?)\s*$", text, re.M | re.I)
    zh = re.search(r"^\s*ZH\s*[:：]\s*(.*?)\s*$", text, re.M | re.I)
    acts = [parse_command(line) for line in re.findall(r"^\s*DO\s*[:：]\s*(.*?)\s*$", text, re.M | re.I)][:8]
    if not acts:
        raise ValueError(f"no DO line in reply: {text[:200]}")
    acts[0] = {"thought": thought.group(1) if thought else "", "thought_zh": zh.group(1) if zh else "", **acts[0]}
    done = re.search(r"^\s*DONE\s*[:：]\s*(yes|no|true|false|是|否)([^\n]*)", text, re.M | re.I)
    if done:
        acts[0]["done"] = done.group(1).lower() in ("yes", "true", "是")
        through = re.search(r"(?:through|thru|up to|to|至)\s*(?:step\s*)?(\d+)", done.group(2), re.I)
        if acts[0]["done"] and through:
            acts[0]["done_through"] = int(through.group(1))
    step = re.search(r"^\s*STEP\s*[:：]\s*(\d+)", text, re.M | re.I)  # older reply format
    if step:
        acts[0]["step"] = int(step.group(1))
    return acts


def parse_plan(text: str) -> tuple[str, str]:
    """Split an optional EN: / ZH: plan reply; a plain reply is used for both."""
    m = re.search(r"EN\s*[:：](.*?)ZH\s*[:：](.*)", text, re.S | re.I)
    return (m.group(1).strip(), m.group(2).strip()) if m else (text.strip(), text.strip())


def plan_steps(plan: str) -> list[str]:
    """The numbered lines of a plan: "1. Tap Settings\\n2) Tap General" -> ["Tap Settings", "Tap General"]."""
    return [m.group(1) for m in re.finditer(r"^\s*\d+\s*[.)、:]\s*(.+?)\s*$", plan, re.M)]


def _key(step: dict) -> dict:
    """The action itself, without thoughts, timings and other bookkeeping."""
    return {k: v for k, v in step.items() if k not in META}


def _pointer_path(act: dict) -> list[tuple[float, float]]:
    """Screen points (fractions) the pointer passes through for one action."""
    f = lambda x, y: (x / 1000, y / 1000)  # noqa: E731
    match act["action"]:
        case "tap": return [f(act["x"], act["y"])]
        case "swipe": return [f(act["x1"], act["y1"]), f(act["x2"], act["y2"])]
        case "path": return [f(*act["points"][-1])]  # masking every point would hide the stroke itself
        case "circle": return [f(act["x"] + act["r"], act["y"])]
        case "home": return [f(x, y) for x, y in act.get("taps", [(500, 995), (500, 600)])]
        case "back": return [(0.005, 0.45), (0.6, 0.45)]
        case "press" | "tapon": return [f(x, y) for x, y in act.get("taps", [])]
        case _: return []


def circle(x: float, y: float, r: float, aspect: float, n: int = 24) -> list[tuple[float, float]]:
    """Points around a circle; r is a fraction of the width, aspect = width / height."""
    return [(x + r * math.cos(2 * math.pi * i / n), y + r * aspect * math.sin(2 * math.pi * i / n))
            for i in range(n + 1)]
