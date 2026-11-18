"""See-think-act loop: a planner model writes the plan, an actor model picks each step."""

import asyncio
import json
import logging
import math
import re
import time
from collections.abc import Callable
from pathlib import Path

from . import vision
from .debug import Trace
from .device import DeviceError
from .llm import LLM, cost
from .phone import Phone
from .screen import Screen
from .task_state import Recovery, TaskContext, TaskState

log = logging.getLogger(__name__)

PLANNER_PROMPT = """
APP LANGUAGE AND SPOTLIGHT SEARCH RULES:
- The iPhone system language is English.
- The user may give tasks in Chinese. Translate the requested app
  name to its official English iOS display name before searching.
- Common mappings:
  备忘录=Notes; 设置=Settings; 照片=Photos;
  提醒事项=Reminders; 日历=Calendar;
  时钟=Clock; 地图=Maps; 文件=Files;
  通讯录=Contacts; 信息=Messages;
  电话=Phone; 相机=Camera; 计算器=Calculator;
  健康=Health; 钱包=Wallet; 天气=Weather.
- For 备忘录, search "Notes", never the Chinese word "备忘录".
- After typing a search term, inspect the next screenshot
  before selecting a result.
- If no match appears, try a relevant English alternative,
  clear the previous query, and inspect the new results.
- Do not guess coordinates or tap an unverified search result.
You plan tasks on an iPhone. Given the goal and the current screen,
write a short numbered plan (at most 6 steps) of what to do on screen. Plain text only, in English.
Plan only what the goal needs, from what you can see; never invent accounts, emails, logins or
screens you have not seen. Later steps can stay general (e.g. "Draw the face on the canvas").
To open an app, prefer search over paging through home screens: tap the "Search" pill above
the dock (or swipe down from the middle of the home screen), type the app name, tap the result.
To reach the Home Screen write the step as "Go to the Home Screen": the actor does it in one action
through AssistiveTouch. Never plan a swipe up from the bottom edge (it does not work on this phone)."""


ACTOR_PROMPT = """You are an AI agent controlling a real iPhone 11
through an ESP32-S3 USB HID mouse and keyboard.

You receive screenshots of the actual iPhone screen.
Your task is to observe, decide, perform ONE action, and verify
the result on the next screenshot.

The iPhone system language is ENGLISH.
The user may give instructions in Chinese.

You must follow these rules carefully.

==================================================
1. RESPONSE FORMAT
==================================================

Reply in exactly this format:

DONE: <yes or no: does this screenshot show the [now] plan step finished?
      "yes through N" when it shows plan steps up to N finished>
THINK: <Briefly describe what is visible, whether the previous
action succeeded, and why you chose the next action.>
DO: <exactly one supported action>

Example:

DONE: yes
THINK: Spotlight is open with an empty search field, so the
"open search" step is done. The field is focused already.
DO: type Settings

PLAN PROGRESS:
- The plan you receive marks each step [done], [now] or [todo].
- [done] steps and the "Already completed" list were verified on
  earlier screens. NEVER redo them, even if the current screen
  (e.g. the Home Screen) looks like the start of the task.
- Seeing the Home Screen after a [done] "return to Home" step means
  you should continue with the NEXT step, not start over.
- DONE answers one question about the [now] step only: does this
  screenshot show its result? Example: [now] is "Open Settings" and
  the Settings list is on screen -> DONE: yes, and your DO line
  already works on the next step. Still needs actions -> DONE: no.
- Say yes as soon as the screen shows the result; do not wait for
  another round. Never say yes for a step whose result you cannot see.
- When the screenshot shows several steps finished at once, say
  DONE: yes through N (N = the last finished step). Example: the plan
  is 2 [now] open Settings, 3 [todo] tap General then About; Settings
  opens straight on the About page (apps reopen on the page you left)
  -> DONE: yes through 3. Then never redo steps 2-3: going Home and
  reopening Settings just shows the same page again.
- The goal outranks the plan. If this screenshot already shows the
  goal reached by this task's own actions (e.g. the result they
  computed), finish now with DO: done <answer>, even if plan steps
  are left. Never clear or undo a screen that answers the goal.
  A value already shown when an app opens (Calculator keeps its last
  result) is left over from before: it never counts as the answer.

IMPORTANT:
- Output exactly ONE DO line per response.
- Do not include Markdown code fences.
- Do not assume an action succeeded until a new screenshot confirms it.
- Use the visible screenshot, not imagined screen contents.

==================================================
2. SCREEN COORDINATES
==================================================

All coordinates use a 0-1000 scale.

X:
0 = left edge
1000 = right edge

Y:
0 = top edge
1000 = bottom edge

Examples:
tap 500 500
swipe 800 500 200 500

Always aim at the CENTER of a visible interactive element.

Never guess the location of an invisible button.

For small controls, search results, app icons, and list rows,
name them and let the system find them:

DO: tapon General row

==================================================
3. DEVICE-SPECIFIC NAVIGATION
==================================================

This device is an iPhone 11.

The physical connection uses an ESP32-S3 USB HID mouse
and keyboard.

Important known behavior:

- Ordinary mouse movement works.
- Mouse clicking works.
- Horizontal and vertical dragging can work.
- Bottom-edge Home swipes are unreliable on this device.

Therefore, to go to the Home Screen from anywhere:

DO: home

That is ONE action: the system finds the floating AssistiveTouch
button, opens its menu and taps Home for you. Do not tap the
AssistiveTouch button or its menu yourself, and do not swipe up
from the bottom edge. Check the next screenshot: if the Home Screen
did not appear, try DO: home once more before anything else.

==================================================
4. APP LANGUAGE MAPPING
==================================================

The iPhone operating system is configured in ENGLISH.

The user may refer to apps using Chinese names.

Translate Chinese app names into their official English
display names before using Spotlight.

Common app names:

设置 = Settings
备忘录 = Notes
照片 = Photos
提醒事项 = Reminders
日历 = Calendar
时钟 = Clock
地图 = Maps
文件 = Files
通讯录 = Contacts
信息 = Messages
电话 = Phone
相机 = Camera
计算器 = Calculator
健康 = Health
钱包 = Wallet
天气 = Weather
邮件 = Mail
音乐 = Music
快捷指令 = Shortcuts
语音备忘录 = Voice Memos
查找 = Find My
Safari浏览器 = Safari

Examples:

User requests:
"打开备忘录"

Spotlight search:
Notes

User requests:
"打开设置"

Spotlight search:
Settings

Do not type the Chinese app name when searching
for an English system app.

If an application is not in the mapping above,
infer its English display name cautiously.

If uncertain, inspect the visible search results
rather than inventing a name.

==================================================
5. SPOTLIGHT SEARCH WORKFLOW
==================================================

When opening an app that is not visible:

STEP A:
Determine whether the Home Screen is already visible.

STEP B:
If inside another app, return to the Home Screen
with DO: home.

STEP C:
Once on the Home Screen, look for the Search button
above the Dock.

STEP D:
If the Search button is visible, tap its center
using precise targeting.

If the Search button is not visible, use a downward
swipe from the middle of the Home Screen to open
Spotlight, then verify that Spotlight appeared.

STEP E:
Wait for a fresh screenshot showing Spotlight.

STEP F:
Locate the Spotlight search text field.

IMPORTANT:
Spotlight's search field has keyboard focus as soon as
Spotlight opens, and keeps it after "clear". Type right
away; do not tap the field first (on this iPhone 11 all
38 recorded "type" actions worked without that tap).
Tap the field only if a typed word did not appear.

STEP G:
Wait for another screenshot to check the input state.

An external HID keyboard may hide the iOS software keyboard.
Therefore, absence of the on-screen keyboard does not
necessarily mean that the input is unfocused.

STEP H:
If an existing query is present (Spotlight keeps the last one),
empty the field first:

DO: clear

and on the next turn:

DO: type Settings

STEP I:
Type the correct ENGLISH application name.

Example:

DO: type Notes

STEP J:
Wait for a new screenshot.

Verify that the expected search text appears and
that relevant search results are visible.

STEP K:
Locate the actual application result.

Distinguish application icons and app result rows from:

- Web search suggestions
- Siri suggestions
- Documents
- Contacts
- Other unrelated search matches

STEP L:
Tap the actual app result: the app icon under "Top Hit"
(Spotlight's first section). Rows under "Siri Suggestions" with a
Safari compass icon, or a magnifier icon, are web searches: tapping
"notes" there opens Safari, not Notes (iPhone 11, 2026-11-18). If the
app is not under Top Hit, use an app icon that shows the app's own
icon and name, never a row with a Safari or magnifier icon.

Example:

DO: tapon Notes app icon under Top Hit

STEP M:
Wait for a new screenshot and verify the requested
application actually opened.

Never report success before verifying the destination.

==================================================
6. TEXT INPUT RULES
==================================================

Before using "type":

- Confirm the intended text field is visible.
- Spotlight's field is focused when it opens: type at once.
  Other fields: tap them first if no cursor is shown.
- If a previous query exists, clear it first.
- Prefer English app names for system app search.
- After typing, inspect the next screenshot.
- If no text appeared, refocus the field.
- Do not repeatedly type into an unfocused screen.
- Do not mistake the absence of the soft keyboard
  for a definite typing failure.

For an existing search term in a focused field:

DO: clear

On the next turn:

DO: type Settings

Verify the resulting search text before selecting an app.

==================================================
7. CLICK ACCURACY
==================================================

The device uses relative USB mouse movements.

Mouse positioning can drift slightly.

For important taps on named things, use tapon (it is located
on the screenshot); for unnamed points, use:

DO: tap X Y precise

Especially for:

- Spotlight search results
- App icons
- Search fields
- Small buttons
- Settings list entries
- AssistiveTouch floating button

The software attempts to locate the actual pointer
and correct its position.

However, precise targeting is not guaranteed to
succeed every time.

After tapping:

- Wait for a fresh screenshot.
- Check whether the target screen appeared.
- If nothing changed, do not assume success.
- Reidentify the target and adjust the coordinates.
- Avoid clicking the same wrong location repeatedly.

==================================================
8. SWIPING AND SCROLLING
==================================================

The device supports mouse-based drag gestures.

Swipe format:

DO: swipe X1 Y1 X2 Y2

To move content upward:

DO: swipe 500 750 500 350

To move content downward:

DO: swipe 500 350 500 750

To switch Home Screen pages:

DO: swipe 850 500 200 500

Important:

- Only swipe in an appropriate visible area.
- Avoid dragging from the extreme bottom edge
  to trigger a Home gesture.
- Prefer moderate swipes when navigating lists.
- Take a fresh screenshot after each swipe.
- Verify whether content actually moved.
- If the content did not move, reassess the screen.

Do not endlessly scroll looking for an app.

For finding applications, prefer Spotlight search.

==================================================
9. FAILURE RECOVERY
==================================================

When an action fails:

1. Look at the current screenshot.
2. Determine whether the screen changed.
3. Identify the last action.
4. Do not blindly repeat it.
5. Choose a different recovery action.

Examples:

CASE 1:
A tap did not open an app.

Recovery:
Reidentify the actual app result and use precise tapping.

CASE 2:
Spotlight opened but typing failed.

Recovery:
Tap the search field, verify focus, and retry typing.

CASE 3:
Spotlight already contains an incorrect search term.

Recovery:
Focus the field, select the existing text, and replace it.

CASE 4:
The user requested Notes but the phone is still in Settings.

Recovery:
DO: home, then open Spotlight.

CASE 5:
A Home swipe did nothing.

Recovery:
Do not repeat the Home swipe.
Use DO: home instead.

CASE 6:
A swipe did not change the screen.

Recovery:
Reassess whether the gesture is appropriate for the
current screen before trying a different swipe.

CASE 7:
The expected control is not visible.

Recovery:
Do not invent coordinates.
Navigate to the appropriate page or report failure.

==================================================
10. VERIFY EVERY ACTION
==================================================

Every action must follow this loop:

OBSERVE
-> THINK
-> ACT ONCE
-> OBSERVE AGAIN
-> VERIFY

Never assume that:

- A tap opened an app.
- A swipe moved the screen.
- A text command entered text.
- Spotlight opened.
- The Home Screen appeared.
- The requested task is complete.

Use the actual screenshot as evidence.

If the previous action did not visibly change the screen,
consider whether the action failed or whether the
intended operation legitimately has no visible effect.

Do not confuse mouse pointer movement with navigation.

==================================================
11. SAFETY
==================================================

Do not perform irreversible actions without clear
authorization from the user's task.

Avoid unintended:

- Purchases
- Payments
- Deletions
- Sending messages
- Changing security settings
- Changing account credentials
- Sharing private information

If a task only requires viewing information,
do not modify settings.

Stop when the requested task is complete.

==================================================
12. SUPPORTED ACTIONS
==================================================

tap X Y [precise]
swipe X1 Y1 X2 Y2
path X1 Y1 X2 Y2 X3 Y3 ...
circle X Y R
type TEXT
pinyin LETTERS
clear        (empties the focused text field; use it instead of
              cmd+a or the field's x button)
press KEYS   (keypad keys by label, e.g. press 1 2 8 x 3 7 =; see KEYPADS)
tapon WHAT   (tap a named thing, located for you; see TAPON)
key COMBO
home
back
wait
replan
done ANSWER
fail REASON

CHINESE TEXT:
"type" sends ASCII keys only; Chinese characters typed with it come out as garbage.
To enter Chinese, use "pinyin" with the toneless pinyin letters of the whole phrase, e.g.
长安姑娘 -> DO: pinyin changanguniang (ü is v: 女 -> nv). It switches to the Chinese
keyboard, types the letters, picks the first candidate and switches back.
Then check the field: if a wrong word was picked, clear it and try a shorter phrase.
Search terms that are English words or app names still use "type".

APP NOTES:
- Calculator: it opens showing the last result, so begin with a full
  reset: tap the top-left key twice (C, then AC). The display reads 0.
- C clears only the number being typed; a pending + - x / stays and is
  applied to the next number (128 x, C, 128 x shows 16,384). To start
  over, always tap the top-left key twice, never C alone.
- Tapping + - x or / does not change the number shown; the key only
  turns lighter. Tap an operator once, then go on with the next
  number; never tap it again because the display looks the same.
- The app ignores the keyboard ("type" does nothing there).

NOTE:
"home" goes to the Home Screen through AssistiveTouch
in one action; use it whenever you need the Home Screen.

All coordinates must use the 0-1000 scale.

Examples:

THINK: Notes is open, but I need the Home Screen to
open another app.
DO: home

THINK: The Home Screen is visible and there is
a Search button above the Dock.
DO: tapon Search pill above the Dock

THINK: Spotlight is open and its search field is empty.
The user requested 备忘录, which is Notes in English.
DO: type Notes

THINK: The Notes app result is now visible.
I will tap its icon.
DO: tapon Notes app icon under Top Hit

THINK: The Notes app is open, so the requested
task has been completed.
DO: done Opened the Notes app successfully.

TAPON:
To tap something you can name (a button, list row, app icon, search
result, tab, pill), describe it instead of giving coordinates:
DO: tapon General row
DO: tapon Search pill above the Dock
DO: tapon Notes app icon under Top Hit
The system finds it on this screenshot and taps its center. Write a
short description that matches only that one thing: its visible text
plus what kind of element it is, and where when the text appears more
than once. Never write "tap X Y" for something you can name: your own
coordinates miss small targets far more often (the Search pill above
the Dock was missed again and again at y 790-890; tapon finds it).
Use "tap X Y" only for things with no name (an empty spot, a point on
a canvas, a picker wheel).

KEYPADS:
To enter a number or expression on a keypad that stays in place
(Calculator, a number pad), name the keys in ONE action:
DO: press 4 5 + 6 =
Labels are separated by spaces, in order (digits, . + - x / = C AC
or the key's text). The system finds each key on this screenshot and
taps it; do not give coordinates. Then check the result on the next
screenshot. If it is wrong, reset (top-left key twice) and press the
whole sequence again; never repair a half-entered number key by key.
The display shows only the number being typed (after 128 x then 3 it
shows 3). Use press only on keypads, never on lists, menus or text.

FINAL RULE:
Perform exactly ONE action per response,
then observe the next screenshot before continuing.
"""



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


class BlindError(RuntimeError):
    """The screen feed is gone; acting now would be guessing."""


FAST_ROUNDS = 3  # rounds on one plan step after which the actor thinks again
# Two wordings, asked together: each found the button where the other did not. iPhone 11, 2026-11-18: on a black
# Notes screen with the button at the bottom edge the first answered the mouse pointer (600, 600) four times in a
# row and home failed the task; the second found (750, 950) there but said none on another frame.
POINTER_PARK = (0.5, 0.5)  # where the pointer waits while the AssistiveTouch button is looked for: never at an edge
AT_BUTTON_Q2 = ("Where is the center of the AssistiveTouch floating button? It is a dark gray rounded square about one "
                "sixth of the screen width, with light concentric circles inside, usually at a screen edge. The mouse "
                "pointer is a plain gray dot about one twentieth of the screen width; it is never the button.")
AT_BUTTON_Q = ("Where is the center of the AssistiveTouch floating button (a translucent gray rounded square "
               "with a white circle in it, floating over the screen; not the small gray mouse pointer dot)?")
# names the panel, so the Home app icon on the Home Screen is not taken for the menu item (the old question did)
AT_HOME_Q = ("Is the AssistiveTouch menu open: a dark rounded panel floating over the screen with items such as "
             "Notification Center, Device, Control Center, Siri, Home and Custom? If it is, where is the center of the "
             "Home item inside that panel? An app icon named Home on the Home Screen is not it.")
# wording matters: "You locate things ... the thing the user describes" made the locator answer none for the
# AssistiveTouch button on 6 of 10 tries on two frames; this wording found it 10 of 10
LOCATE_PROMPT = ("You locate keys on iPhone screenshots. Coordinates use a 0-1000 scale for x (left to right) "
                 "and y (top to bottom). Reply with the center of the key the user names.")
# only for things named by their text (tapon, press): on the textless, faint AssistiveTouch button it made the
# locator answer none on 2 of 8 frames
EXACT_TEXT = (" Match its visible text exactly (same word and spelling): a label that only contains or resembles it "
              "is a different thing (Notes is not rednote). If it is not on this screenshot, say none.")
GOAL_CHECK = ("First look at the screenshot: if it shows the goal reached by the actions of this task (e.g. the "
              "result they computed), reply DO: done <answer> now, even with plan steps left; never undo it. "
              "Something already on screen when an app opened is left over from before and does not count.\n")
GRID_NOTE = ("\n\nThe screenshot has thin cyan grid lines every 100 units of the 0-1000 scale, labeled in "
             "yellow along the top and left edges. Read tap positions off them. They are not part of the screen.")


class Agent:
    def __init__(self, phone: Phone, screen: Screen, actor: LLM, planner: LLM, trace: Trace,
                 max_steps: int = 40, max_failures: int = 3, max_seconds: float = 300,
                 runs_file: Path | None = None, price: str | Callable[[float], str] = "", grid: bool = False,
                 at_double_tap_home: bool = False):
        self.phone, self.screen, self.actor, self.planner = phone, screen, actor, planner
        self.grid = grid  # draw a coordinate grid on the actor's screenshots (ACTOR_GRID=1)
        # the phone's AssistiveTouch Double-Tap action is set to Home (AT_DOUBLE_TAP_HOME=1)
        self.at_double_tap_home = at_double_tap_home
        self.trace = trace
        self.max_steps, self.max_failures = max_steps, max_failures
        self.max_seconds = max_seconds  # whole-task time limit: a hung step or a slow loop can't run forever
        self.runs_file, self.price = runs_file, price  # one summary line per run; LLM_PRICE for the cost
        self.usage: dict = {}  # this run: LLM calls, seconds and tokens
        self._started = 0.0
        self.task, self.plan, self.steps = "", "", []
        self.completed: list[str] = []  # plan steps finished so far; survives replans
        self.plan_step = 1  # 1-based step of the current plan being worked on; only moves forward
        self.shots: dict[int, bytes] = {}  # step index -> screenshot the model decided on
        self._pointer = (0.0, 0.0)  # last pointer position (screen fractions)
        # 构造时就有状态机：未 start() 时 UI / 测试也能读到状态，start() 会再重置一次
        self.task_ctx = TaskContext(task_name="")
        self._job: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self._job is not None and not self._job.done()

    def start(self, task: str) -> None:
        if self.running:
            raise RuntimeError("agent already running")
        if not self.screen.live:
            raise RuntimeError("no phone screen: start Screen Mirroring first")
        self.task, self.plan, self.steps, self.shots = task, "", [], {}
        self.completed, self.plan_step = [], 1
        self._pointer = (0.0, 0.0)
        self.usage, self._started = {}, time.time()
        self.task_ctx = TaskContext(task_name=task)
        self.trace.start(task)
        self._job = asyncio.create_task(self._run())

    def stop(self) -> None:
        if self.running:
            self._job.cancel()

    def clear(self) -> None:
        if self.running:
            raise RuntimeError("stop the agent first")
        self.task, self.plan, self.steps, self.shots = "", "", [], {}
        self.completed, self.plan_step = [], 1
        self.usage = {}
        self.task_ctx = TaskContext(task_name="")  # 清空后 UI 回到 idle

    def _record(self, shot: bytes | None = None, **step) -> None:
        if shot:
            self.shots[len(self.steps)] = shot
        self.steps.append({"t": time.time(), "shot": bool(shot), **step})
        self.trace.write(kind="step", **step)
        log.info("agent: %s", step)
        action = step.get("action", "")
        if action == "done":
            self.task_ctx.succeed()
        elif action in ("fail", "error", "stopped"):
            self.task_ctx.fail(
                str(step.get("reason", action))
            )
        elif action == "hiccup":
            self.task_ctx.transition(
                TaskState.RECOVER,
                str(step.get("reason", "执行异常"))
            )

    async def _ask(self, role: str, llm: LLM, system: str, prompt: str,
                   temperature: float = 0, fast: bool = False, why: str = "") -> tuple[str, float, bytes | None]:
        if not self.screen.live:  # never let the model act blind
            raise BlindError("lost the phone screen: reconnect Screen Mirroring")
        self.task_ctx.transition(
            TaskState.OBSERVE,
            f"读取当前截图，准备调用 {role}"
        )

        self.task_ctx.transition(
            TaskState.PLAN,
            f"调用 DeepSeek {role} 模型"
        )
        fast = fast and bool(getattr(llm, "fast_extra", None))  # only when FAST_BODY is configured
        frame, started = self.screen.frame, time.time()
        image = frame
        if self.grid and role == "actor" and frame:  # the trace keeps the clean frame
            image, system = vision.grid(frame), system + GRID_NOTE
        reply = await (llm.chat(system, prompt, image, temperature=temperature, fast=True) if fast
                       else llm.chat(system, prompt, image, temperature=temperature))
        seconds = round(time.time() - started, 2)
        used = getattr(llm, "last_usage", {}) or {}
        self.usage["calls"] = self.usage.get("calls", 0) + 1
        self.usage["llm_s"] = round(self.usage.get("llm_s", 0) + seconds, 2)
        self.usage["fast_calls"] = self.usage.get("fast_calls", 0) + fast
        if why and not fast:  # where the slow calls come from: think_missed, think_stuck, ... (count and seconds)
            self.usage[f"think_{why}"] = self.usage.get(f"think_{why}", 0) + 1
            self.usage[f"think_{why}_s"] = round(self.usage.get(f"think_{why}_s", 0) + seconds, 2)
        for k, v in used.items():
            self.usage[k] = self.usage.get(k, 0) + v
        self.trace.write(frame, kind="llm", role=role, model=llm.model, prompt=prompt,
                         reply=reply, seconds=seconds, fast=fast, why=why)
        return reply, seconds, frame

    async def _run(self) -> None:
        try:
            await self._attempt()
        finally:
            self._summarize()

    def _summarize(self) -> dict:
        """One line per run in runs.jsonl: outcome, steps, time, tokens, cost. The success rate and the
        cost per task come from these lines."""
        last = self.steps[-1] if self.steps else {}
        acts = [s for s in self.steps if s.get("action") not in ("plan", "replan (stuck)", "stalled", "hiccup")]
        run = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._started or time.time())),
               "task": self.task, "result": last.get("action", ""), "reason": str(last.get("reason", ""))[:200],
               "seconds": round(time.time() - self._started, 1) if self._started else 0,
               "actions": len(acts), "replans": sum(s.get("action") == "replan (stuck)" for s in self.steps),
               "hiccups": sum(s.get("action") == "hiccup" for s in self.steps),
               "models": {"actor": self.actor.model, "planner": self.planner.model},
               **self.usage}
        price = self.price(self._started or time.time()) if callable(self.price) else self.price
        run["cost"] = cost(self.usage, price)
        self.usage["cost"] = run["cost"]
        log.info("run: %s", run)
        if self.runs_file:
            try:
                with self.runs_file.open("a") as f:
                    f.write(json.dumps(run, ensure_ascii=False) + "\n")
            except OSError as e:
                log.warning("could not write %s: %s", self.runs_file, e)
        return run

    async def _attempt(self) -> None:
        failures, deadline = 0, time.time() + self.max_seconds
        try:
            await self._replan()
            for _ in range(self.max_steps):
                try:
                    if await asyncio.wait_for(self._step(), max(deadline - time.time(), 0.01)):
                        return
                    failures = 0
                except asyncio.TimeoutError:
                    raise
                except BlindError:
                    raise
                except Exception as e:  # a bad reply or a dropped link: recover, don't crash
                    self._record(action="hiccup", reason=str(e)[:300])
                    board_drop = isinstance(e, DeviceError) and await self.phone.device.wait_connected(15)
                    failures += 0 if board_drop else 1  # the board came back: not the agent's fault
                    if failures >= self.max_failures:
                        self._record(action="fail", reason=f"{failures} errors in a row")
                        return
                    await self.screen.settled()
                    await self._replan(stuck=True)
                if time.time() >= deadline:
                    raise asyncio.TimeoutError
            self._record(action="fail", reason="step limit reached")
        except asyncio.TimeoutError:
            self._record(action="fail", reason=f"time limit reached ({self.max_seconds:.0f} s)")
            await self._release()
        except asyncio.CancelledError:
            self._record(action="stopped")
        except Exception as e:  # e.g. the planner itself is unreachable
            self._record(action="error", reason=repr(e))

    async def _release(self) -> None:
        """A step cut off mid-gesture may leave the button down: let go and re-home."""
        try:
            if self.phone.device.connected:
                await self.phone.reset()
        except Exception as e:  # best effort; the task is over anyway
            log.warning("release after time limit failed: %s", e)

    async def _step(self) -> bool:
        """One see-think-act round. Return True when task is over."""

        acts = await self._next_actions()
        done = len(self.completed)
        self._advance(acts[0].get("step"), acts[0].get("done"), acts[0].get("done_through"))
        stall = await self._check_stall(len(self.completed) > done, acts[0])
        if stall is not Recovery.NONE:  # ABORT ends the task; after a REPLAN look again before acting
            return stall is Recovery.ABORT
        seen = acts[0]["_shot"]
        # the screen while the model was thinking: whatever moved since `seen` is animation, not our doing
        idle = self.screen.frame

        spots = [self._pointer]
        stuck = self._repeating(acts[0])

        for i, act in enumerate(acts):
            name = act["action"]

            # 保留原有的重复动作检测与重新规划机制
            if name == "replan" or stuck:
                self._record(act.pop("_shot"), **act)
                await self._replan(stuck=stuck)
                return False

            # 执行动作
            if name not in ("done", "fail"):
                started = time.time()
                await self._do(act)
                act["act_s"] = round(time.time() - started, 2)

                spots += _pointer_path(act)
                self._pointer = spots[-1]

            self._record(act.pop("_shot"), **act)

            # 保留原有任务终止逻辑
            if name in ("done", "fail"):
                return True

            following = (
                acts[i + 1]["action"]
                if i < len(acts) - 1
                else None
            )

            # 关键保护：点击输入框后，先确认屏幕有变化
            if name == "tap" and following in ("type", "key"):
                await self.screen.settled()

                self.task_ctx.transition(
                    TaskState.VERIFY,
                    "检查输入前的点击是否生效"
                )

                changed = vision.changed(
                    seen,
                    self.screen.frame,
                    ignore=tuple(spots),
                    idle=idle,
                )

                self.steps[-1]["changed"] = changed

                if not changed:
                    # 自动失败检测：状态机决定换动作重试 / 重新规划 / 放弃
                    return await self._handle_no_change("点击输入框后")

        # 本轮动作结束，等待屏幕稳定
        await self.screen.settled()

        self.task_ctx.transition(
            TaskState.VERIFY,
            "检查执行后的屏幕变化",
        )

        changed = vision.changed(
            seen,
            self.screen.frame,
            ignore=tuple(spots),
            idle=idle,
        )

        self.steps[-1]["changed"] = changed

        if not changed:
            # 自动失败检测：屏幕没有任何变化，按状态机给出的策略恢复
            return await self._handle_no_change(self.steps[-1].get("action", "动作"))

        self.task_ctx.record_change(True)  # 有进展：清零连败计数
        return False


    async def _handle_no_change(self, what: str) -> bool:
        """屏幕没有变化：让状态机决定重试 / 重新规划 / 放弃。返回 True 表示任务结束。"""
        recovery = self.task_ctx.record_change(False, what)
        self.steps[-1]["recovery"] = recovery.value
        if recovery is Recovery.ABORT:
            self._record(action="fail",
                         reason=f"连续 {self.task_ctx.no_progress} 步屏幕无变化（{what}）")
            return True
        if recovery is Recovery.REPLAN:
            await self._replan(stuck=True)
        return False


    async def _check_stall(self, progressed: bool, act: dict) -> Recovery:
        """Screen keeps changing but the plan step never finishes (wrong page, tapping around):
        replan after a while (REPLAN), give up the second time (ABORT, recorded as fail)."""
        steps = plan_steps(self.plan)
        current = steps[self.plan_step - 1] if 0 < self.plan_step <= len(steps) else ""
        recovery = self.task_ctx.record_round(progressed, current)
        if recovery is Recovery.NONE:
            return recovery
        rounds = self.task_ctx.max_rounds_per_step
        if recovery is Recovery.ABORT:
            self._record(act.pop("_shot"), action="fail",
                         reason=f"卡在步骤「{current}」：重新规划后仍然 {rounds} 轮没有完成")
        else:
            self._record(act.pop("_shot"), action="stalled", reason=f"步骤「{current}」{rounds} 轮未完成")
            await self._replan(stuck=True)
        return recovery

    def _missed(self) -> bool:
        """The previous round changed nothing on screen, so a tap may have landed off target."""
        return bool(self.steps) and self.steps[-1].get("changed") is False

    def _repeating(self, act: dict, limit: int = 3) -> bool:
        """True when the actor picks the same action as its last limit-1 moves and the last one did nothing.

        Repeating a move that still changes the screen (e.g. scrolling down a long list) is progress, not a loop.
        """
        moves = [s for s in self.steps if "plan" not in s][-(limit - 1):]
        return (len(moves) == limit - 1 and all(_key(m) == _key(act) for m in moves)
                and moves[-1].get("changed") is not True)

    def _advance(self, step: int | None, done: bool | None = None, through: int | None = None) -> None:
        """Move the plan forward: one step when the actor says the [now] step is `done` on screen, or up to
        `through` when it says the screen shows the steps up to there finished (Settings reopening straight on
        About covers "open Settings" and "tap General, About"; iPhone 11, 2026-11-18: advancing one step there
        sent the agent Home and back into Settings, which reopened on About again). A bare step number (older
        replies) still moves one step: "STEP: 4" from step 1 skipped work. Progress never goes back."""
        steps = plan_steps(self.plan)
        if not steps:
            return
        if done is not None:  # a yes/no is easier to get right than a step number, even without thinking
            if done and through and through >= self.plan_step:
                step = min(through + 1, len(steps) + 1)
                if step > self.plan_step:
                    self.completed += steps[self.plan_step - 1:step - 1]
                    self.plan_step = step
                return
            step = self.plan_step + 1 if done else None
        if not step:
            return
        step = min(step, self.plan_step + 1, len(steps) + 1)
        if step > self.plan_step:
            self.completed += steps[self.plan_step - 1:step - 1]
            self.plan_step = step

    def _progress(self) -> str:
        """The plan with [done] / [now] / [todo] marks, plus what earlier plans already finished."""
        steps = plan_steps(self.plan)
        if not steps:
            plan = self.plan
        else:
            mark = lambda i: "[done]" if i < self.plan_step else "[now]" if i == self.plan_step else "[todo]"  # noqa: E731
            plan = "\n".join(f"{i}. {mark(i)} {s}" for i, s in enumerate(steps, 1))
        done = "".join(f"\n- {s}" for s in self.completed)
        return (f"Already completed (verified earlier, NEVER redo):{done}\n\n" if done else "") + f"Plan:\n{plan}"

    async def _replan(self, stuck: bool = False) -> None:
        prompt = f"Goal: {self.task}"
        if self.completed:
            prompt += ("\n\nAlready completed (verified, do NOT plan these again; plan only the remaining work):"
                       + "".join(f"\n- {s}" for s in self.completed))
        if stuck:
            last = self.steps[-1] if self.steps else {}
            prompt += (f"\n\nThe previous plan got stuck (last event: {last.get('action')} {last.get('reason', '')}):"
                       f"\n{self.plan}\nLook at the current screen and plan from here, another way if needed.")
        reply, seconds, frame = await self._ask("planner", self.planner, PLANNER_PROMPT, prompt)
        self.plan, plan_zh = parse_plan(reply)
        self.plan_step = 1  # a new plan covers only what is left
        self.task_ctx.rounds_on_step = 0  # and gets a full round budget (stalls still count up)
        self._record(frame, action="replan (stuck)" if stuck else "plan", plan=self.plan, plan_zh=plan_zh,
                     think_s=seconds)

    def _think_reason(self, retry: str = "") -> str:
        """Why the actor should think this round instead of answering fast; "" = answer fast.

        rejected / repeated: its last reply this round was unparseable, or the move that just did nothing.
        missed: the previous action changed nothing on screen (a tap may have landed off target).
        stuck: FAST_ROUNDS rounds on one plan step (fast replies tend to keep the old STEP number after the
        screen moved on, which the stall budget reads as being stuck)."""
        if retry:
            return retry
        if self._missed():
            return "missed"
        if self.task_ctx.rounds_on_step >= FAST_ROUNDS:
            return "stuck"
        return ""

    async def _next_actions(self) -> list[dict]:
        # actions and their effect only: echoing earlier thoughts makes the model repeat them instead of looking
        hidden = ("t", "plan", "plan_zh", "think_s", "act_s", "shot", "thought", "thought_zh", "changed",
                  "recovery", "step", "done", "done_through")
        effect = {True: " -> screen changed", False: " -> screen did NOT change"}
        history = "\n".join(json.dumps({k: v for k, v in s.items() if k not in hidden})[:300]
                            + effect.get(s.get("changed"), "")
                            for s in self.steps[-8:] if "plan" not in s) or "(none)"
        prompt = (f"Goal: {self.task}\n\n{self._progress()}\n\nRecent actions:\n{history}\n\n"
                  f"{GOAL_CHECK}Next action? Start with DONE: yes or no (is the [now] step finished on this "
                  "screen?).")
        retry = ""  # why the previous reply in this round was refused
        for attempt in range(3):  # retry unparseable replies, a little less deterministically
            # routine step: answer fast; think it through when there is a reason to look harder
            why = self._think_reason(retry)
            reply, seconds, frame = await self._ask("actor", self.actor, ACTOR_PROMPT, prompt, 0.4 * attempt,
                                                    fast=not why, why=why)
            try:
                acts = parse_actions(reply)
            except ValueError as e:
                log.warning("unparseable actor reply (%s): %s", e, reply[:200])
                if reply.strip():  # tell the model what was wrong instead of asking the same thing again
                    prompt += f"\n\nYour last reply was rejected: {e}"
                retry = "rejected"
                continue
            if self._missed() and _key(acts[0]) == _key(self.steps[-1]):  # same move that just did nothing
                log.info("actor repeated an action that changed nothing; asking for another")
                prompt += (f"\n\nYou just proposed {json.dumps(_key(acts[0]))} again, but it changed nothing last "
                           "time. Choose a different action (another position, a shorter swipe, or another way).")
                retry = "repeated"
                continue
            acts[0]["think_s"] = seconds
            return [{**a, "_shot": frame} for a in acts]  # every action is drawn on the same screenshot
        raise ValueError(f"actor gave no valid action: {reply[:200]}")

    async def _press_key(self, label: str, x: int, y: int) -> bool:
        """Tap one keypad key; True when the screen changed (a digit shows up on the display)."""
        before, started = self.screen.frame, time.time()
        await self.phone.tap(x / 1000, y / 1000, aim=True)
        await self.screen.settled(quiet=0.15, least=0.3, most=1.0)
        hit = vision.changed(before, self.screen.frame)
        log.info("[PRESS] %s at (%d, %d): %s in %.2f s", label, x, y, "screen changed" if hit else "NO CHANGE",
                 time.time() - started)
        return hit

    async def _locate(self, question: str, by_text: bool = False) -> tuple[int, int] | None:
        """One fast question about the current frame: where is <thing>? (0-1000), None if it is not there.
        by_text: the thing is named by its visible text, which must match exactly."""
        frame = self.screen.frame
        image = vision.grid(frame) if self.grid else frame
        system = LOCATE_PROMPT + (EXACT_TEXT if by_text else "") + (GRID_NOTE if self.grid else "")
        text = question + " If there is none, answer: none. Otherwise answer with one line: X Y"
        started = time.time()
        reply = await (self.actor.chat(system, text, image, max_tokens=300, fast=True)
                       if getattr(self.actor, "fast_extra", None) else self.actor.chat(system, text, image))
        self.usage["calls"] = self.usage.get("calls", 0) + 1
        self.usage["locate_s"] = round(self.usage.get("locate_s", 0) + time.time() - started, 2)
        for k, v in (getattr(self.actor, "last_usage", {}) or {}).items():
            self.usage[k] = self.usage.get(k, 0) + v
        last = reply.strip().splitlines()[-1] if reply.strip() else ""
        nums = re.findall(r"\d+(?:\.\d+)?", last)
        spot = None
        if "none" not in last.lower() and len(nums) >= 2:
            x, y = round(float(nums[-2])), round(float(nums[-1]))
            spot = (x, y) if 0 <= x <= 1000 and 0 <= y <= 1000 else None
        # the frame it looked at: iPhone 11, 2026-11-18 a tapon went to 'rednote' although the actor's screenshot
        # showed Notes; the layout had likely changed since (Spotlight results load late)
        self.trace.write(frame, kind="locate", question=question, reply=reply[-120:], spot=spot)
        return spot

    async def go_home(self) -> list[tuple[int, int]]:
        """The home action outside a task (e.g. before a regression run starts)."""
        if self.running:
            raise RuntimeError("agent is running")
        return await self._home()

    async def _home(self) -> list[tuple[int, int]]:
        """Home Screen in one action: tap the AssistiveTouch button, then Home in its menu, each located on the
        current frame. iPhone 11, 2026-11-18: going home took the actor 2-4 rounds (button, look, Home, look,
        misses in between); cmd+h does nothing on iPhone. Offline: button found 8/8, menu Home 8/8 (the button
        moves between runs, so it is never a fixed position). No button: the old bottom-edge swipe."""
        button = await self._locate_at_button()
        if button is None:  # the menu may be open already (it hides the button)
            item = await self._locate(AT_HOME_Q)
            if item is not None:
                await self._tap_until_changed(item)
                log.info("[HOME] AssistiveTouch menu was open; Home at %s", item)
                return [item]
            log.warning("[HOME] no AssistiveTouch button on screen; swiping up from the bottom edge")
            await self.phone.go_home()
            return [(500, 995), (500, 600)]
        if self.at_double_tap_home:
            return await self._home_by_double_tap(button)
        item = None
        for attempt in range(2):  # iPhone 11, 2026-11-18: the menu did not show up after the first tap twice
            await self.phone.tap(button[0] / 1000, button[1] / 1000, aim=True)
            for wait in (0.5, 1.0):  # it may still be opening
                await self.screen.settled(quiet=0.2, least=wait, most=1.5)
                item = await self._locate(AT_HOME_Q)
                if item is not None:
                    break
            if item is not None:
                break
            log.warning("[HOME] AssistiveTouch menu did not open after tap %d at %s", attempt + 1, button)
        if item is None:
            raise ValueError("home: the AssistiveTouch menu did not open (no Home item on screen)")
        await self._tap_until_changed(item)
        log.info("[HOME] AssistiveTouch at %s, Home at %s", button, item)
        return [button, item]

    async def _locate_at_button(self) -> tuple[int, int] | None:
        """The AssistiveTouch button. The mouse pointer is parked in the middle of the screen first (a move, no
        click): the button always sits at an edge, so an answer there is the pointer. Both wordings are asked in
        parallel and the first answer away from the pointer wins. iPhone 11, 2026-11-18: right after a restart the
        pointer's place was unknown and the locator's (500, 600), the pointer, was double-tapped twice."""
        await self.phone.point(*POINTER_PARK)
        self._pointer = POINTER_PARK
        await self.screen.settled(quiet=0.2, least=0.4, most=1.0)
        found = await asyncio.gather(self._locate(AT_BUTTON_Q2), self._locate(AT_BUTTON_Q))
        park = (POINTER_PARK[0] * 1000, POINTER_PARK[1] * 1000)
        for spot in found:
            if spot is not None and math.dist(spot, park) > 120:
                return spot
        log.warning("[HOME] AssistiveTouch button not found (answers %s; the pointer is parked at %s)", found, park)
        return None

    async def _home_by_double_tap(self, button: tuple[int, int]) -> list[tuple[int, int]]:
        """The phone's AssistiveTouch Double-Tap action is Home: double-tap the button in place (one located spot,
        no menu). Taken as a single tap, it opens the menu (Single-Tap stays Open Menu): then tap Home there."""
        spot = (button[0] / 1000, button[1] / 1000)
        for attempt in range(2):
            before = self.screen.frame
            await self.phone.double_tap(*spot, aim=True)
            await self.screen.settled(quiet=0.3, least=0.8, most=2.0)  # iOS waits out the double-tap timeout
            item = await self._locate(AT_HOME_Q)
            if item is not None:  # counted as a single tap: the menu opened
                log.warning("[HOME] double tap opened the menu; tapping its Home at %s", item)
                await self._tap_until_changed(item)
                return [button, item]
            if vision.changed(before, self.screen.frame, ignore=(spot,)):
                log.info("[HOME] double tap on AssistiveTouch at %s", button)
                return [button]
            log.warning("[HOME] double tap %d at %s changed nothing", attempt + 1, button)
        raise ValueError("home: double-tapping AssistiveTouch changed nothing")

    async def _tap_until_changed(self, spot: tuple[int, int], tries: int = 2) -> bool:
        """Tap, wait for the screen, and tap once more if nothing changed. iPhone 11, 2026-11-18: the first tap
        on the AssistiveTouch menu's Home (located right) did nothing, probably while the menu still opened."""
        for _ in range(tries):
            before = self.screen.frame
            await self.phone.tap(spot[0] / 1000, spot[1] / 1000, aim=True)
            await self.screen.settled(quiet=0.3, least=0.6, most=2.0)
            if vision.changed(before, self.screen.frame, ignore=((spot[0] / 1000, spot[1] / 1000),)):
                return True
            log.warning("[HOME] tap at %s changed nothing; tapping again", spot)
        return False

    async def _locate_keys(self, labels: list[str]) -> list[tuple[int, int]]:
        """Positions (0-1000) of keypad keys named by label, one fast question per distinct label, in parallel.
        iPhone 11 Calculator, 2026-11-18: asked one key at a time the actor found 16/16 keys, but writing a
        7-tap sequence of coordinates itself it got 0/16 sequences right (1 2 8 came out as 1 2 3)."""
        distinct = list(dict.fromkeys(labels))
        spots = await asyncio.gather(*(self._locate(f"Where is the center of the key labeled '{k}'?", by_text=True)
                                       for k in distinct))
        missing = [k for k, p in zip(distinct, spots, strict=True) if p is None]
        if missing:
            raise ValueError(f"could not find key(s) {' '.join(missing)} on screen")
        found = dict(zip(distinct, spots, strict=True))
        log.info("[PRESS] %s", {k: found[k] for k in labels})
        return [found[k] for k in labels]

    async def _do(self, a: dict) -> None:
        self.task_ctx.transition(
            TaskState.ACT,
            f"执行操作：{a['action']}"
        )
        p = lambda k: a[k] / 1000  # noqa: E731
        match a["action"]:
            case "tap": await self.phone.tap(p("x"), p("y"), aim=True)
            case "swipe": await self.phone.swipe(p("x1"), p("y1"), p("x2"), p("y2"))
            case "path": await self.phone.drag([(x / 1000, y / 1000) for x, y in a["points"]])
            case "circle": await self.phone.drag(circle(p("x"), p("y"), p("r"), self.screen.aspect))
            case "type": await self.phone.type(a["text"])
            case "pinyin": await self.phone.pinyin(a["text"])
            case "clear": await self.phone.clear_field()
            case "press":
                a["taps"] = await self._locate_keys(a["keys"].split())
                a["registered"], a["retapped"] = [], []
                for label, (x, y) in zip(a["keys"].split(), a["taps"], strict=True):
                    hit = await self._press_key(label, x, y)
                    # iPhone 11, 2026-11-18: the first key of a sequence was lost 3 times in one run (128 came
                    # out as 28; not reproduced in 10 later sequences). A digit always changes the display, so one
                    # more tap is safe; operators and C/AC are never repeated (twice can change the meaning).
                    if not hit and re.fullmatch(r"[0-9.]", label):
                        log.warning("[PRESS] %s did not register; tapping it again", label)
                        a["retapped"].append(label)
                        hit = await self._press_key(label, x, y)
                    a["registered"].append(hit)
            case "key": await self.phone.key(a["combo"])
            case "home": a["taps"] = await self._home()
            case "tapon":
                spot = await self._locate(f"Where is the center of the {a['what']}?", by_text=True)
                if spot is None:
                    raise ValueError(f"tapon: could not find {a['what']!r} on screen")
                a["taps"] = [spot]
                await self.phone.tap(spot[0] / 1000, spot[1] / 1000, aim=True)
            case "back": await self.phone.go_back()
            case "wait": pass
            case other: raise ValueError(f"unknown action {other!r}")


def _pointer_path(act: dict) -> list[tuple[float, float]]:
    """Screen points (fractions) the pointer passes through for one action."""
    f = lambda x, y: (x / 1000, y / 1000)  # noqa: E731
    match act["action"]:
        case "tap": return [f(act["x"], act["y"])]
        case "swipe": return [f(act["x1"], act["y1"]), f(act["x2"], act["y2"])]
        case "path": return [f(*p) for p in act["points"]]
        case "circle": return [f(act["x"] + act["r"], act["y"])]
        case "home": return [f(x, y) for x, y in act.get("taps", [(500, 995), (500, 600)])]
        case "back": return [(0.005, 0.45), (0.6, 0.45)]
        case "press" | "tapon": return [f(x, y) for x, y in act.get("taps", [])]
        case _: return []


def circle(x: float, y: float, r: float, aspect: float, n: int = 24) -> list[tuple[float, float]]:
    """Points around a circle; r is a fraction of the width, aspect = width / height."""
    return [(x + r * math.cos(2 * math.pi * i / n), y + r * aspect * math.sin(2 * math.pi * i / n))
            for i in range(n + 1)]
