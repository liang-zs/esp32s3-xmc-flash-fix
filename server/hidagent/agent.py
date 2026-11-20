"""See-think-act loop: a planner model writes the plan, an actor model picks each step."""

import asyncio
import json
import logging
import math
import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

from . import vision
from .apps import app_hints
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
through AssistiveTouch. Never plan a swipe up from the bottom edge (it does not work on this phone).
Apps marked third-party in the goal's app names: open them from the App Library (go to the Home Screen,
swipe left past the last page, tap the App Library search field, type the name, tap the app icon): it lists apps
only, while Spotlight may offer just the app's page in Settings or web searches. Also when Spotlight's Top Hit
for an app is a gear icon with that app's name (its page in Settings).
Never plan changes to iPhone settings (keyboards, languages, accessibility, ...) to work around a failure, unless
the goal asks for that setting: plan another way, or end with a step that reports the failure.
What the agent can do: tap or describe-and-tap things, swipe, type ASCII text, enter Chinese through Pinyin, press
keypad keys, go to the Home Screen, go back. It cannot speak or listen (no dictation, no voice Siri), cannot use
the camera, Face ID or a passcode, and cannot read sound: never plan steps that need those.
Chinese text must end up as Chinese characters: plan a step "enter <the Chinese text> with pinyin" (the agent
switches keyboards itself); never settle for Latin pinyin letters in its place, and delete such letters first.
Drawing: plan each part as a stroke (dots and eyes are small circles, a mouth or leaf a curved line); never plan
taps on the canvas while a pen is active, not even to deselect (the gray dot there is the mouse pointer). Start on a
new, empty board unless the goal says to add to an existing drawing.
Full-width punctuation in the goal (！？，。：；) is only the user's Chinese keyboard: next to Latin text it means the
ASCII mark (输入Hello world！ -> type "Hello world!"). Never plan a hunt for symbols on the on-screen keyboard."""


ACTOR_PROMPT = """You control a real iPhone 11 through a USB mouse and keyboard (an ESP32-S3 board). Each turn you get a
screenshot, decide ONE action, and check its effect on the next screenshot. The iPhone's language is English;
the user may write in Chinese.

REPLY FORMAT (nothing else, no Markdown):
DONE: <yes or no: does this screenshot show the [now] plan step finished?
      "yes through N" when it shows plan steps up to N finished>
THINK: <what you see, whether the last action worked, why this action>
DO: <one action>

PLAN PROGRESS:
- The plan marks steps [done], [now], [todo]. [done] steps and "Already completed" were verified earlier:
  NEVER redo them, even if the screen (e.g. the Home Screen) looks like the start of the task. The Home Screen
  after a [done] "return to Home" step means: go on with the NEXT step.
- DONE is about the [now] step only: yes as soon as this screenshot shows its result (e.g. [now] "Open
  Settings" and the Settings list is on screen; your DO then already works on the next step); no while it
  still needs actions. Never yes for a result you cannot see: a step counts only when its OWN result shows
  ("Tap About" is finished when the About page is open, not when the About row is visible in General).
- Several steps finished at once: DONE: yes through N. Example: 2 [now] open Settings, 3 [todo] tap General
  then About; Settings opens straight on About (apps reopen on the page you left) -> DONE: yes through 3. Then
  never redo steps 2-3: going Home and reopening Settings shows the same page again.
- The goal outranks the plan: if this screenshot shows the goal reached by this task's own actions (e.g. the
  result they computed), DO: done <answer> now, even with steps left; never clear or undo it. A value already
  shown when an app opens (Calculator keeps its last result) is left over and never counts.

COORDINATES: 0-1000 on both axes, x left to right, y top to bottom; aim at the center of a visible element.
Never guess where an invisible control is.

ACTIONS:
tapon WHAT     tap a named thing; the system locates it on this screenshot (see TAPON)
tap X Y        an unnamed point only (empty spot, canvas, picker wheel)
press KEYS     keypad keys by label, e.g. press 1 2 8 x 3 7 = (see KEYPADS)
swipe X1 Y1 X2 Y2   content up: swipe 500 750 500 350; down: swipe 500 350 500 750;
               next Home Screen page / App Library: swipe 850 450 150 450
path X1 Y1 X2 Y2 ...   circle X Y R   (drawing; see DRAWING)
type TEXT      ASCII text into the focused field
pinyin LETTERS Chinese text (see CHINESE)
clear          empty the focused text field (instead of cmd+a or the field's x button)
key COMBO      e.g. key enter
home           go to the Home Screen (see HOME)
back           wait           replan
done ANSWER    fail REASON

TAPON: describe what to tap instead of giving coordinates: "DO: tapon General row", "DO: tapon Search pill
above the Dock", "DO: tapon Notes app icon under Top Hit". Name its visible text plus what kind of element it
is, and where, if that text appears more than once. Never write "tap X Y" for something you can name: your own
coordinates miss small targets far more often (the Search pill was missed again and again at y 790-890).

DRAWING: draw every part as a stroke big enough to see: a dot or an eye is "circle X Y 15" or bigger, a mouth
or leaf a path of several points. Never "tap" on a canvas while a pen is active (it leaves a dot you cannot
see, under the pointer), not even to deselect. The plain gray dot on the canvas is the mouse pointer, not a
selection handle. Unless the goal says to draw on the existing drawing, start on a new, empty board.

HOME: "DO: home" is ONE action: the system double-taps the AssistiveTouch button (or uses its menu) for you.
Never tap AssistiveTouch or its menu yourself, never swipe up from the bottom edge (unreliable here). If the
Home Screen did not appear, DO: home once more before anything else.

OPENING AN APP:
1. DO: home (unless the Home Screen shows), then DO: tapon Search pill above the Dock (or swipe down from the
   middle of the Home Screen) to open Spotlight.
2. Spotlight's field is focused when it opens and after "clear": type at once, no tap on the field first (all
   38 recorded "type" actions worked). It keeps the last query: DO: clear first, then DO: type <name>.
3. Search the app's ENGLISH name (the goal lists Chinese -> English names; otherwise infer the English
   display name, or check the results). Never type Chinese app names.
4. Tap the app icon under "Top Hit". Rows under "Siri Suggestions" with a Safari compass or a magnifier icon
   are web searches (tapping "notes" there opens Safari). A Top Hit with the gray gear icon but ANOTHER app's
   name ("bilibili", and "bilibili — Settings" in the field) is that app's page in Settings, not the app; only
   "Settings" itself is the Settings app.
5. Apps marked third-party in the goal, or offered only as such a Settings page: use the App Library (apps
   only): DO: home, swipe left page by page until the page with the "App Library" search field at the top,
   DO: tapon App Library search field, DO: type <name>, DO: tapon <name> app icon.
6. Check on the next screenshot that the app really opened.

TEXT INPUT: type sends ASCII only. Other fields than Spotlight: tap them first if no cursor shows. After
typing, check the text appeared; if not, refocus the field. A hidden on-screen keyboard does not mean the
field lost focus (the USB keyboard can hide it). The input-source picker (a small list: English (US), 简体拼音,
Emoji, ...) closes by itself: DO: wait, never tap it or near it (a tap there hit a search suggestion).
Never tap on-screen keyboard keys for letters or punctuation:
type them (full-width ！？，。 in the goal next to Latin text: type ! ? , .).

CHINESE: "type" turns Chinese characters into garbage. Use pinyin with the toneless letters of the whole
phrase: 长安姑娘 -> DO: pinyin changanguniang (ü is v: 女 -> nv). It switches to the Pinyin keyboard, types,
takes the first candidate and switches back. Check the field; a wrong word: clear it and try a shorter phrase.
English words and app names still use "type".

KEYPADS: on a keypad that stays in place (Calculator, a number pad) enter the whole number or expression in
ONE action: DO: press 4 5 + 6 = (labels in order: digits, . + - x / = C AC or the key's text). Then check the
result. Wrong: reset (top-left key twice) and press the whole sequence again; never repair a half-entered
number key by key. The display shows only the number being typed (after 128 x then 3 it shows 3).

APP NOTES:
- Calculator: opens showing the last result: begin with a full reset, the top-left key twice (C, then AC).
  C alone clears only the number being typed; a pending + - x / stays (128 x, C, 128 x shows 16,384). Tapping
  an operator does not change the display (the key only turns lighter): tap it once. It ignores "type".
- Notes: the new-note button (square with a pencil) is at the BOTTOM right of the notes list.

WHEN SOMETHING DID NOT WORK: look at the screenshot, decide whether it changed, and never repeat the same
action blindly; choose a different one (tapon with a better description, clear and retype, home). A pointer
moving is not navigation. A legitimately invisible effect (e.g. an operator key) is not a failure. A control
you cannot see: navigate to it or DO: fail <reason>; never invent coordinates. Do not scroll endlessly to find
an app: search for it.

SAFETY: no purchases, payments, deletions, messages, credential or security changes, sharing private data
unless the task asks. Never change iPhone settings (keyboards, languages, accessibility, AssistiveTouch,
display, privacy, ...) to work around a problem unless the task asks for that setting; try another way or
DO: fail <reason>. Viewing tasks change nothing. Stop when the task is complete.

EXAMPLE:
DONE: yes
THINK: Spotlight is open with an empty search field, so the "open search" step is done. The user asked for
备忘录, which is Notes in English.
DO: type Notes
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


# The board answers "err phone usb asleep" when its USB link to the iPhone is down (phone locked, or the iPhone
# stopped using the board, e.g. after it rebooted on poor power). Retrying or replanning cannot fix that: on
# 2026-11-20 a bilibili task replanned 9 times in a minute and never ended. Stop on the second one in a row.
USB_ASLEEP = "usb asleep"
USB_ASLEEP_STOP = 2
USB_ASLEEP_REASON = ("板子和 iPhone 之间的 USB 不通（板子报 phone usb asleep）：确认手机已解锁，"
                     "把板子接 iPhone 的线拔下重插（另一个口要接电源），然后重新运行任务")


# Ask for help instead of failing (tasks started from the web console): the task pauses, the user picks replan
# (optionally with a hint the models then see), continue, or stop. Unattended runs (tools/regress.py) still fail.
HELP_WAIT_S = 300   # nobody answers in this long: stop, as before
MAX_HELPS = 3       # per task, so a hopeless task still ends
HELP_CHOICES = ("replan", "continue", "stop")


AT_AVOID = 70  # a coordinate tap this close to the AssistiveTouch button (in screen-width units) is refused
DRAWING = ("circle", "path")  # strokes: a missed "no change" here says little (see _step)


class GiveUp(Exception):
    """The task is about to fail; _attempt asks the user (outside the step's time limit) before it does."""

    def __init__(self, reason: str, shot: bytes | None = None):
        super().__init__(reason)
        self.reason, self.shot = reason, shot


class BlindError(RuntimeError):
    """The screen feed is gone; acting now would be guessing."""


class LandscapeError(BlindError):
    """The phone shows a landscape screen (e.g. a full-screen video). Pointer calibration, the AssistiveTouch spot
    and every coordinate assume portrait; iPhone 11, 2026-11-19: a bilibili video went full screen in landscape,
    the AssistiveTouch button was hidden and two tasks double-tapped the video (likely 'liking' it) until they
    failed. Stop and ask for portrait instead."""


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
# iPhone 11, 2026-11-19: with the keyboard up the button sat over its bottom row (next to the microphone) and
# neither wording found it on 7 of 8 frames; with this sentence one of them did on 7 of 8 (no change without one)
AT_KEYBOARD = (" With the on-screen keyboard showing, it often sits over the keyboard's bottom row (next to the globe "
               "or microphone key) or just above the keyboard; it is not a keyboard key.")
AT_HOME_Q = ("Is the AssistiveTouch menu open: a dark rounded panel floating over the screen with items such as "
             "Notification Center, Device, Control Center, Siri, Home and Custom? If it is, where is the center of the "
             "Home item inside that panel? An app icon named Home on the Home Screen is not it.")
# wording matters: "You locate things ... the thing the user describes" made the locator answer none for the
# AssistiveTouch button on 6 of 10 tries on two frames; this wording found it 10 of 10
KEYBOARD_SWITCH = 1.2  # seconds after ctrl+space before looking: by hand, 1.5 s worked and 0.4 s did not
ONE_WORD_PROMPT = "You look at iPhone screenshots and answer questions about them with one word."
KEYBOARD_Q = ("Which on-screen keyboard is showing? Look at the space bar: 'pinyin' if it reads 空格 or 选定 (Chinese "
              "Simplified Pinyin), 'english' if it reads space, 'other' for any other keyboard (emoji, Japanese, "
              "numbers), 'none' if no keyboard is showing. Answer with that one word.")
AT_BUTTON_AT_Q = ("Is the AssistiveTouch floating button (a translucent gray rounded square with a white circle in it) "
                   "centered at about x={x}, y={y} on the 0-1000 grid, within 40 units? It may sit over the "
                   "keyboard's bottom row or a toolbar. The small plain gray dot is the mouse pointer, not the "
                   "button. Answer yes or no.")
# The screen turned landscape (a full-screen video): nothing can be done in landscape, but the goal may already be
# reached (bilibili-search, 2026-11-20: the first video opened, went full screen, and the run counted as failed).
LANDSCAPE_DONE_PROMPT = ("You check whether a task on an iPhone is finished, from one screenshot. The phone shows a "
                         "landscape screen now, e.g. a video playing full screen.")
LANDSCAPE_DONE_Q = ("Goal: {task}\nIs this goal achieved on this screen (only what the goal asks for counts)? Reply "
                    "with one line for the user, in the goal's language, describing what the screen shows, then a last "
                    "line: yes or no.")
PORTRAIT_WAIT = 2.0  # a landscape frame mid-task can be a mirroring glitch: wait this long for portrait to return

IS_HOME_PROMPT = "You look at iPhone screenshots and answer questions about them with one word: yes or no."
IS_HOME_Q = ("Is this the iPhone Home Screen: a page of app icons with names, with the Dock row of apps at the "
             "bottom (or the App Library page), and no app open, no full-screen video, no search screen? "
             "Answer yes or no.")
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
        self.ask_help = False  # this task pauses and asks before giving up (see GiveUp)
        self.help: dict | None = None  # the open question while paused: reason, since
        self.user_hints: list[str] = []  # hints the user gave; shown to both models
        self.helps = 0
        self._decision: asyncio.Future | None = None

    @property
    def running(self) -> bool:
        return self._job is not None and not self._job.done()

    def start(self, task: str, ask_help: bool = False) -> None:
        if self.running:
            raise RuntimeError("agent already running")
        if not self.screen.live:
            raise RuntimeError("no phone screen: start Screen Mirroring first")
        self.task, self.plan, self.steps, self.shots = task, "", [], {}
        self.completed, self.plan_step = [], 1
        self._pointer = (0.0, 0.0)
        self.usage, self._started = {}, time.time()
        self.task_ctx = TaskContext(task_name=task)
        self.ask_help, self.help, self.user_hints, self.helps, self._decision = ask_help, None, [], 0, None
        self.trace.start(task)
        self._job = asyncio.create_task(self._run())

    def stop(self) -> None:
        if self.running:
            self._job.cancel()

    def decide(self, choice: str, hint: str = "") -> None:
        """The user's answer while the task is paused: replan (with an optional hint), continue, or stop."""
        if choice not in HELP_CHOICES:
            raise ValueError(f"choice must be one of {', '.join(HELP_CHOICES)}")
        if not self._decision or self._decision.done():
            raise RuntimeError("the agent is not waiting for an answer")
        self._decision.set_result((choice, hint.strip()[:500]))

    def _quit(self, reason: str, shot: bytes | None = None) -> bool:
        """Give up on the task: ask first when this task may (raises GiveUp), else record the failure."""
        if self.ask_help and self.helps < MAX_HELPS:
            raise GiveUp(reason, shot)
        self._record(shot, action="fail", reason=reason)
        return True

    async def _give_up(self, reason: str, shot: bytes | None = None) -> bool:
        """Pause and ask the user what to do. True = the task ends (failure recorded)."""
        if not (self.ask_help and self.helps < MAX_HELPS):
            self._record(shot, action="fail", reason=reason)
            return True
        self.helps += 1
        self._record(shot or self.screen.frame, action="ask", reason=reason)
        self.help = {"reason": reason, "since": time.time()}
        self._decision = asyncio.get_running_loop().create_future()
        try:
            choice, hint = await asyncio.wait_for(self._decision, HELP_WAIT_S)
        except asyncio.TimeoutError:
            choice, hint = "stop", ""
            reason += f"（{HELP_WAIT_S // 60} 分钟无人回应）"
        finally:
            self.help, self._decision = None, None
        self._record(action="user", choice=choice, hint=hint)
        if choice == "stop":
            self._record(action="fail", reason=reason)
            return True
        if hint:
            self.user_hints.append(hint)
        self.task_ctx.no_progress = self.task_ctx.rounds_on_step = self.task_ctx.stalls = 0
        if choice == "replan":
            await self._replan(stuck=True)
        return False

    def _hint_text(self) -> str:
        return ("\nHints from the person watching the phone (trust them over your own guesses):"
                + "".join(f"\n- {h}" for h in self.user_hints)) if self.user_hints else ""

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
        if self.screen.aspect > 1:
            raise LandscapeError("the phone screen is landscape (a full-screen video?): turn it back to portrait "
                                 "(and lock portrait orientation), then run the task again")
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
        acts = [s for s in self.steps if s.get("action") not in ("plan", "replan (stuck)", "stalled", "hiccup", "ask",
                                                                 "user")]
        run = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._started or time.time())),
               "task": self.task, "result": last.get("action", ""), "reason": str(last.get("reason", ""))[:200],
               "seconds": round(time.time() - self._started, 1) if self._started else 0,
               "actions": len(acts), "replans": sum(s.get("action") == "replan (stuck)" for s in self.steps),
               "hiccups": sum(s.get("action") == "hiccup" for s in self.steps),
               "helps": sum(s.get("action") == "ask" for s in self.steps),
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
        failures, asleep = 0, 0
        deadline, steps_left = time.time() + self.max_seconds, self.max_steps
        try:
            await self._replan()
            while True:
                try:
                    if steps_left <= 0 or time.time() >= deadline:
                        if time.time() >= deadline:
                            await self._release()
                        raise GiveUp("step limit reached" if steps_left <= 0
                                     else f"time limit reached ({self.max_seconds:.0f} s)")
                    steps_left -= 1
                    try:
                        if await asyncio.wait_for(self._step(), max(deadline - time.time(), 0.01)):
                            return
                        failures = asleep = 0
                    except asyncio.TimeoutError:
                        deadline = 0  # handled at the top of the loop
                    except (BlindError, GiveUp):
                        raise
                    except Exception as e:  # a bad reply or a dropped link: recover, don't crash
                        self._record(action="hiccup", reason=str(e)[:300])
                        if isinstance(e, DeviceError) and USB_ASLEEP in str(e):
                            asleep += 1
                            if asleep >= USB_ASLEEP_STOP:
                                raise GiveUp(USB_ASLEEP_REASON) from e
                            await self.screen.settled()
                            continue  # the board link is up; the USB side may wake on the next try
                        asleep = 0
                        board_drop = isinstance(e, DeviceError) and await self.phone.device.wait_connected(15)
                        failures += 0 if board_drop else 1  # the board came back: not the agent's fault
                        if failures >= self.max_failures:
                            raise GiveUp(f"{failures} errors in a row") from e
                        await self.screen.settled()
                        await self._replan(stuck=True)
                except GiveUp as g:
                    if await self._give_up(g.reason, g.shot):
                        return
                    # the user stepped in: a fresh budget of time, steps and errors
                    failures, asleep = 0, 0
                    deadline, steps_left = time.time() + self.max_seconds, self.max_steps
        except asyncio.CancelledError:
            self._record(action="stopped")
        except LandscapeError as e:  # acting would be guessing, but the goal may already be reached
            answer = await self._landscape_done()
            if answer:
                self._record(self.screen.frame, action="done", answer=answer, note="judged on a landscape screen")
            else:
                self._record(action="fail", reason=str(e))
        except BlindError as e:  # no screen: acting would be guessing
            self._record(action="fail", reason=str(e))
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
                # where a run's time goes: acting (incl. locating and aiming), thinking (llm_s), settling
                self.usage["act_s"] = round(self.usage.get("act_s", 0) + act["act_s"], 2)

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
        started = time.time()
        await self.screen.settled()
        self.usage["settle_s"] = round(self.usage.get("settle_s", 0) + time.time() - started, 2)

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

        if not changed and self.steps[-1].get("action") in DRAWING:
            # a short or thin stroke can sit under the pointer, which the check ignores: not seeing it is no
            # proof it failed (iPhone 11, 2026-11-20: leaves and eye dots counted as "no change", the smiley
            # task failed and the flower one stopped to ask). Leave it to the actor's next look.
            self.steps[-1]["changed"] = None
            return False

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
            return self._quit(f"连续 {self.task_ctx.no_progress} 步屏幕无变化（{what}）")
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
            self._quit(f"卡在步骤「{current}」：重新规划后仍然 {rounds} 轮没有完成", act.pop("_shot"))
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
        prompt = f"Goal: {self.task}" + (f"\n{hints}" if (hints := app_hints(self.task)) else "") + self._hint_text()
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
        hints = app_hints(self.task)
        goal = f"Goal: {self.task}" + (f"\n{hints}" if hints else "") + self._hint_text()
        prompt = (f"{goal}\n\n{self._progress()}\n\nRecent actions:\n{history}\n\n"
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
            if blocked := self._on_at_button(acts):
                log.info("actor tapped the AssistiveTouch button at %s; asking for another spot", blocked)
                prompt += (f"\n\nYour tap at {blocked[0]} {blocked[1]} would hit the AssistiveTouch floating button "
                           f"(at about {blocked[2]} {blocked[3]}), which opens its menu instead. For a control under "
                           f"or next to it, use tapon with the control's name, or tap at least {AT_AVOID + 10} "
                           "units away from the button.")
                retry = "rejected"
                continue
            acts[0]["think_s"] = seconds
            return [{**a, "_shot": frame} for a in acts]  # every action is drawn on the same screenshot
        raise ValueError(f"actor gave no valid action: {reply[:200]}")

    async def _press_key(self, label: str, x: int, y: int, aim: bool = True) -> bool:
        """Tap one keypad key; True when the screen changed (a digit shows up on the display)."""
        before, started = self.screen.frame, time.time()
        await self.phone.tap(x / 1000, y / 1000, aim=aim)
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
        if self.screen.aspect > 1:
            raise LandscapeError("the phone screen is landscape (a full-screen video?): turn it back to portrait")
        return await self._home()

    async def _home(self) -> list[tuple[int, int]]:
        """Home Screen in one action: tap the AssistiveTouch button, then Home in its menu, each located on the
        current frame. iPhone 11, 2026-11-18: going home took the actor 2-4 rounds (button, look, Home, look,
        misses in between); cmd+h does nothing on iPhone. Offline: button found 8/8, menu Home 8/8 (the button
        moves between runs, so it is never a fixed position). No button: the old bottom-edge swipe."""
        known = self._known_at_button() if self.at_double_tap_home else None
        if known and await self._button_at(known):
            # fast path (iPhone 11: the button stayed at 750 947 for days, keyboard up or not): no pointer parking,
            # no two locate questions. A double tap that does not reach the Home Screen falls back to the full search.
            try:
                return await self._home_by_double_tap(known)
            except ValueError as e:
                log.warning("[HOME] remembered spot %s did not work (%s); looking for the button", known, e)
        button = await self._locate_at_button()
        if button is None:  # the menu may be open already (it hides the button)
            item = await self._locate(AT_HOME_Q)
            if item is not None:
                await self._tap_until_changed(item)
                log.info("[HOME] AssistiveTouch menu was open; Home at %s", item)
                return [item]
            # no bottom-edge swipe: it is unreliable here, and with the keyboard up iOS's slide-to-type read it as
            # the word "by" (iPhone 11, 2026-11-19: "by by by" in Safari's search field after failed homes)
            raise ValueError("home: no AssistiveTouch button found on screen")
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
        self._remember_at_button(button)
        return [button, item]

    async def _locate_at_button(self) -> tuple[int, int] | None:
        """The AssistiveTouch button. The mouse pointer is parked in the middle of the screen first (a move, no
        click): the button always sits at an edge, so an answer there is the pointer. Both wordings are asked in
        parallel and the first answer away from the pointer wins. iPhone 11, 2026-11-18: right after a restart the
        pointer's place was unknown and the locator's (500, 600), the pointer, was double-tapped twice."""
        await self.phone.point(*POINTER_PARK)
        self._pointer = POINTER_PARK
        await self.screen.settled(quiet=0.2, least=0.4, most=1.0)
        found = await asyncio.gather(self._locate(AT_BUTTON_Q2 + AT_KEYBOARD), self._locate(AT_BUTTON_Q + AT_KEYBOARD))
        park = (POINTER_PARK[0] * 1000, POINTER_PARK[1] * 1000)
        for spot in found:
            if spot is not None and math.dist(spot, park) > 120:
                return spot
        known = self._known_at_button()
        log.warning("[HOME] AssistiveTouch button not found (answers %s; the pointer is parked at %s); %s", found,
                    park, f"using where it was last time, {known}" if known else "no earlier position")
        return known

    async def _landscape_done(self) -> str:
        """The screen went landscape: is the goal reached anyway? The answer for the user, or "" (not reached, or
        nothing done yet, or the check failed). One question, no action."""
        if not any(s.get("act_s") is not None for s in self.steps) or not self.screen.frame:
            return ""
        try:
            reply = await self.actor.chat(LANDSCAPE_DONE_PROMPT, LANDSCAPE_DONE_Q.format(task=self.task),
                                          self.screen.frame)
        except Exception as e:  # best effort: the task fails as before
            log.warning("landscape goal check failed: %r", e)
            return ""
        lines = [x.strip() for x in reply.strip().splitlines() if x.strip()]
        self.trace.write(self.screen.frame, kind="locate", question="landscape done?", reply=reply[-200:], spot=None)
        if not lines or not lines[-1].lower().strip(" .*").startswith("yes"):
            return ""
        return (" ".join(lines[:-1]) or "目标已达成")[:300]

    async def _keyboard(self, recheck: bool = True) -> str:
        """Which on-screen keyboard shows: pinyin, english, other or none (one fast question on the frame).
        bilibili-search, 2026-11-20: the question got a 1920x1080 frame (a mirroring glitch while bilibili showed a
        landscape overlay), answered none, and the blind fallback typed Latin 'changanguniang'. So: look at a
        portrait frame only, and ask once more before believing 'none'."""
        waited = 0.0
        while self.screen.aspect > 1 and waited < PORTRAIT_WAIT:
            await asyncio.sleep(0.2)
            waited += 0.2
        frame = self.screen.frame
        fast = bool(getattr(self.actor, "fast_extra", None))
        reply = await (self.actor.chat(ONE_WORD_PROMPT, KEYBOARD_Q, frame, max_tokens=300, fast=True) if fast
                       else self.actor.chat(ONE_WORD_PROMPT, KEYBOARD_Q, frame))
        for k, v in (getattr(self.actor, "last_usage", {}) or {}).items():
            self.usage[k] = self.usage.get(k, 0) + v
        self.usage["calls"] = self.usage.get("calls", 0) + 1
        last = reply.strip().splitlines()[-1].lower() if reply.strip() else ""
        kind = ("pinyin" if any(w in last for w in ("pinyin", "空格", "选定")) else "english" if "english" in last
                else "none" if "none" in last else "other")
        self.trace.write(frame, kind="locate", question=KEYBOARD_Q, reply=reply[-60:], spot=None, keyboard=kind)
        if kind == "none" and recheck:
            await asyncio.sleep(0.5)
            return await self._keyboard(recheck=False)
        return kind

    async def _switch_keyboard(self, want: str, tries: int = 3) -> str:
        """ctrl+space until the on-screen keyboard is `want`; returns the keyboard it ended on."""
        kind = await self._keyboard()
        for _ in range(tries):
            if kind in (want, "none"):
                break
            await self.phone.key("ctrl+space")
            await asyncio.sleep(KEYBOARD_SWITCH)  # the input-source picker shows for a moment
            await self.screen.settled(quiet=0.3, least=0.3, most=1.5)
            kind = await self._keyboard()
        return kind

    async def _pinyin(self, letters: str) -> list[str]:
        """Chinese text: make sure the Simplified Pinyin keyboard is on (look, don't assume), type the letters, take
        the first candidate with space, and go back to English. iPhone 11, 2026-11-19: the blind version (ctrl+space,
        0.4 s, type) typed Latin 'ceshibiji' into Notes: the switch had not finished, or the keyboard was already
        Pinyin and ctrl+space switched it away. Done by hand with 1.5 s waits it gave 测试."""
        seen = [await self._switch_keyboard("pinyin")]
        if seen[0] == "none":  # no on-screen keyboard to look at: the old blind sequence
            await self.phone.pinyin(letters)
            return seen
        if seen[0] != "pinyin":
            raise ValueError(f"pinyin: could not switch to the Simplified Pinyin keyboard (it shows {seen[0]})")
        await self.phone.type(letters)
        await asyncio.sleep(0.8)  # let the candidate bar fill before taking the first one
        await self.phone.key("space")
        await asyncio.sleep(0.5)
        seen.append(await self._switch_keyboard("english"))
        if seen[-1] != "english":
            log.warning("[PINYIN] text entered, but the keyboard did not go back to English (%s)", seen[-1])
        return seen

    async def _button_at(self, spot: tuple[int, int]) -> bool:
        """Is the AssistiveTouch button at `spot`? One yes/no question. Offline: yes 8/8 where it was (keyboard up,
        Safari toolbar), no 3/3 on full-screen video frames (where a double tap 'likes' a bilibili video)."""
        frame = self.screen.frame
        image = vision.grid(frame) if self.grid else frame
        q = AT_BUTTON_AT_Q.format(x=spot[0], y=spot[1])
        fast = bool(getattr(self.actor, "fast_extra", None))
        system = IS_HOME_PROMPT + (GRID_NOTE if self.grid else "")
        reply = await (self.actor.chat(system, q, image, max_tokens=300, fast=True) if fast
                       else self.actor.chat(system, q, image))
        for k, v in (getattr(self.actor, "last_usage", {}) or {}).items():
            self.usage[k] = self.usage.get(k, 0) + v
        self.usage["calls"] = self.usage.get("calls", 0) + 1
        yes = (reply.strip().splitlines()[-1].strip().lower() if reply.strip() else "").startswith("yes")
        self.trace.write(frame, kind="locate", question=q, reply=reply[-40:], spot=list(spot) if yes else None)
        return yes

    async def _is_home_screen(self) -> bool:
        """One fast yes/no question about the current frame: is this the Home Screen?"""
        frame = self.screen.frame
        fast = bool(getattr(self.actor, "fast_extra", None))
        reply = await (self.actor.chat(IS_HOME_PROMPT, IS_HOME_Q, frame, max_tokens=300, fast=True) if fast
                       else self.actor.chat(IS_HOME_PROMPT, IS_HOME_Q, frame))
        for k, v in (getattr(self.actor, "last_usage", {}) or {}).items():
            self.usage[k] = self.usage.get(k, 0) + v
        self.usage["calls"] = self.usage.get("calls", 0) + 1
        last = reply.strip().splitlines()[-1].strip().lower() if reply.strip() else ""
        yes = last.startswith("yes")
        self.trace.write(frame, kind="locate", question=IS_HOME_Q, reply=reply[-80:], spot=None, home=yes)
        return yes

    def _at_file(self) -> Path | None:
        return self.runs_file.parent / "assistivetouch.json" if self.runs_file else None

    def _known_at_button(self) -> tuple[int, int] | None:
        """Where the button was when home last worked (it stays put: iPhone 11, 750 945-950 for days, keyboard or
        not). Used when locating fails, e.g. the button over the keyboard or over Safari's toolbar."""
        f = self._at_file()
        try:
            x, y = json.loads(f.read_text())["button"] if f and f.exists() else (None, None)
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return (int(x), int(y)) if x is not None else None

    def _on_at_button(self, acts: list[dict]) -> tuple[int, int, int, int] | None:
        """A coordinate tap that would land on the AssistiveTouch button (at its remembered spot): (x, y, bx, by).
        iPhone 11, 2026-11-20: "tap 690 940" for Freeform's pen tool opened the AssistiveTouch menu instead."""
        button = self._known_at_button()
        if not button:
            return None
        aspect = self.screen.aspect or 0.46
        for a in acts:
            if a.get("action") == "tap":
                # distance in width units: y counts in units of the height, which is 1/aspect times longer
                dx, dy = a["x"] - button[0], (a["y"] - button[1]) / aspect
                if math.hypot(dx, dy) < AT_AVOID:
                    return int(a["x"]), int(a["y"]), *button
        return None

    def _remember_at_button(self, spot: tuple[int, int]) -> None:
        f = self._at_file()
        if f:
            try:
                f.write_text(json.dumps({"button": list(spot), "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
            except OSError as e:
                log.warning("could not save %s: %s", f, e)

    async def _home_by_double_tap(self, button: tuple[int, int]) -> list[tuple[int, int]]:
        """The phone's AssistiveTouch Double-Tap action is Home: double-tap the button in place (one located spot,
        no menu). Taken as a single tap, it opens the menu (Single-Tap stays Open Menu): then tap Home there."""
        spot = (button[0] / 1000, button[1] / 1000)
        for attempt in range(2):
            before = self.screen.frame
            await self.phone.double_tap(*spot, aim=True)
            await self.screen.settled(quiet=0.3, least=0.8, most=2.0)  # iOS waits out the double-tap timeout
            # both questions at once: did the menu open (taken as a single tap), or is this the Home Screen
            item, home = await asyncio.gather(self._locate(AT_HOME_Q), self._is_home_screen())
            if item is not None:  # counted as a single tap: the menu opened
                log.warning("[HOME] double tap opened the menu; tapping its Home at %s", item)
                await self._tap_until_changed(item)
                self._remember_at_button(button)
                return [button, item]
            # "the screen changed" is no proof: a playing video changes all the time (iPhone 11, 2026-11-19)
            if home and vision.changed(before, self.screen.frame, ignore=(spot,)):
                log.info("[HOME] double tap on AssistiveTouch at %s", button)
                self._remember_at_button(button)
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
            case "pinyin": a["keyboards"] = await self._pinyin(a["text"])
            case "clear": await self.phone.clear_field()
            case "press":
                a["taps"] = await self._locate_keys(a["keys"].split())
                a["registered"], a["retapped"] = [], []
                for i, (label, (x, y)) in enumerate(zip(a["keys"].split(), a["taps"], strict=True)):
                    # aim at the first key only: keypad keys are big, and the aimed first tap puts the pointer
                    # where the next short moves start from (each aim costs ~1.3 s; 128 x 37 = was 42 s of pressing)
                    hit = await self._press_key(label, x, y, aim=i == 0)
                    # iPhone 11, 2026-11-18: the first key of a sequence was lost 3 times in one run (128 came
                    # out as 28; not reproduced in 10 later sequences). A digit always changes the display, so one
                    # more tap is safe; operators and C/AC are never repeated (twice can change the meaning).
                    if not hit and re.fullmatch(r"[0-9.]", label):
                        log.warning("[PRESS] %s did not register; tapping it again", label)
                        a["retapped"].append(label)
                        hit = await self._press_key(label, x, y, aim=True)
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
