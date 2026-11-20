"""See-think-act loop: a planner model writes the plan, an actor model picks each step."""

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

from . import vision
from .apps import app_hints
from .commands import _key, _pointer_path, circle, parse_actions, parse_plan, plan_steps
from .debug import Trace
from .device import DeviceError
from .errors import BlindError, GiveUp, LandscapeError
from .home import AT_AVOID, HomeMixin
from .keyboard import KeyboardMixin
from .llm import LLM, cost
from .locate import LocateMixin
from .phone import Phone
from .prompts import ACTOR_PROMPT, GOAL_CHECK, GRID_NOTE, LANDSCAPE_DONE_PROMPT, LANDSCAPE_DONE_Q, PLANNER_PROMPT
from .screen import Screen
from .task_state import Recovery, TaskContext, TaskState

log = logging.getLogger(__name__)

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


DRAWING = ("circle", "path")  # strokes: a missed "no change" here says little (see _step)


FAST_ROUNDS = 3  # rounds on one plan step after which the actor thinks again


class Agent(LocateMixin, HomeMixin, KeyboardMixin):
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
