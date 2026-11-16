
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskState(str, Enum):
    """智能体任务状态"""

    IDLE = "idle"
    OBSERVE = "observe"
    PLAN = "plan"
    ACT = "act"
    VERIFY = "verify"
    RECOVER = "recover"
    SUCCESS = "success"
    FAILED = "failed"


class Recovery(str, Enum):
    """一次动作之后，状态机给出的恢复决策。"""

    NONE = "none"        # 有进展，继续
    RETRY = "retry"      # 换一个动作再试一步
    REPLAN = "replan"    # 连续无进展，重新规划
    ABORT = "abort"      # 达到上限，放弃任务


@dataclass
class TaskContext:
    """保存一个任务的执行状态和历史记录"""

    task_name: str
    state: TaskState = TaskState.IDLE
    retries: int = 0
    max_retries: int = 2
    no_progress: int = 0        # 连续「屏幕无变化」的步数
    max_no_progress: int = 3    # 无变化达到这个步数就放弃
    history: list[dict] = field(default_factory=list)

    def transition(self, new_state: TaskState, reason: str = ""):
        """切换状态并保存历史"""
        old_state = self.state
        self.state = new_state

        self.history.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "from": old_state.value,
            "to": new_state.value,
            "reason": reason,
        })

        print(f"[STATE] {old_state.value} → {new_state.value}")

        if reason:
            print(f"        原因：{reason}")

    def succeed(self):
        """标记任务成功"""
        self.transition(TaskState.SUCCESS, "任务已验证完成")

    def fail(self, reason: str):
        """标记任务失败"""
        self.transition(TaskState.FAILED, reason)

    def retry(self, reason: str) -> bool:
        """登记一次重试，超过上限则失败"""
        if self.retries >= self.max_retries:
            self.fail(f"超过最大重试次数：{reason}")
            return False

        self.retries += 1
        self.transition(
            TaskState.RECOVER,
            f"第 {self.retries} 次恢复：{reason}"
        )
        return True

    def record_change(self, changed: bool, what: str = "") -> Recovery:
        """把一步的执行结果喂回状态机，并给出恢复决策。

        有变化就清零连败计数；无变化按 RETRY → REPLAN → ABORT 三级升级。
        """
        if changed:
            self.no_progress = 0
            self.retries = 0
            return Recovery.NONE

        self.no_progress += 1
        if self.no_progress >= self.max_no_progress:
            return Recovery.ABORT
        if self.no_progress >= 2:
            self.transition(TaskState.RECOVER, f"连续 {self.no_progress} 步屏幕无变化，重新规划")
            return Recovery.REPLAN
        self.transition(TaskState.RECOVER, f"{what or '动作'}未改变屏幕，换一个动作重试")
        return Recovery.RETRY

    def snapshot(self) -> dict:
        """给 Web UI / 测试用的只读快照。"""
        return {
            "task": self.task_name,
            "state": self.state.value,
            "retries": self.retries,
            "no_progress": self.no_progress,
            "history": self.history[-10:],
        }
