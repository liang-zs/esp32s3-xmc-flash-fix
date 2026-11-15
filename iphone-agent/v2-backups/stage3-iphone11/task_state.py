
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


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


@dataclass
class TaskContext:
    """保存一个任务的执行状态和历史记录"""

    task_name: str
    state: TaskState = TaskState.IDLE
    retries: int = 0
    max_retries: int = 2
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

