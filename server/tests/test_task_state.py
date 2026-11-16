"""无进展检测：RETRY → REPLAN → ABORT 的升级，以及给 UI 的快照。"""

from hidagent.task_state import Recovery, TaskContext, TaskState


def test_a_change_clears_the_streak():
    ctx = TaskContext("打开 Notes")
    assert ctx.record_change(False, "tap") is Recovery.RETRY
    assert ctx.record_change(False, "tap") is Recovery.REPLAN
    assert ctx.no_progress == 2

    assert ctx.record_change(True) is Recovery.NONE
    assert ctx.no_progress == 0
    assert ctx.retries == 0


def test_no_progress_escalates_retry_replan_abort():
    ctx = TaskContext("打开 Notes")
    assert ctx.record_change(False, "tap") is Recovery.RETRY
    assert ctx.state is TaskState.RECOVER
    assert ctx.record_change(False, "tap") is Recovery.REPLAN
    assert ctx.record_change(False, "tap") is Recovery.ABORT


def test_abort_alone_does_not_fail_the_task():
    ctx = TaskContext("打开 Notes")
    for _ in range(3):
        decision = ctx.record_change(False, "tap")
    assert decision is Recovery.ABORT
    assert ctx.state is TaskState.RECOVER  # 由 Agent 记录 fail，才能带上最后一步与日志


def test_snapshot_is_small_and_serialisable():
    ctx = TaskContext("打开 Notes")
    ctx.transition(TaskState.OBSERVE, "看屏幕")
    snap = ctx.snapshot()
    assert snap["task"] == "打开 Notes"
    assert snap["state"] == "observe"
    assert snap["no_progress"] == 0
    assert len(snap["history"]) == 1 and snap["history"][0]["to"] == "observe"


def test_legacy_retry_still_caps_at_max_retries():
    ctx = TaskContext("打开 Notes", max_retries=1)
    assert ctx.retry("第一次") is True
    assert ctx.retry("第二次") is False
    assert ctx.state is TaskState.FAILED
