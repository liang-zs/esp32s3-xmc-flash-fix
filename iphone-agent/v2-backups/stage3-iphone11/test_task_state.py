
from hidagent.task_state import TaskContext, TaskState


def main():
    task = TaskContext(
        task_name="打开 Notes",
        max_retries=2
    )

    print("任务：", task.task_name)

    task.transition(TaskState.OBSERVE, "读取 iPhone 截图")
    task.transition(TaskState.PLAN, "规划 Spotlight 搜索")
    task.transition(TaskState.ACT, "点击搜索框")
    task.transition(TaskState.VERIFY, "确认搜索框焦点")

    # 模拟首次失败
    if task.retry("搜索框没有获得焦点"):
        task.transition(TaskState.ACT, "重新点击搜索框")
        task.transition(TaskState.VERIFY, "重新检查输入焦点")

    # 模拟第二次成功
    task.transition(TaskState.ACT, "输入 Notes 并打开 App")
    task.transition(TaskState.VERIFY, "确认 Notes 页面出现")
    task.succeed()

    assert task.state == TaskState.SUCCESS
    assert task.retries == 1

    print("\n最终状态：", task.state.value)
    print("重试次数：", task.retries)
    print("历史记录数：", len(task.history))


if __name__ == "__main__":
    main()

