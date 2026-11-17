# iPhone Agent V2 —— 让视觉模型操作真机 iPhone

一个本地服务：Mac 通过 **AirPlay（UxPlay）** 拿到 iPhone 实时画面，把截图交给视觉语言模型，模型输出一个动作，Mac 再通过 **ESP32-S3 USB HID 桥**（板子对 iPhone 伪装成鼠标/键盘）把动作执行掉，然后截图验证结果。

和上游 [esp32-iphone-agent](https://github.com/onehumanbeing/esp32-iphone-agent) 的 legacy Python agent 相比，这是重写后的第二代：双模型（planner + actor）、参数化动作协议、显式任务状态机、单步屏幕变化校验。

```
 iPhone ──AirPlay 镜像──▶ Mac (UxPlay) ──JPEG 帧──▶ Agent (FastAPI)
    ▲                                                     │
    │ USB HID（鼠标/键盘）                                 │ 截图 + 任务
    └──────── ESP32-S3 板 ◀──Wi-Fi TCP 7300───────────────┘
                                                          │
                                        DeepSeek / 任意 OpenAI 兼容视觉模型
```

## 当前状态（2026-11-16）

| 能力 | 状态 | 依据 / 位置 |
| --- | --- | --- |
| AirPlay 实时取画面 | 可用 | `server/hidagent/screen.py`，UxPlay + GStreamer `fdsink`，进程退出自动重启 |
| DeepSeek 接入 | 可用 | `llm.py` 走 OpenAI 兼容 `/chat/completions`；`agent.py` 规划/执行两个角色可分别配模型 |
| USB HID 鼠标 + 键盘 | 实机可用 | 固件侧，`device.py` 的协议对端；**固件源码不在本仓库** |
| 精确点击 `aim=True` | 实机可用 | `phone.py` 闭环：从角落出发 → 找指针 → 修正，最多 3 轮再点击；iPhone 11 网格实测 15/18 点误差 ≤ 0.01（底部 Dock 行找不到指针） |
| AssistiveTouch 回主屏 | 实机可用 | `agent.py:124-148` 的 prompt 规则；`phone.go_home()` |
| Spotlight 搜英文 App 名 | 实机可用 | `agent.py` 中英映射表（备忘录=Notes、设置=Settings…）+ "先聚焦搜索框再 type" |
| 连续多步任务 | 目视成功 | iPhone 11 上 Settings → General → About → Home → Notes 11 步 / 101 秒跑通（修复前 40 步）；单次运行，**非可靠性基准** |
| XMC Flash `0x207018` 启动修复 | 单板实验性通过 | 不在本仓库：见 [esp32s3-xmc-flash-fix](https://github.com/liang-zs/esp32s3-xmc-flash-fix)，ESP-IDF v5.5.5 bootloader 补丁，5/5 冷启动 + `0x900000` 回读一致 |

"实机可用"指在本人一台 iPhone 上人工确认过；没有自动化回归，也不代表换设备/换系统版本仍然成立。

## 目录结构

```
server/                 第二代服务端（本仓库主体）
  hidagent/
    app.py              FastAPI：Web UI + /api/* 控制接口 + 设备接入
    agent.py            see-think-act 循环：planner 出计划，actor 每步一个动作
    phone.py            标定、aim 补偿、tap/swipe/type/key/scroll 等高层手势
    device.py           ESP32 板的 TCP 链路：单命令串行、超时、心跳保活
    screen.py           UxPlay 接收端，维护最新 JPEG 帧
    vision.py           帧间变化检测、指针定位
    llm.py              OpenAI 兼容客户端（环境变量配 base_url/model/key）
    task_state.py       任务状态机：OBSERVE/PLAN/ACT/VERIFY/RECOVER/SUCCESS/FAILED
    static/             Web 控制台（index.html + app.js + i18n）
  tests/                pytest：agent 解析、device 协议、llm、vision
iphone-agent/           给上游 legacy 分支用的补丁安装器（旧代，与本仓库的 server/ 无关）
  apply_agent_improvements.py   AST 级替换 ACTOR_PROMPT / PLANNER_PROMPT，并把 tap 改成 aim=True
  actor_prompt.txt             补丁后的 actor prompt
```

注意 `iphone-agent/` 和 `server/` 是**两代东西**：前者改的是 `~/esp32-iphone-agent/server/hidagent/agent.py`（上游遗留代码），后者是本仓库自己的实现。不要把安装器跑在 `server/` 上。

## 快速开始

前置条件：iPhone 系统语言为英文；已开启 AssistiveTouch 且**单击悬浮球 = 回主屏**；Mac 已装 [UxPlay](https://github.com/FDH2/UxPlay)；ESP32-S3 板已烧 HID 桥固件并连上同一 Wi-Fi。

```bash
cd server
cp .env.example .env      # 填 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
uv run --env-file .env python -m hidagent
# 或已有虚拟环境：.venv/bin/python -m hidagent
```

然后打开 <http://127.0.0.1:8890>：

1. iPhone 控制中心 → 屏幕镜像 → 选 `iPhone Agent`（名字由 `AIRPLAY_NAME` 决定），画面应出现在网页里。
2. 点"校准"：服务会让指针经过若干点，你在画面上确认指针实际落点，生成 `calibration.json`。
3. 输入任务（如"打开备忘录，新建一条记录 hello"），点运行。每步的截图、模型思考、动作耗时都会出现在时间线里。

常用环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `PORT` | `8890` | Web UI 端口 |
| `DEVICE_PORT` | `7300` | ESP32 板反向连接的端口 |
| `AIRPLAY_NAME` | `iPhone Agent` | 投屏列表里显示的名字 |
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | — | 任意 OpenAI 兼容视觉模型；`ACTOR_*` / `PLANNER_*` 可按角色覆盖 |
| `HIDAGENT_DEBUG` | `0` | `1` 时把每次运行的 prompt、回复、截图落到 `~/.hidagent/traces` |
| `HIDAGENT_DATA` | `~/.hidagent` | 标定文件与 trace 的根目录 |

## 关键设计（都是踩坑换来的）

- **一轮只做一个动作**，做完必须重新截图确认。多动作连发会让失败无法归因。
- **先聚焦再输入**：Spotlight 打开 ≠ 搜索框有焦点。`tap` 后紧跟 `type` 时，服务会先等画面稳定，若点击没引起任何变化就跳过输入、重新观察（`agent.py` 的 tap→type 保护）。
- **`aim=True` 是闭环**：iOS 指针有加速度，移动距离与鼠标单位不成正比，且误差随连续相对移动累积。所以精确点击先回角落再移动（落点可复现、可学习），然后左右轻推指针、用推后两帧找到它，偏差 > 0.008 就修正并重新确认，最多 3 轮。
- **指针识别按形状挑**：iOS 指针静止一会儿就隐藏，所以只用刚移动过的两帧；变化区域里只认指针大小、实心的圆点，排除时钟图标秒针（细线）和 AssistiveTouch 按钮（过大）这类自己会动的东西。屏幕变化检测同理会屏蔽"模型思考期间就在动"的区域。
- **计划进度只进不退**：actor 每轮报告 `STEP: n`，已完成的步骤标 `[done]` 并在重新规划后保留，避免回到主屏后从第 1 步重来。
- **回主屏用 AssistiveTouch 单击**，不要用底部上滑或软件 home 动作——在测试机 iPhone 11 上反复失败。
- **App 名必须用英文**：中文名在英文语系下搜不到；prompt 里内置了常见映射。
- **不盲操作**：AirPlay 断流时 `screen.live` 为假，agent 直接抛 `BlindError` 而不是拿旧帧猜。
- **屏幕变化检测**会屏蔽指针移动区域和状态栏，避免"指针动了"被误判成"界面变了"。
- **无进展即升级**：一步做完屏幕没变化时，状态机按 `RETRY → REPLAN → ABORT` 处理——先让 actor 换个动作，再没变化就带着失败上下文重新规划，连续 3 步无变化直接判失败，而不是空转到步数上限。
- **步骤卡住也会停**：屏幕一直在变、但计划步骤迟迟完成不了（进错页面、来回点同一处）时，无变化检测看不出来。所以同一步骤超过 8 轮就重新规划，第二次再卡住就判失败。整个任务另有 300 秒总时限，超时后先松开按键并回到角落再结束。
- **中文输入走拼音**：板子的键盘只能发 ASCII 按键，直接 `type` 中文会变成乱码（「长安姑娘」→「9j;5952f53m」）。所以 `type` 拒绝非 ASCII，中文改用 `pinyin` 动作：Ctrl+空格切到 iPhone 的简体拼音，输入字母，空格选第一个候选，再切回英文。需要 iPhone 已添加简体拼音键盘，且平时停在英文输入。
- **推理模型的回复预算**：deepseek-flash 会先「思考」，思考也算在 `max_tokens` 里。原来上限是 600，复杂截图时思考就把额度用完了，返回空内容（日志里的 `actor gave no valid action`）。现在上限是 2000，遇到「用完额度但没有答案」会自动加倍重试一次。
- **每次运行有统计**：结束时向 `~/.hidagent/runs.jsonl` 写一行（结果、用时、动作数、重规划次数、LLM 调用次数和 tokens、费用），`GET /api/runs` 汇总成功率、平均用时、平均 tokens。在 `.env` 里设置 `LLM_PRICE` 后才计算费用。
- **板子连接可查**：`/api/status` 的 `board` 字段给出对方地址、固件问候、已连接时长、上次断线时间和原因，能区分「板子从没连上」和「连上后断了」。

## 测试

`server/tests/` 有 62 个测试，当前全部通过（agent 回复解析、恢复与计划进度、device 协议与超时、llm 的 token 参数回退、vision 差分与指针识别、phone 闭环瞄准、状态机升级）。

`server/.venv` 只装了运行时依赖；装上 dev extra 后即可运行：

```bash
cd server
uv sync --extra dev && uv run pytest -q
```

`server/test_task_state.py` 是早期的手工演示脚本，仍在 `tests/` 外面；pytest 版在 `tests/test_task_state.py`。

## 已知限制

- **实机验证极窄**：一台 iPhone、一块板、若干次手动运行；没有连续 N 次成功率，也没有跨设备数据。
- **硬件层不在本仓库**：ESP32-S3 固件与 XMC flash 补丁在另一个仓库；这里的 `device.py` 只描述协议（`move` / `tap` / `type` / `ping` / 带 `hello <reset reason> <uptime ms>` 的握手）。
- **AirPlay 静止不推帧**：画面不变时没有新帧，`live` 靠 90 秒宽限判断，长时间静止后的第一步可能拿到旧画面。
- **`aim` 仍是 best-effort**：指针在 Dock 等深色区域或与背景同色时找不到，会退化为不补偿的点击；每次精确点击多花约 1–3 秒。
- **模型自己的坐标估计会偏**：指针已能落到模型给的位置，但模型有时把图标位置估错（如 Settings 图标 y 给 697，实际约 593）。
- **成功判定依赖 actor 自述**：`task_state.py` 已接管"无进展"的检测与升级，但 `done` 仍由模型在回复里声明，没有基于目标的终态校验。

## 仓库与分支

- 本地 `main` 跟踪 `origin/rebuild`，已快进合并 `rebuild_ds`（`aea4545`，task_state 恢复逻辑），比远端多 1 个提交，尚未推送。
- `origin/main` 是**另一个仓库**的内容（[esp32s3-xmc-flash-fix](https://github.com/liang-zs/esp32s3-xmc-flash-fix) 的 README 与 patches），和本仓库的 agent 代码无关。

也就是说：**不要在这个工作区直接 `git push origin main`**，否则会把 agent 代码推进 XMC 补丁仓库的 main。要么统一用 `rebuild` 分支，要么给 agent 代码单开一个仓库。

## 归属与许可

本项目基于 [ESP32 iPhone Agent by Henry Yuan](https://github.com/onehumanbeing/esp32-iphone-agent)（MIT）的思路与接口重建，HID 桥固件与原始 Python agent 版权归原作者所有。本仓库的改动以 MIT 发布。

`server/.env`、`secrets.h`、Wi-Fi 密码、API key、标定数据和截图都已在 `.gitignore` 中排除，请勿提交。
