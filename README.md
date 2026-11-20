# iPhone Agent V2 —— 让视觉模型操作真机 iPhone

一个本地服务：Mac 通过 **AirPlay（UxPlay）** 拿到 iPhone 实时画面，把截图交给视觉语言模型，模型输出一个动作，Mac 再通过 **ESP32-S3 USB HID 桥**（板子对 iPhone 伪装成鼠标/键盘）把动作执行掉，然后截图验证结果。

和上游 [esp32-iphone-agent](https://github.com/onehumanbeing/esp32-iphone-agent) 的 legacy Python agent 相比，这是重写后的第二代：双模型（planner + actor）、参数化动作协议、显式任务状态机、单步屏幕变化校验。

```
 iPhone ──AirPlay 镜像──▶ Mac (UxPlay) ──JPEG 帧──▶ Agent (FastAPI)
    ▲                                                     │
    │ USB HID（鼠标/键盘）                                 │ 截图 + 任务
    └──────── ESP32-S3 板 ◀──USB 串口，或 Wi-Fi TCP 7300──┘
                                                          │
                                        DeepSeek / 任意 OpenAI 兼容视觉模型
```

## 当前状态（2026-11-18）

| 能力 | 状态 | 依据 / 位置 |
| --- | --- | --- |
| AirPlay 实时取画面 | 可用 | `server/hidagent/screen.py`，UxPlay + GStreamer `fdsink`，进程退出自动重启 |
| DeepSeek 接入 | 可用 | `llm.py` 走 OpenAI 兼容 `/chat/completions`；`agent.py` 规划/执行两个角色可分别配模型 |
| USB HID 鼠标 + 键盘 | 实机可用 | 固件侧，`device.py` 的协议对端；**固件源码不在本仓库** |
| 精确点击 `aim=True` | 实机可用 | `phone.py` 闭环：从角落出发 → 找指针 → 修正，最多 3 轮再点击；iPhone 11 网格实测 15/18 点误差 ≤ 0.01（底部 Dock 行找不到指针） |
| 回主屏 `home` | 实机可用 | 程序定位 AssistiveTouch 悬浮按钮后原地双击（手机设「轻点两下 = 主屏幕」），约 6 秒；`agent.py` 的 `_home` |
| 按名称点击 `tapon` | 实机可用 | 模型只描述目标（`tapon General row`），程序单独定位再点；离线 35/37 命中，模型自报坐标 22/37 |
| 键盘 `press` | 实机可用 | `press 1 2 8 × 3 7 =`：按标签逐键定位后依次点击，没生效的数字键补点；计算器任务实机通过 |
| Spotlight 搜英文 App 名 | 实机可用 | `apps.py` 中英映射表（含本机第三方 App），按任务提示给规划/执行模型；Spotlight 打开即可输入 |
| 连续多步任务 | 实机多次通过 | `tools/regress.py` 跑 `tools/tasks.json`：settings-chain 87–142 秒、calculator 80 秒；另 6 个任务尚无结果，**非可靠性基准** |
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

前置条件：iPhone 系统语言为英文；已开启 AssistiveTouch，「自定操作」设为**单点 = 打开菜单、轻点两下 = 主屏幕、长按 = 无**（长按别设锁屏：锁屏后 USB 挂起，agent 就动不了）；「全键盘控制」保持**关闭**（开着时空格被它当作「激活」，拼音候选无法用空格确认）；已添加「简体拼音」键盘；Mac 已装 [UxPlay](https://github.com/FDH2/UxPlay)；ESP32-S3 板已烧 HID 桥固件，并用下面两种方式之一连到 Mac：

- **USB 串口（推荐，不依赖网络）**：板子的原生 USB 口接 iPhone，另一个 UART 口（USB 转串口芯片那个）接 Mac；`.env` 设 `DEVICE_SERIAL=auto`。需要 `serial-link` 分支或之后的固件。这样只剩 AirPlay 需要手机和 Mac 在同一网络（同一个 Wi-Fi 或同一个热点都行，但开了「客户端隔离」的公共网络不行）。
- **Wi-Fi**：`secrets.h` 的 `WIFI_NETWORKS` 可写多个网络（家里 Wi-Fi、手机热点……，只支持 2.4 GHz），板子加入范围内排在最前的那个。Mac 每秒在局域网广播 `hidagent 7300`（UDP 7301），板子据此找到 Mac，不用写死 IP；换网络不用重烧。广播谁都能发，所以板子连上后会发一个随机数，服务器必须用共享密钥 `LINK_KEY`（`secrets.h` 和 `.env` 里同一个值）算出 HMAC 回答，否则板子挂断并一分钟内不理这个地址。没设 `LINK_KEY` 时不做发现，只连 `SERVER_HOST`。手机、Mac、板子三者要在同一个网络里，开了「客户端隔离」的公共网络不行。

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
| `LINK_KEY` | — | 与固件 `secrets.h` 相同的随机串：开启 Wi-Fi 自动发现（UDP 7301 广播）并校验板子连的是这台 Mac |
| `DEVICE_SERIAL` | — | 板子走 USB 串口：串口路径（如 `/dev/cu.usbserial-110`），或 `auto` 自动找（CP210x / CH34x / FTDI）；串口在线时板子不再连 Wi-Fi，服务器也拒绝 Wi-Fi 接入 |
| `AIRPLAY_NAME` | `iPhone Agent` | 投屏列表里显示的名字 |
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | — | 任意 OpenAI 兼容视觉模型；`ACTOR_*` / `PLANNER_*` 可按角色覆盖 |
| `ACTOR_FAST_BODY` | — | 常规步骤不推理（DeepSeek：`{"thinking": {"type": "disabled"}}`）；删掉即一直推理 |
| `ACTOR_GRID` | — | `1` 时给执行模型的截图画 100 单位坐标网格（trace 里保存原图） |
| `AT_DOUBLE_TAP_HOME` | — | `1` = 手机上「轻点两下 = 主屏幕」已设好，`home` 用双击；不设则走菜单里的 Home |
| `LLM_PRICE` / `LLM_PRICE_IDLE` / `LLM_PEAK_HOURS` | — | 计费（见下文「每次运行有统计」） |
| `HIDAGENT_DEBUG` | `0` | `1` 时把每次运行的 prompt、回复、截图（含每次定位用的画面）落到 `~/.hidagent/traces` |
| `HIDAGENT_DATA` | `~/.hidagent` | 标定文件与 trace 的根目录 |

## 关键设计（都是踩坑换来的）

- **一轮只做一个动作**，做完必须重新截图确认。多动作连发会让失败无法归因。
- **Spotlight 打开即可输入**：录像里 38 次 `type` 全部生效，不管之前有没有点搜索框，所以不再先点一下。旧搜索词用 `clear`（全选、删除、删到行首）一步清掉。
- **`aim=True` 是闭环**：iOS 指针有加速度，移动距离与鼠标单位不成正比，且误差随连续相对移动累积。所以精确点击先回角落再移动（落点可复现、可学习），然后左右轻推指针、用推后两帧找到它，偏差 > 0.008 就修正并重新确认，最多 3 轮。
- **指针识别按形状挑**：iOS 指针静止一会儿就隐藏，所以只用刚移动过的两帧；变化区域里只认指针大小、实心的圆点，排除时钟图标秒针（细线）和 AssistiveTouch 按钮（过大）这类自己会动的东西。屏幕变化检测同理会屏蔽"模型思考期间就在动"的区域。
- **计划进度只进不退**：actor 每轮先答 `DONE: yes/no`（当前步骤在截图上完成了没有），程序推进一步；截图同时证明多步完成时答 `DONE: yes through N`（例如「设置」直接停在「关于本机」）。单独报一个更大的步骤号只算一步，防止无凭据跳步。已完成的步骤标 `[done]` 并在重新规划后保留。截图已显示本任务算出的目标结果就直接 `done`，屏幕上上次留下的结果不算。
- **回主屏 = 一个 `home` 动作**：手机上 AssistiveTouch 设为「轻点两下 = 主屏幕」。先问一句「悬浮按钮还在记住的位置吗」（`~/.hidagent/assistivetouch.json`），是就原地双击；否则把指针停在屏幕中间（按钮总在边缘，答在中间的就是指针），用两种问法并行定位按钮（含「可能叠在键盘上」）。双击后并行问「菜单弹出了吗」「这是主屏幕吗」，确认是主屏幕才算成功——视频画面一直在变，「画面变了」不算数。找不到按钮就报错，不再从底部上滑（键盘开着时会被滑行输入读成「by」）。`cmd+h`、全键盘控制对这块板子都无效（实测）。
- **能叫出名字的就别给坐标**：模型在长上下文里写坐标常偏（Dock 上方的搜索按钮一次次点在 y=790–890，实际 835），单独问一句「X 在哪」却很准。所以 `tapon <描述>`、`press <键标签>`、`home` 都由程序在当前截图（带网格）上快速定位，文字要求完全匹配（Notes 不是 rednote）。离线验证脚本：`tools/replay_actor.py`。
- **第三方 App 从 App 资源库打开**：Spotlight 搜 bilibili 时 Top Hit 是它在「设置」里的权限页（齿轮图标），其余是网页搜索；App 资源库的搜索只列 App。`apps.py` 把第三方 App 标出来，规划直接走 App 资源库（回主屏 → 左滑过最后一页 → 搜索）。
- **App 名必须用英文**：中文名在英文语系下搜不到。`apps.py` 按任务给出「中文 = 手机上的名字」（如 小红书 = rednote），长名优先（语音备忘录不是备忘录），「信息」「文件」这类常用词只在「打开」等动词后才算 App。
- **不盲操作**：AirPlay 断流时 `screen.live` 为假，agent 直接抛 `BlindError`；画面横屏（全屏视频）时抛 `LandscapeError` 结束任务——指针标定和所有坐标都按竖屏，横屏下双击曾点在 bilibili 视频上。回归运行器开跑每个任务前也检查横屏。建议手机打开竖排方向锁定。
- **屏幕变化检测**会屏蔽指针移动区域和状态栏，避免"指针动了"被误判成"界面变了"。
- **无进展即升级**：一步做完屏幕没变化时，状态机按 `RETRY → REPLAN → ABORT` 处理——先让 actor 换个动作，再没变化就带着失败上下文重新规划，连续 3 步无变化直接判失败，而不是空转到步数上限。
- **步骤卡住也会停**：屏幕一直在变、但计划步骤迟迟完成不了（进错页面、来回点同一处）时，无变化检测看不出来。所以同一步骤超过 8 轮就重新规划，第二次再卡住就判失败。整个任务另有 300 秒总时限，超时后先松开按键并回到角落再结束。
- **中文输入走拼音**：板子的键盘只能发 ASCII，直接 `type` 中文会变成乱码，所以 `type` 拒绝非 ASCII，中文用 `pinyin` 动作。它先问一句「现在是哪个键盘」（空格键显示「空格/选定」= 拼音，「space」= 英文），不是拼音才按 Ctrl+空格（每次等 1.2 秒再看，最多 3 次），输入字母、空格选第一个候选，再同样确认切回英文。盲切（等 0.4 秒、默认从英文开始）曾在备忘录里打出「ceshibiji」。前提：已添加「简体拼音」键盘，且**全键盘控制关闭**（它把空格当作「激活」，候选无法确认）。规划提示词要求中文最终必须是汉字，不许用拼音字母充数，也不许为绕开问题去改键盘等设置。
- **推理模型的回复预算**：deepseek-flash 会先「思考」，思考也算在 `max_tokens` 里。原来上限是 600，复杂截图时思考就把额度用完了，返回空内容（日志里的 `actor gave no valid action`）。现在上限是 2000，遇到「用完额度但没有答案」会自动加倍重试一次。
- **执行模型的快速模式**：设置 `ACTOR_FAST_BODY`（DeepSeek：`{"thinking": {"type": "disabled"}}`）后，常规步骤不推理直接回答（每次约 1.2 秒，推理约 3.3 秒）。以下情况仍会推理：上一步画面没变化（`missed`）、回复被拒（`rejected`）或重复了无效动作（`repeated`）、同一计划步骤超过 3 轮（`stuck`）。`runs.jsonl` 每行记录 `fast_calls` 和 `think_<原因>` 的次数与秒数，`GET /api/runs` 的 `think` 字段汇总。删掉这个变量即恢复为一直推理。离线对比：`uv run --env-file .env python tools/replay_actor.py ~/.hidagent/traces/<运行>`。iPhone 11 回归任务 8/8 成功，平均 93 秒（之前 123 秒）。
- **每次运行有统计**：结束时向 `~/.hidagent/runs.jsonl` 写一行（结果、用时、动作数、重规划次数、LLM 调用次数和 tokens、费用，以及时间去向：`llm_s` 思考、`act_s` 执行动作含定位与瞄准、`locate_s` 定位、`settle_s` 等画面稳定），`GET /api/runs` 汇总成功率、平均用时、平均 tokens。最近几次运行里执行动作占 40–60%，思考 16–32%：每次闭环瞄准约 1.3 秒，所以 `press` 只瞄准第一个键。在 `.env` 里设置 `LLM_PRICE`（高峰价）后才计算费用；再设 `LLM_PRICE_IDLE` 就按北京时间区分高峰（工作日 9–12、14–18 点，可用 `LLM_PEAK_HOURS` 改）和空闲时段，按任务开始时间计价。
- **按画面内容判断「稳定」**：AirPlay 在手机静止时也持续以约 8 fps 推流，每帧带少量编码噪声，所以「一段时间没有新帧」永远不成立，每次动作后都要等满 4 秒，每轮找指针要等满 0.8 秒。现在每收到一帧就和上一帧比较（1/4 灰度图，≥8 个像素变化超过 20 才算真变化），`settled()` 看的是内容最后一次真正变化的时间。找指针的晃动太小，不足以算变化，所以那里保留固定的 0.55 秒等待。真机：每轮找指针从 2.0 秒降到 1.27 秒，回归任务从 151 秒降到 95 秒。
- **板子连接可查**：`/api/status` 的 `board` 字段给出对方地址、固件问候、已连接时长、上次断线时间和原因，能区分「板子从没连上」和「连上后断了」；`link` 字段说明当前走 `serial` 还是 `wifi`。
- **USB 串口链路**：和 Wi-Fi 同一套行协议，115200 波特率。打开串口可能让板子重启，ROM 启动日志会先到，服务器发 `hello` 直到收到 `hello <重启原因> <运行毫秒>` 才算连上，其余行一律跳过；会话中途收到 `hello` 说明板子重启过，`links` 加一，手势会重新归位。拔掉后每 2 秒重新找一次。串口在线时串口优先：板子停止拨 Wi-Fi，服务器关掉 Wi-Fi 接入。
- **Wi-Fi 自动发现**：板子按 `WIFI_NETWORKS` 的顺序后台扫描、加入范围内的第一个网络（扫不到时轮流试，兼容隐藏网络）；Mac 在没有板子连着时每秒向每个网卡的定向广播地址和 255.255.255.255 发信标（macOS 的 255.255.255.255 只走一个网卡），每 10 秒重读一次网卡列表，切换网络后自动跟上。连上后的第一行必须是 `auth <HMAC-SHA256(LINK_KEY, 随机数)>`。

## 测试

`server/tests/` 有 117 个测试，当前全部通过（agent 回复解析、恢复与计划进度、home/tapon/press、device 协议与超时、串口链路（pty 模拟板子：启动日志、中途重启、拔线、Wi-Fi 让位）、Wi-Fi 发现与密钥校验、llm 的 token 参数与快速模式、vision 差分与指针识别、phone 闭环瞄准与双击、状态机升级、App 名映射、回归判定）。

真机回归（会占用手机几分钟到二十几分钟）：

```bash
cd server
uv run python tools/regress.py                      # tasks.json 里全部任务各一次
uv run python tools/regress.py -n 3 calculator      # 指定任务、多次
```

每个任务先用 agent 自己的 `home` 回主屏再开始；结果连同 git 提交号写入 `~/.hidagent/regress.jsonl`。答对但动作数少于 `min_actions` 的记为 SUSPECT（比如计算器屏幕上还留着上次的答案）。带 `teardown` 的任务结束后再让 agent 收尾（删掉测试备忘录、关掉视频），收尾结果单独记录、不计分。任务顺序上 bilibili 放最后（它会留下正在播放的视频）。

`server/.venv` 只装了运行时依赖；装上 dev extra 后即可运行：

```bash
cd server
uv sync --extra dev && uv run pytest -q
```

`server/test_task_state.py` 是早期的手工演示脚本，仍在 `tests/` 外面；pytest 版在 `tests/test_task_state.py`。

## 已知限制

- **实机验证极窄**：一台 iPhone、一块板、若干次手动运行；没有连续 N 次成功率，也没有跨设备数据。
- **硬件层不在本仓库**：ESP32-S3 固件与 XMC flash 补丁在另一个仓库；这里的 `device.py` 只描述协议（`move` / `tap` / `type` / `ping` / 带 `hello <reset reason> <uptime ms>` 的握手）。
- **定位用的是最新画面**：`tapon` 在模型做决定之后才定位，这几秒里界面若变了（Spotlight 结果晚到），可能找不到或找错；找不到时这一步报错、模型重看。
- **`aim` 仍是 best-effort**：指针在 Dock 等深色区域或与背景同色时找不到，会退化为不补偿的点击；每次精确点击多花约 1–3 秒。
- **每次定位多一次模型调用**（约 1 秒）：`tapon`、`press`、`home` 都依赖快速模式的定位回答，模型换了要重新验证（`tools/replay_actor.py` 与 trace 里的 locate 记录）。
- **成功判定依赖 actor 自述**：`task_state.py` 已接管"无进展"的检测与升级，但 `done` 仍由模型在回复里声明，没有基于目标的终态校验。

## 仓库与分支

- 工作分支 `iphone11-progress-aim`，推送到 `origin`（[esp32s3-xmc-flash-fix](https://github.com/liang-zs/esp32s3-xmc-flash-fix)）的同名分支；改动前的版本打了 tag `baseline-95s`。`llm-speed`、`progress-done`、`clear-field`、`aim-grid` 是已合并的实验分支。
- `origin/main` 是 XMC 补丁仓库自己的内容，和 agent 代码无关：**不要 `git push origin main`**。
- ESP32-S3 HID 桥固件在私有仓库 `liang-zs/hid-bridge-firmware`（本地 `~/embodied-ai/hid_bridge`）；`link-watchdog` 分支加了 10 秒空闲重连和 Wi-Fi 重连，尚未烧录。

## 归属与许可

本项目基于 [ESP32 iPhone Agent by Henry Yuan](https://github.com/onehumanbeing/esp32-iphone-agent)（MIT）的思路与接口重建，HID 桥固件与原始 Python agent 版权归原作者所有。本仓库的改动以 MIT 发布。

`server/.env`、`secrets.h`、Wi-Fi 密码、API key、标定数据和截图都已在 `.gitignore` 中排除，请勿提交。
