# DeepSeek iPhone Agent navigation improvements

This directory contains **reproducible changes** for the original project's
[legacy Python iPhone Agent](https://github.com/onehumanbeing/esp32-iphone-agent/tree/legacy).
It is **not** a copy of the user's full local `agent.py`: the exact final Mac-side
file could not be read or independently verified. These rules and the `aim=True`
change were reconstructed from the troubleshooting conversation and successful
user-reported tests.

## Observed problems and resolutions

| Symptom | Observed outcome / approach |
| --- | --- |
| App not on visible Home Screen | Navigate to Home, then use Spotlight |
| Home swipe failed on an iPhone 18 Pro | AssistiveTouch floating button configured to return Home on single tap |
| `cmd+space` worked on Home, failed in Settings | Do not rely on shortcut from inside an app |
| Chinese search for 备忘录 failed on English iOS | Search for `Notes`; similarly 设置 -> `Settings` |
| Tap on Spotlight app result missed | Enable `aim=True` for all taps |
| Spotlight opened, but `Settings` text did not enter | Focus visible search field before `type` |
| Multi-action runs made recovery difficult | ONE `DO` action per model response followed by screenshot verification |

The user reported success opening Notes and subsequently completing a multistep
Settings -> General -> About -> Home -> Notes navigation task. These are limited
manual observations, **not** long-term reliability benchmarks.

## Install on the Mac

First **stop the running Agent** (Ctrl-C). Back up your entire working
`server/hidagent` tree if it contains additional changes. Then:

```bash
cd ~/esp32-iphone-agent/server
python3 /path/to/esp32s3-xmc-flash-fix/iphone-agent/apply_agent_improvements.py
python3 -m py_compile hidagent/agent.py
uv run --env-file .env python -m hidagent
```

The installer makes a timestamped backup beside `agent.py`. It replaces both
`ACTOR_PROMPT` and extends `PLANNER_PROMPT`, while setting the tap call's
`aim` argument to `True`. Review its changes before using on another branch.
It deliberately does **not** modify `phone.py` or embedded firmware. If the
local Agent has changed substantially, the installer refuses to proceed.

**Practical constraints:** AssistiveTouch must be enabled and single-tap assigned
to Home; pointer calibration must already work; iPhone system language is English.
Precise pointer correction is best effort and adds latency. The prompt encourages
one action at a time, but the original Agent is not yet a formal state machine.

## Validation workflow

1. From Settings, find and tap the floating AssistiveTouch button to go Home.
2. Open Spotlight, focus the search input, enter `Notes` and verify the text.
3. Tap the actual Notes app result and verify the destination.
4. Repeat via Settings and check a multi-step read-only navigation flow.

**Never commit** `server/.env`, `secrets.h`, passwords, API keys, calibration
data containing personal details, or screen captures.

### Copyright

Original Python Agent and HID Bridge remain copyright their respective authors.
This directory contains new instructions/installer and links back to the
original source; it does not distribute upstream source files.
