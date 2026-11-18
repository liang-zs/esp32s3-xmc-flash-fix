import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("regress", Path(__file__).parents[1] / "tools" / "regress.py")
regress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regress)


def test_judge_flags_a_too_quick_pass_as_suspect():
    calc = {"id": "calculator", "expect": ["4736", "4,736"], "min_actions": 5}
    assert regress.judge(calc, "done", "128 × 37 = 4,736", 7) == "PASS"
    # iPhone 11, 2026-11-18: Calculator still showed 4,736 and the agent read it off in 4 actions, 16 s
    assert regress.judge(calc, "done", "128 × 37 = 4736", 4) == "SUSPECT"
    assert regress.judge(calc, "done", "4,700", 9) == "FAIL"
    assert regress.judge(calc, "fail", "", 40) == "FAIL"
    assert regress.judge({"id": "x"}, "done", "", 1) == "PASS"
