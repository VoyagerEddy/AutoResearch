import pytest

from autoresearch.experiments import execution_succeeded


@pytest.mark.parametrize("status,execute,step_status,expected", [
    ("ready", False, "not_run", False),
    ("blocked", False, "not_run", False),
    ("succeeded", True, "succeeded", True),
    ("failed", True, "failed", False),
    ("succeeded", True, "not_run", False),
])
def test_runner_report_requires_actual_successful_steps(status, execute, step_status, expected):
    report = {"runner": "autoresearch", "status": status, "execute": execute,
              "exit_code": 0, "steps": [{"status": step_status, "exit_code": 0}]}
    assert execution_succeeded(0, report) is expected


def test_nonzero_outer_exit_never_passes():
    assert execution_succeeded(1, {}) is False
    assert execution_succeeded(None, {}) is False


def test_legacy_exit_code_behavior_is_preserved():
    assert execution_succeeded(0, {"accuracy": 0.5}) is True
