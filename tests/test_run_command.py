"""Tests for the ``claude-benchmark run`` CLI command.

Uses extensive mocking to isolate CLI flag parsing and integration wiring
from actual task/profile loading and benchmark execution.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from claude_benchmark.cli.main import app
from claude_benchmark.cli.commands.run import _TaskProxy, _ProfileProxy, _write_manifest

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

runner = CliRunner()


def _plain(text: str) -> str:
    """Strip ANSI escape codes from Rich/Typer output."""
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _fake_task_proxies() -> list[_TaskProxy]:
    return [
        _TaskProxy(name="task-a", path=Path("tasks/builtin/task-a")),
        _TaskProxy(name="task-b", path=Path("tasks/builtin/task-b")),
    ]


def _fake_profile_proxies() -> list[_ProfileProxy]:
    return [
        _ProfileProxy(name="empty", path=Path("profiles/empty.md")),
        _ProfileProxy(name="typical", path=Path("profiles/typical.md")),
    ]


# Shortcut: mock both loaders to return proxies directly
LOAD_TASKS = "claude_benchmark.cli.commands.run._load_tasks"
LOAD_PROFILES = "claude_benchmark.cli.commands.run._load_profiles"
BUILD_MATRIX = "claude_benchmark.cli.commands.run.build_run_matrix"
FILTER_RUNS = "claude_benchmark.cli.commands.run.filter_runs"
SHOW_DRY_RUN = "claude_benchmark.cli.commands.run.show_dry_run"
CONFIRM = "claude_benchmark.cli.commands.run.confirm_or_abort"
RUN_PARALLEL = "claude_benchmark.cli.commands.run.run_benchmark_parallel"
COST_TRACKER = "claude_benchmark.cli.commands.run.CostTracker"
DETECT_COMPLETED = "claude_benchmark.cli.commands.run.detect_completed_runs"
FILTER_REMAINING = "claude_benchmark.cli.commands.run.filter_remaining_runs"


# ---------------------------------------------------------------------------
# Tests: --help shows all flags
# ---------------------------------------------------------------------------


class TestRunHelp:
    """Verify all expected flags appear in ``run --help`` output."""

    def test_help_shows_concurrency_flag(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--concurrency" in _plain(result.output)

    def test_help_shows_max_cost_flag(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--max-cost" in _plain(result.output)

    def test_help_shows_task_flag(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--task" in _plain(result.output)

    def test_help_shows_profile_flag(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--profile" in _plain(result.output)

    def test_help_shows_model_flag(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--model" in _plain(result.output)

    def test_help_shows_reps_flag(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--reps" in _plain(result.output)

    def test_help_shows_results_dir_flag(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--results-dir" in _plain(result.output)

    def test_help_shows_yes_flag(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--yes" in _plain(result.output)

    def test_help_shows_dry_run_flag(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--dry-run" in _plain(result.output)


# ---------------------------------------------------------------------------
# Tests: --dry-run shows preview and exits without executing
# ---------------------------------------------------------------------------


class TestDryRun:
    """Verify --dry-run shows the plan preview and does not execute."""

    @patch(RUN_PARALLEL)
    @patch(SHOW_DRY_RUN)
    @patch(FILTER_RUNS, side_effect=lambda runs, **kw: runs)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_dry_run_shows_preview_no_execution(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_show_dry_run,
        mock_run_parallel,
    ):
        fake_run = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        mock_build_matrix.return_value = [fake_run]

        result = runner.invoke(app, ["run", "--dry-run"])

        # Preview shown
        mock_show_dry_run.assert_called_once()
        # Execution NOT called
        mock_run_parallel.assert_not_called()
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Tests: --concurrency is parsed and passed
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Verify --concurrency flag is parsed and forwarded to the orchestrator."""

    @patch(SHOW_DRY_RUN)
    @patch(FILTER_RUNS, side_effect=lambda runs, **kw: runs)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_concurrency_appears_in_preview(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_show_dry_run,
    ):
        fake_run = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        mock_build_matrix.return_value = [fake_run]

        runner.invoke(app, ["run", "--dry-run", "--concurrency", "8"])

        # show_dry_run called with concurrency=8 as 3rd positional arg
        args, kwargs = mock_show_dry_run.call_args
        assert args[2] == 8


# ---------------------------------------------------------------------------
# Tests: --max-cost creates CostTracker with correct limit
# ---------------------------------------------------------------------------


class TestMaxCost:
    """Verify --max-cost creates a CostTracker with the right cap."""

    @patch(COST_TRACKER)
    @patch(SHOW_DRY_RUN)
    @patch(FILTER_RUNS, side_effect=lambda runs, **kw: runs)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_max_cost_passed_to_cost_tracker(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_show_dry_run,
        mock_cost_tracker_cls,
    ):
        fake_run = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        mock_build_matrix.return_value = [fake_run]

        runner.invoke(app, ["run", "--dry-run", "--max-cost", "5.50"])

        mock_cost_tracker_cls.assert_called_once_with(max_cost=5.50)


# ---------------------------------------------------------------------------
# Tests: --yes skips confirmation prompt
# ---------------------------------------------------------------------------


class TestYesFlag:
    """Verify --yes flag skips the confirmation prompt."""

    @patch(CONFIRM)
    @patch(SHOW_DRY_RUN)
    @patch(FILTER_RUNS, side_effect=lambda runs, **kw: runs)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_yes_skips_confirm(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_show_dry_run,
        mock_confirm,
    ):
        fake_run = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        mock_build_matrix.return_value = [fake_run]

        fake_result = MagicMock(status="success", cost=0.01)

        with patch("claude_benchmark.cli.commands.run.asyncio") as mock_asyncio, \
             patch("claude_benchmark.cli.commands.run.Console") as mock_console_cls, \
             patch("claude_benchmark.execution.dashboard.Dashboard") as mock_dashboard, \
             patch("claude_benchmark.execution.logger.LogLineOutput") as mock_log:
            mock_console_cls.return_value.is_terminal = False
            mock_asyncio.run.return_value = [fake_result]
            runner.invoke(app, ["run", "--yes"])

        # confirm_or_abort should NOT have been called
        mock_confirm.assert_not_called()

    @patch(CONFIRM)
    @patch(SHOW_DRY_RUN)
    @patch(FILTER_RUNS, side_effect=lambda runs, **kw: runs)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_no_yes_calls_confirm(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_show_dry_run,
        mock_confirm,
    ):
        fake_run = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        mock_build_matrix.return_value = [fake_run]

        fake_result = MagicMock(status="success", cost=0.01)

        with patch("claude_benchmark.cli.commands.run.asyncio") as mock_asyncio, \
             patch("claude_benchmark.cli.commands.run.Console") as mock_console_cls, \
             patch("claude_benchmark.execution.dashboard.Dashboard"), \
             patch("claude_benchmark.execution.logger.LogLineOutput"):
            mock_console_cls.return_value.is_terminal = False
            mock_asyncio.run.return_value = [fake_result]
            runner.invoke(app, ["run"])

        # confirm_or_abort SHOULD have been called (no --yes)
        mock_confirm.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: --task and --model filters narrow the run matrix
# ---------------------------------------------------------------------------


class TestFilters:
    """Verify filter flags narrow the run matrix."""

    @patch(SHOW_DRY_RUN)
    @patch(FILTER_RUNS)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_task_filter_passed_to_filter_runs(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_show_dry_run,
    ):
        fake_run = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        mock_build_matrix.return_value = [fake_run]
        mock_filter_runs.return_value = [fake_run]

        runner.invoke(app, ["run", "--dry-run", "--task", "task-a"])

        mock_filter_runs.assert_called_once()
        _, kwargs = mock_filter_runs.call_args
        assert kwargs["task_names"] == ["task-a"]

    @patch(SHOW_DRY_RUN)
    @patch(FILTER_RUNS)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_model_filter_passed_to_filter_runs(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_show_dry_run,
    ):
        fake_run = MagicMock(task_name="task-a", profile_name="empty", model="haiku")
        mock_build_matrix.return_value = [fake_run]
        mock_filter_runs.return_value = [fake_run]

        runner.invoke(app, ["run", "--dry-run", "--model", "haiku"])

        mock_filter_runs.assert_called_once()
        _, kwargs = mock_filter_runs.call_args
        assert kwargs["model_names"] == ["haiku"]

    @patch(SHOW_DRY_RUN)
    @patch(FILTER_RUNS)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_profile_filter_passed_to_filter_runs(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_show_dry_run,
    ):
        fake_run = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        mock_build_matrix.return_value = [fake_run]
        mock_filter_runs.return_value = [fake_run]

        runner.invoke(app, ["run", "--dry-run", "--profile", "empty"])

        mock_filter_runs.assert_called_once()
        _, kwargs = mock_filter_runs.call_args
        assert kwargs["profile_names"] == ["empty"]


# ---------------------------------------------------------------------------
# Tests: Resume detection
# ---------------------------------------------------------------------------


class TestResume:
    """Verify resume detection skips completed runs."""

    @patch(SHOW_DRY_RUN)
    @patch(FILTER_REMAINING)
    @patch(DETECT_COMPLETED)
    @patch(FILTER_RUNS, side_effect=lambda runs, **kw: runs)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_resume_detects_completed_runs(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_detect,
        mock_filter_remaining,
        mock_show_dry_run,
        tmp_path,
    ):
        fake_run1 = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        fake_run2 = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        mock_build_matrix.return_value = [fake_run1, fake_run2]

        # Simulate one completed run
        mock_detect.return_value = {"sonnet/empty/task-a/run-1"}
        mock_filter_remaining.return_value = [fake_run2]

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        result = runner.invoke(
            app, ["run", "--dry-run", "--results-dir", str(results_dir)]
        )

        # detect_completed_runs should have been called
        mock_detect.assert_called_once()

    @patch(SHOW_DRY_RUN)
    @patch(FILTER_REMAINING)
    @patch(DETECT_COMPLETED)
    @patch(FILTER_RUNS, side_effect=lambda runs, **kw: runs)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_resume_all_complete_exits_cleanly(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_detect,
        mock_filter_remaining,
        mock_show_dry_run,
        tmp_path,
    ):
        fake_run = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        mock_build_matrix.return_value = [fake_run]

        # All runs completed
        mock_detect.return_value = {"sonnet/empty/task-a/run-1"}
        mock_filter_remaining.return_value = []  # no remaining

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        result = runner.invoke(
            app, ["run", "--results-dir", str(results_dir)]
        )

        assert "already completed" in result.output.lower()
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Tests: build_run_matrix called with correct arguments
# ---------------------------------------------------------------------------


class TestMatrixBuilding:
    """Verify build_run_matrix receives correct arguments from CLI flags."""

    @patch(SHOW_DRY_RUN)
    @patch(FILTER_RUNS, side_effect=lambda runs, **kw: runs)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_reps_flag_passed_to_build_matrix(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_show_dry_run,
    ):
        fake_run = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        mock_build_matrix.return_value = [fake_run]

        runner.invoke(app, ["run", "--dry-run", "--reps", "7"])

        mock_build_matrix.assert_called_once()
        _, kwargs = mock_build_matrix.call_args
        assert kwargs["reps"] == 7

    @patch(SHOW_DRY_RUN)
    @patch(FILTER_RUNS, side_effect=lambda runs, **kw: runs)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_default_model_is_sonnet(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_show_dry_run,
    ):
        fake_run = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        mock_build_matrix.return_value = [fake_run]

        runner.invoke(app, ["run", "--dry-run"])

        mock_build_matrix.assert_called_once()
        _, kwargs = mock_build_matrix.call_args
        assert kwargs["models"] == ["sonnet"]

    @patch(SHOW_DRY_RUN)
    @patch(FILTER_RUNS, side_effect=lambda runs, **kw: runs)
    @patch(BUILD_MATRIX)
    @patch(LOAD_PROFILES, return_value=_fake_profile_proxies())
    @patch(LOAD_TASKS, return_value=_fake_task_proxies())
    def test_multiple_models_passed(
        self,
        mock_load_tasks,
        mock_load_profiles,
        mock_build_matrix,
        mock_filter_runs,
        mock_show_dry_run,
    ):
        fake_run = MagicMock(task_name="task-a", profile_name="empty", model="sonnet")
        mock_build_matrix.return_value = [fake_run]

        runner.invoke(app, ["run", "--dry-run", "--model", "haiku", "--model", "opus"])

        mock_build_matrix.assert_called_once()
        _, kwargs = mock_build_matrix.call_args
        assert kwargs["models"] == ["haiku", "opus"]


# ---------------------------------------------------------------------------
# Tests: _write_manifest creates manifest.json
# ---------------------------------------------------------------------------


class TestWriteManifest:
    """Verify _write_manifest creates a valid manifest.json for report discovery."""

    def test_creates_manifest_json(self, tmp_path):
        results_dir = tmp_path / "results" / "20260227-120000"
        results_dir.mkdir(parents=True)

        _write_manifest(
            results_dir=results_dir,
            models=["sonnet"],
            profiles=["empty", "typical"],
            tasks=["task-a", "task-b"],
            reps=3,
            total_runs=12,
        )

        manifest_path = results_dir / "manifest.json"
        assert manifest_path.exists()

        data = json.loads(manifest_path.read_text())
        assert data["models"] == ["sonnet"]
        assert data["profiles"] == ["empty", "typical"]
        assert data["tasks"] == ["task-a", "task-b"]
        assert data["runs_per_combination"] == 3
        assert data["total_combinations"] == 4  # 2 tasks * 2 profiles * 1 model
        assert data["total_runs"] == 12
        assert "timestamp" in data

    def test_manifest_enables_find_latest_results(self, tmp_path):
        """Verify find_latest_results discovers directories with manifest.json."""
        from claude_benchmark.reporting.loader import find_latest_results

        results_dir = tmp_path / "20260227-120000"
        results_dir.mkdir()

        # Before manifest: not found
        assert find_latest_results(base_dir=tmp_path) is None

        _write_manifest(
            results_dir=results_dir,
            models=["sonnet"],
            profiles=["empty"],
            tasks=["task-a"],
            reps=1,
            total_runs=1,
        )

        # After manifest: found
        found = find_latest_results(base_dir=tmp_path)
        assert found is not None
        assert found.name == "20260227-120000"

    def test_manifest_overwrites_on_resume(self, tmp_path):
        """Verify manifest is updated (overwritten) on subsequent writes."""
        results_dir = tmp_path / "20260227-120000"
        results_dir.mkdir()

        _write_manifest(
            results_dir=results_dir,
            models=["sonnet"],
            profiles=["empty"],
            tasks=["task-a"],
            reps=1,
            total_runs=5,
        )

        _write_manifest(
            results_dir=results_dir,
            models=["sonnet"],
            profiles=["empty"],
            tasks=["task-a"],
            reps=1,
            total_runs=10,
        )

        data = json.loads((results_dir / "manifest.json").read_text())
        assert data["total_runs"] == 10
