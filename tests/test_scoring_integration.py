"""Integration tests for the execution-to-scoring pipeline.

Verifies the end-to-end pipeline: execution results feed into
score_all_runs() which orchestrates static, LLM, and composite scoring,
populates RunResult.scores, and produces per-variant aggregation.

Also tests that CLI flags (--skip-llm-judge, --strict-scoring) thread
through to the scoring pipeline and that progress callbacks fire correctly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_benchmark.execution.parallel import BenchmarkRun, RunResult
from claude_benchmark.scoring.errors import LLMJudgeError, ScoringError, StaticAnalysisError
from claude_benchmark.scoring.models import (
    CompositeScore,
    LLMCriterionScore,
    LLMScore,
    StaticScore,
    TokenEfficiency,
)
from claude_benchmark.scoring.pipeline import ScoringProgressCallback, score_all_runs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def task_dir(tmp_path: Path) -> Path:
    """Create a minimal task directory with task.toml and test file."""
    d = tmp_path / "task"
    d.mkdir()
    toml_content = """\
name = "test-task"
task_type = "code-gen"
difficulty = "easy"
size = "function"
description = "Test task"
prompt = "Write hello world"
tags = ["test"]

[scoring]
test_file = "test_solution.py"
"""
    (d / "task.toml").write_text(toml_content)
    (d / "test_solution.py").write_text("def test_hello(): assert True")
    return d


@pytest.fixture()
def profile_path(tmp_path: Path) -> Path:
    """Create a minimal profile .md file."""
    p = tmp_path / "profile.md"
    p.write_text("# Test Profile\nSome instructions here.")
    return p


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Create a minimal output directory with a Python file."""
    d = tmp_path / "output"
    d.mkdir()
    (d / "solution.py").write_text("def hello(): return 'world'")
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_static(weighted_total: float = 80.0) -> StaticScore:
    """Create a StaticScore with the given weighted_total."""
    return StaticScore(
        test_pass_rate=80.0,
        tests_passed=8,
        tests_total=10,
        lint_score=70.0,
        lint_errors=3,
        complexity_score=65.0,
        avg_complexity=8.0,
        weighted_total=weighted_total,
        lines_of_code=100,
    )


def _make_llm(normalized: float = 60.0, average: float = 3.4) -> LLMScore:
    """Create an LLMScore with the given normalized score."""
    return LLMScore(
        criteria=[
            LLMCriterionScore(name="code_readability", score=3, reasoning="Decent"),
            LLMCriterionScore(name="architecture_quality", score=4, reasoning="Good"),
            LLMCriterionScore(name="instruction_adherence", score=3, reasoning="OK"),
            LLMCriterionScore(name="correctness_reasoning", score=4, reasoning="Solid"),
        ],
        average=average,
        normalized=normalized,
        model_used="test-model",
    )


def _make_run_result(
    tmp_path: Path,
    task_name: str = "test-task",
    profile_name: str = "test-profile",
    model: str = "test-model",
    run_number: int = 1,
    total_tokens: int = 5000,
) -> RunResult:
    """Create a RunResult with valid task dir, profile, and output dir."""
    # Build task directory
    task_d = tmp_path / task_name
    task_d.mkdir(parents=True, exist_ok=True)
    toml = f"""\
name = "{task_name}"
task_type = "code-gen"
difficulty = "easy"
size = "function"
description = "Test task"
prompt = "Write hello world"

[scoring]
test_file = "test_solution.py"
"""
    (task_d / "task.toml").write_text(toml)
    (task_d / "test_solution.py").write_text("def test_hello(): assert True")

    # Profile
    profile_p = tmp_path / f"{profile_name}.md"
    profile_p.write_text("# Test Profile\nSome instructions here.")

    # Results dir
    results_d = tmp_path / "results"
    results_d.mkdir(parents=True, exist_ok=True)

    # Output dir with solution
    out_d = tmp_path / f"output-{task_name}-{run_number}"
    out_d.mkdir(parents=True, exist_ok=True)
    (out_d / "solution.py").write_text("def hello(): return 'world'")

    run = BenchmarkRun(
        task_name=task_name,
        profile_name=profile_name,
        model=model,
        run_number=run_number,
        task_dir=task_d,
        profile_path=profile_p,
        results_dir=results_d,
    )

    return RunResult(
        run=run,
        status="success",
        output_dir=out_d,
        total_tokens=total_tokens,
        input_tokens=total_tokens // 2,
        output_tokens=total_tokens // 2,
        cost=0.01,
        duration_seconds=5.0,
    )


# ---------------------------------------------------------------------------
# Test 1: Full pipeline execution -> scoring -> populated RunResult.scores
# ---------------------------------------------------------------------------


class TestFullPipelineExecutionToScoring:
    """Integration test verifying the full execution-to-scoring pipeline."""

    @patch("claude_benchmark.scoring.pipeline.LLMJudgeScorer")
    @patch("claude_benchmark.scoring.pipeline.StaticScorer")
    def test_full_pipeline_execution_to_scoring(
        self, mock_static_cls, mock_llm_cls, tmp_path: Path
    ) -> None:
        """Full pipeline: execution -> scoring -> populated RunResult.scores."""
        mock_static = mock_static_cls.return_value
        mock_static.score.return_value = _make_static(80.0)

        mock_llm = mock_llm_cls.return_value
        mock_llm.score.return_value = _make_llm(60.0)

        sub = tmp_path / "run1"
        sub.mkdir()
        result = _make_run_result(sub)

        scored_results, aggregation = score_all_runs(
            [result], skip_llm=False, strict=False
        )

        assert result.scores is not None

        # Static
        assert result.scores["static"] is not None
        assert result.scores["static"]["weighted_total"] == 80.0

        # LLM
        assert result.scores["llm"] is not None
        assert result.scores["llm"]["normalized"] == 60.0

        # Composite: 80*0.5 + 60*0.5 = 70.0
        assert result.scores["composite"] is not None
        assert result.scores["composite"]["composite"] == 70.0
        assert result.scores["composite"]["static_only"] is False

        # Token efficiency
        assert result.scores["token_efficiency"] is not None
        assert result.scores["token_efficiency"]["points_per_1k_tokens"] > 0

        # Degraded flag
        assert result.scores["degraded"] is False


# ---------------------------------------------------------------------------
# Test 2: Pipeline with skip_llm flag
# ---------------------------------------------------------------------------


class TestPipelineSkipLLMFlag:
    """Tests that --skip-llm-judge flag threads through to scoring."""

    @patch("claude_benchmark.scoring.pipeline.LLMJudgeScorer")
    @patch("claude_benchmark.scoring.pipeline.StaticScorer")
    def test_pipeline_skip_llm_flag(
        self, mock_static_cls, mock_llm_cls, tmp_path: Path
    ) -> None:
        """skip_llm=True skips LLM scoring entirely."""
        mock_static = mock_static_cls.return_value
        mock_static.score.return_value = _make_static(80.0)

        sub = tmp_path / "run1"
        sub.mkdir()
        result = _make_run_result(sub)

        scored_results, aggregation = score_all_runs(
            [result], skip_llm=True, strict=False
        )

        assert result.scores is not None

        # LLM should be None when skipped
        assert result.scores["llm"] is None

        # Composite should be static-only
        assert result.scores["composite"] is not None
        assert result.scores["composite"]["static_only"] is True

        # LLMJudgeScorer.score should NOT have been called
        mock_llm_cls.return_value.score.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: Strict scoring failure
# ---------------------------------------------------------------------------


class TestPipelineStrictScoringFailure:
    """Tests that strict=True propagates scorer failures as exceptions."""

    @patch("claude_benchmark.scoring.pipeline.StaticScorer")
    def test_pipeline_strict_scoring_failure(
        self, mock_static_cls, tmp_path: Path
    ) -> None:
        """strict=True raises ScoringError when static scoring fails."""
        mock_static = mock_static_cls.return_value
        mock_static.score.side_effect = StaticAnalysisError("Ruff crashed", tool="ruff")

        sub = tmp_path / "run1"
        sub.mkdir()
        result = _make_run_result(sub)

        with pytest.raises(ScoringError, match="Static scoring failed"):
            score_all_runs([result], skip_llm=True, strict=True)


# ---------------------------------------------------------------------------
# Test 4: Graceful degradation
# ---------------------------------------------------------------------------


class TestPipelineGracefulDegradation:
    """Tests graceful degradation when LLM scoring fails."""

    @patch("claude_benchmark.scoring.pipeline.time")
    @patch("claude_benchmark.scoring.pipeline.LLMJudgeScorer")
    @patch("claude_benchmark.scoring.pipeline.StaticScorer")
    def test_pipeline_graceful_degradation(
        self, mock_static_cls, mock_llm_cls, mock_time, tmp_path: Path
    ) -> None:
        """LLM failure with strict=False degrades gracefully."""
        mock_static = mock_static_cls.return_value
        mock_static.score.return_value = _make_static(80.0)

        mock_llm = mock_llm_cls.return_value
        mock_llm.score.side_effect = LLMJudgeError("API error")
        mock_time.sleep = MagicMock()

        sub = tmp_path / "run1"
        sub.mkdir()
        result = _make_run_result(sub)

        scored_results, aggregation = score_all_runs(
            [result], skip_llm=False, strict=False
        )

        assert result.scores is not None
        assert result.scores["degraded"] is True
        assert "llm_judge" in result.scores["failed_scorers"]

        # Composite should be static-only due to degradation
        assert result.scores["composite"] is not None
        assert result.scores["composite"]["static_only"] is True


# ---------------------------------------------------------------------------
# Test 5: Aggregation with multiple runs
# ---------------------------------------------------------------------------


class TestPipelineAggregationMultipleRuns:
    """Tests that multiple runs for the same variant are aggregated."""

    @patch("claude_benchmark.scoring.pipeline.StaticScorer")
    def test_pipeline_aggregation_multiple_runs(
        self, mock_static_cls, tmp_path: Path
    ) -> None:
        """3 runs of the same variant produce aggregation with n=3."""
        # Return slightly different scores for each run
        scores = [_make_static(80.0), _make_static(85.0), _make_static(90.0)]
        mock_static = mock_static_cls.return_value
        mock_static.score.side_effect = scores

        results = []
        for i in range(3):
            sub = tmp_path / f"run{i}"
            sub.mkdir()
            results.append(
                _make_run_result(
                    sub,
                    task_name="test-task",
                    profile_name="test-profile",
                    model="test-model",
                    run_number=i + 1,
                )
            )

        scored_results, aggregation = score_all_runs(
            results, skip_llm=True, strict=False
        )

        # All 3 should be scored
        assert all(r.scores is not None for r in scored_results)

        # Aggregation should have 1 variant key
        assert len(aggregation) == 1

        key = "test-task|test-profile|test-model"
        assert key in aggregation

        scores_agg = aggregation[key]["scores"]
        assert "composite" in scores_agg
        assert scores_agg["composite"]["n"] == 3

        # Mean should be close to 85.0 (average of 80, 85, 90)
        assert abs(scores_agg["composite"]["mean"] - 85.0) < 0.1


# ---------------------------------------------------------------------------
# Test 6: Aggregation with multiple variants
# ---------------------------------------------------------------------------


class TestPipelineAggregationMultipleVariants:
    """Tests aggregation with multiple distinct variants."""

    @patch("claude_benchmark.scoring.pipeline.StaticScorer")
    def test_pipeline_aggregation_multiple_variants(
        self, mock_static_cls, tmp_path: Path
    ) -> None:
        """Results for 2 different profiles produce 2 variant keys."""
        mock_static = mock_static_cls.return_value
        mock_static.score.return_value = _make_static(80.0)

        results = []
        for profile in ["profile-a", "profile-b"]:
            sub = tmp_path / f"run-{profile}"
            sub.mkdir()
            results.append(
                _make_run_result(
                    sub,
                    task_name="test-task",
                    profile_name=profile,
                    model="test-model",
                    run_number=1,
                )
            )

        scored_results, aggregation = score_all_runs(
            results, skip_llm=True, strict=False
        )

        # Should have 2 variant keys
        assert len(aggregation) == 2
        assert "test-task|profile-a|test-model" in aggregation
        assert "test-task|profile-b|test-model" in aggregation


# ---------------------------------------------------------------------------
# Test 7: Scoring progress callback
# ---------------------------------------------------------------------------


class TestScoringProgressCallbackCalled:
    """Tests that progress callbacks fire correctly during scoring."""

    @patch("claude_benchmark.scoring.pipeline.LLMJudgeScorer")
    @patch("claude_benchmark.scoring.pipeline.StaticScorer")
    def test_scoring_progress_callback_called(
        self, mock_static_cls, mock_llm_cls, tmp_path: Path
    ) -> None:
        """Progress callback receives started/progress/completed for each phase."""
        mock_static = mock_static_cls.return_value
        mock_static.score.return_value = _make_static(80.0)

        mock_llm = mock_llm_cls.return_value
        mock_llm.score.return_value = _make_llm(60.0)

        sub = tmp_path / "run1"
        sub.mkdir()
        result = _make_run_result(sub)

        mock_cb = MagicMock(
            spec=["scoring_started", "scoring_progress", "scoring_completed"]
        )

        score_all_runs(
            [result], skip_llm=False, strict=False, progress=mock_cb
        )

        # started called for static, llm, composite
        started_phases = [c.args[0] for c in mock_cb.scoring_started.call_args_list]
        assert "static" in started_phases
        assert "llm" in started_phases
        assert "composite" in started_phases

        # progress called with correct counts
        mock_cb.scoring_progress.assert_any_call(
            "static", 1, 1, result.run.result_key
        )
        mock_cb.scoring_progress.assert_any_call(
            "llm", 1, 1, result.run.result_key
        )
        mock_cb.scoring_progress.assert_any_call(
            "composite", 1, 1, result.run.result_key
        )

        # completed called for each phase
        completed_phases = [
            c.args[0] for c in mock_cb.scoring_completed.call_args_list
        ]
        assert "static" in completed_phases
        assert "llm" in completed_phases
        assert "composite" in completed_phases

    @patch("claude_benchmark.scoring.pipeline.StaticScorer")
    def test_scoring_progress_skip_llm_no_llm_phase(
        self, mock_static_cls, tmp_path: Path
    ) -> None:
        """When skip_llm=True, no LLM phase callbacks are fired."""
        mock_static = mock_static_cls.return_value
        mock_static.score.return_value = _make_static(80.0)

        sub = tmp_path / "run1"
        sub.mkdir()
        result = _make_run_result(sub)

        mock_cb = MagicMock(
            spec=["scoring_started", "scoring_progress", "scoring_completed"]
        )

        score_all_runs(
            [result], skip_llm=True, strict=False, progress=mock_cb
        )

        started_phases = [c.args[0] for c in mock_cb.scoring_started.call_args_list]
        assert "llm" not in started_phases


# ---------------------------------------------------------------------------
# Test 8: CLI flags exist
# ---------------------------------------------------------------------------


class TestCLIFlagsExist:
    """Tests that scoring CLI flags are present on the run command."""

    _ANSI_RE = __import__("re").compile(r"\x1b\[[0-9;]*m")

    def test_cli_flags_exist(self) -> None:
        """--skip-llm-judge and --strict-scoring are valid CLI options."""
        from typer.testing import CliRunner

        from claude_benchmark.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        output = self._ANSI_RE.sub("", result.output)

        assert result.exit_code == 0
        assert "--skip-llm-judge" in output
        assert "--strict-scoring" in output
