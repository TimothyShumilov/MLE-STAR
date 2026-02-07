"""Unit tests for monitoring components."""

import pytest
import time
from pathlib import Path

from mle_star.monitoring.metrics import (
    MetricsCollector,
    TaskMetrics,
    AggregateMetrics,
    MetricType
)
from mle_star.monitoring.resource_monitor import ResourceMonitor, ResourceSnapshot
from mle_star.monitoring.guardrails import (
    GuardrailsManager,
    InputGuardrails,
    OutputGuardrails,
    RateLimiter,
    ValidationResult
)


@pytest.mark.unit
class TestMetricsCollector:
    """Test MetricsCollector class."""

    def test_metrics_collector_init(self, metrics_dir):
        """Test metrics collector initialization."""
        collector = MetricsCollector(metrics_dir=metrics_dir)

        assert collector.metrics_dir == metrics_dir
        assert len(collector.task_metrics) == 0
        assert len(collector.completed_tasks) == 0

    def test_start_task(self, metrics_collector):
        """Test starting task tracking."""
        metrics_collector.start_task("task_1", "classification")

        assert "task_1" in metrics_collector.task_metrics
        metrics = metrics_collector.task_metrics["task_1"]
        assert metrics.task_id == "task_1"
        assert metrics.task_type == "classification"
        assert metrics.status == "active"

    def test_record_iteration(self, metrics_collector):
        """Test recording iteration."""
        metrics_collector.start_task("task_1", "classification")
        metrics_collector.record_iteration("task_1", strategies=3, best_score=0.85)

        metrics = metrics_collector.task_metrics["task_1"]
        assert metrics.iterations_count == 1
        assert metrics.strategies_generated == 3
        assert metrics.best_score == 0.85

    def test_record_execution(self, metrics_collector):
        """Test recording execution."""
        metrics_collector.start_task("task_1", "classification")
        metrics_collector.record_execution("task_1", success=True, score=0.90)
        metrics_collector.record_execution("task_1", success=False)

        metrics = metrics_collector.task_metrics["task_1"]
        assert metrics.executions_total == 2
        assert metrics.executions_successful == 1
        assert metrics.executions_failed == 1

    def test_record_agent_call(self, metrics_collector):
        """Test recording agent calls."""
        metrics_collector.start_task("task_1", "classification")
        metrics_collector.record_agent_call("task_1", "planner")
        metrics_collector.record_agent_call("task_1", "executor")
        metrics_collector.record_agent_call("task_1", "executor")

        metrics = metrics_collector.task_metrics["task_1"]
        assert metrics.planner_calls == 1
        assert metrics.executor_calls == 2
        assert metrics.verifier_calls == 0

    def test_end_task(self, metrics_collector, metrics_dir):
        """Test ending task tracking."""
        metrics_collector.start_task("task_1", "classification")
        metrics_collector.record_iteration("task_1", strategies=3, best_score=0.95)

        time.sleep(0.1)  # Small delay

        metrics_collector.end_task("task_1", "success")

        # Should be moved to completed
        assert "task_1" not in metrics_collector.task_metrics
        assert len(metrics_collector.completed_tasks) == 1

        completed = metrics_collector.completed_tasks[0]
        assert completed.task_id == "task_1"
        assert completed.status == "success"
        assert completed.total_duration > 0

        # Should be saved to file
        task_file = metrics_dir / "task_1.json"
        assert task_file.exists()

    def test_get_aggregate_stats(self, metrics_collector):
        """Test getting aggregate statistics."""
        # Add some completed tasks
        for i in range(3):
            metrics_collector.start_task(f"task_{i}", "classification")
            metrics_collector.record_iteration(f"task_{i}", strategies=3, best_score=0.8 + i * 0.05)
            metrics_collector.record_execution(f"task_{i}", success=True)
            metrics_collector.end_task(f"task_{i}", "success")

        # Add one failed task
        metrics_collector.start_task("task_fail", "regression")
        metrics_collector.record_execution("task_fail", success=False)
        metrics_collector.end_task("task_fail", "failed")

        stats = metrics_collector.get_aggregate_stats()

        assert stats.total_tasks == 4
        assert stats.successful_tasks == 3
        assert stats.failed_tasks == 1
        assert stats.success_rate == 0.75

    def test_get_task_metrics(self, metrics_collector):
        """Test retrieving task metrics."""
        metrics_collector.start_task("task_1", "classification")

        # Get active task
        metrics = metrics_collector.get_task_metrics("task_1")
        assert metrics is not None
        assert metrics.task_id == "task_1"

        # End task
        metrics_collector.end_task("task_1", "success")

        # Get completed task
        metrics = metrics_collector.get_task_metrics("task_1")
        assert metrics is not None
        assert metrics.status == "success"


@pytest.mark.unit
class TestResourceMonitor:
    """Test ResourceMonitor class."""

    def test_resource_monitor_init(self):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor(
            gpu_memory_threshold_mb=20000,
            cpu_threshold_percent=80.0
        )

        assert monitor.gpu_memory_threshold_mb == 20000
        assert monitor.cpu_threshold_percent == 80.0

    def test_get_current_snapshot(self):
        """Test getting current resource snapshot."""
        monitor = ResourceMonitor()
        snapshot = monitor.get_current_snapshot()

        assert isinstance(snapshot, ResourceSnapshot)
        assert snapshot.timestamp > 0
        assert snapshot.cpu_percent >= 0
        assert snapshot.ram_percent >= 0

    def test_peak_tracking(self):
        """Test peak resource tracking."""
        monitor = ResourceMonitor()

        # Get a few snapshots
        for _ in range(3):
            monitor.get_current_snapshot()
            time.sleep(0.1)

        # Peak values should be set
        assert monitor.peak_cpu_percent >= 0
        assert monitor.peak_ram_percent >= 0

    def test_reset_peaks(self):
        """Test resetting peak values."""
        monitor = ResourceMonitor()
        monitor.get_current_snapshot()

        # Peaks should be set
        assert monitor.peak_cpu_percent > 0 or monitor.peak_ram_percent > 0

        monitor.reset_peaks()

        assert monitor.peak_cpu_percent == 0
        assert monitor.peak_ram_percent == 0

    def test_get_peak_usage(self):
        """Test getting peak usage."""
        monitor = ResourceMonitor()
        monitor.get_current_snapshot()

        peaks = monitor.get_peak_usage()

        assert 'peak_gpu_memory_mb' in peaks
        assert 'peak_cpu_percent' in peaks
        assert 'peak_ram_percent' in peaks


@pytest.mark.unit
class TestInputGuardrails:
    """Test InputGuardrails class."""

    def test_validate_task_description(self):
        """Test task description validation."""
        guardrails = InputGuardrails()

        # Valid description
        result = guardrails.validate_task_description(
            "Train a classifier on Iris dataset"
        )
        assert result.valid is True

        # Too short
        result = guardrails.validate_task_description("short")
        assert result.valid is False

        # Too long
        long_desc = "x" * 10000
        result = guardrails.validate_task_description(long_desc)
        assert result.valid is False

    def test_detect_dangerous_patterns(self):
        """Test detection of dangerous patterns."""
        guardrails = InputGuardrails()

        dangerous_descriptions = [
            "Delete all files with rm -rf /",
            "Run sudo commands",
            "Execute eval() on user input"
        ]

        for desc in dangerous_descriptions:
            result = guardrails.validate_task_description(desc)
            assert result.valid is False
            assert len(result.issues) > 0

    def test_detect_suspicious_keywords(self):
        """Test detection of suspicious keywords."""
        guardrails = InputGuardrails()

        result = guardrails.validate_task_description(
            "Load the password file and api_key from secrets"
        )

        # Should generate warnings
        assert len(result.warnings) > 0

    def test_validate_file_path(self):
        """Test file path validation."""
        guardrails = InputGuardrails()

        # Safe path
        result = guardrails.validate_file_path("data/train.csv")
        assert result.valid is True

        # Path traversal
        result = guardrails.validate_file_path("../../etc/passwd")
        assert result.valid is False

        # Sensitive location
        result = guardrails.validate_file_path("/etc/shadow")
        assert result.valid is False


@pytest.mark.unit
class TestOutputGuardrails:
    """Test OutputGuardrails class."""

    def test_validate_safe_code(self, sample_code):
        """Test validation of safe code."""
        guardrails = OutputGuardrails()
        result = guardrails.validate_generated_code(sample_code)

        assert result.valid is True

    def test_validate_dangerous_code(self, dangerous_code):
        """Test validation of dangerous code."""
        guardrails = OutputGuardrails()
        result = guardrails.validate_generated_code(dangerous_code)

        assert result.valid is False
        assert len(result.issues) > 0

    def test_whitelist_mode(self):
        """Test whitelist mode for imports."""
        code = """
import pandas as pd
import unknown_package

df = pd.DataFrame()
"""

        guardrails = OutputGuardrails(whitelist_mode=True)
        result = guardrails.validate_generated_code(code)

        # pandas is allowed, unknown_package should warn
        assert len(result.warnings) > 0

    def test_validate_code_length(self):
        """Test code length validation."""
        guardrails = OutputGuardrails()

        # Very long code
        long_code = "x = 1\n" * 100000

        result = guardrails.validate_generated_code(long_code)
        assert result.valid is False


@pytest.mark.unit
class TestRateLimiter:
    """Test RateLimiter class."""

    def test_rate_limiter_init(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(
            max_calls_per_day=100,
            max_tasks_per_hour=20
        )

        assert limiter.max_calls_per_day == 100
        assert limiter.max_tasks_per_hour == 20

    def test_api_call_limiting(self):
        """Test API call rate limiting."""
        limiter = RateLimiter(max_calls_per_day=5)

        # Should allow 5 calls
        for i in range(5):
            assert limiter.check_and_increment('api_call') is True

        # 6th call should be blocked
        assert limiter.check_and_increment('api_call') is False

    def test_task_start_limiting(self):
        """Test task start rate limiting."""
        limiter = RateLimiter(max_tasks_per_hour=3)

        # Should allow 3 tasks
        for i in range(3):
            assert limiter.check_and_increment('task_start') is True

        # 4th task should be blocked
        assert limiter.check_and_increment('task_start') is False

    def test_get_remaining_quota(self):
        """Test getting remaining quota."""
        limiter = RateLimiter(max_calls_per_day=10)

        # Use 3 calls
        for i in range(3):
            limiter.check_and_increment('api_call')

        remaining = limiter.get_remaining_quota('api_call')
        assert remaining == 7

    def test_reset(self):
        """Test resetting rate limiter."""
        limiter = RateLimiter(max_calls_per_day=5)

        # Use all quota
        for i in range(5):
            limiter.check_and_increment('api_call')

        assert limiter.get_remaining_quota('api_call') == 0

        # Reset
        limiter.reset()

        assert limiter.get_remaining_quota('api_call') == 5


@pytest.mark.unit
class TestGuardrailsManager:
    """Test GuardrailsManager class."""

    def test_guardrails_manager_init(self):
        """Test guardrails manager initialization."""
        manager = GuardrailsManager(
            strict_mode=True,
            max_api_calls_per_day=50
        )

        assert manager.input_guardrails is not None
        assert manager.output_guardrails is not None
        assert manager.rate_limiter is not None
        assert manager.budget_tracker is not None

    def test_validate_task_input(self):
        """Test task input validation."""
        manager = GuardrailsManager()

        result = manager.validate_task_input(
            "Train a classifier on Iris dataset",
            config={'subtasks': []}
        )

        assert result.valid is True

    def test_validate_code_output(self, sample_code):
        """Test code output validation."""
        manager = GuardrailsManager()

        result = manager.validate_code_output(sample_code)
        assert result.valid is True

    def test_check_rate_limit(self):
        """Test rate limit checking."""
        manager = GuardrailsManager(max_api_calls_per_day=5)

        # Should allow first few calls
        assert manager.check_rate_limit('api_call') is True
        assert manager.check_rate_limit('api_call') is True

    def test_get_status(self):
        """Test getting guardrails status."""
        manager = GuardrailsManager()

        # Make a few calls
        manager.check_rate_limit('api_call')
        manager.check_rate_limit('api_call')

        status = manager.get_status()

        assert 'rate_limits' in status
        assert 'budget' in status
        assert 'api_calls_remaining' in status['rate_limits']


@pytest.mark.unit
class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_creation(self):
        """Test creating validation result."""
        result = ValidationResult(valid=True)

        assert result.valid is True
        assert len(result.issues) == 0
        assert len(result.warnings) == 0

    def test_add_issue(self):
        """Test adding issue."""
        result = ValidationResult(valid=True)
        result.add_issue("Something went wrong")

        assert result.valid is False
        assert len(result.issues) == 1
        assert result.issues[0] == "Something went wrong"

    def test_add_warning(self):
        """Test adding warning."""
        result = ValidationResult(valid=True)
        result.add_warning("Be careful")

        assert result.valid is True  # Warnings don't change validity
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Be careful"
