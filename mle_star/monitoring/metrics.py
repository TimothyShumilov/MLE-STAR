"""Metrics collection and tracking for MLE-STAR framework."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging
from pathlib import Path
from enum import Enum


class MetricType(Enum):
    """Types of metrics tracked."""

    TASK_COMPLETION = "task_completion"
    ITERATION_COUNT = "iteration_count"
    STRATEGY_COUNT = "strategy_count"
    EXECUTION_SUCCESS = "execution_success"
    VERIFICATION_SCORE = "verification_score"
    EXECUTION_TIME = "execution_time"
    AGENT_CALL = "agent_call"
    MODEL_INFERENCE_TIME = "model_inference_time"
    GPU_MEMORY_USAGE = "gpu_memory_usage"
    API_CALL = "api_call"


@dataclass
class TaskMetrics:
    """Metrics for a single task execution."""

    task_id: str
    task_type: str
    status: str  # 'success', 'failed', 'partial_success'

    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: float = 0.0  # seconds

    # Iterations
    iterations_count: int = 0
    strategies_generated: int = 0

    # Execution
    executions_total: int = 0
    executions_successful: int = 0
    executions_failed: int = 0
    executions_timeout: int = 0

    # Scores
    best_score: float = 0.0
    average_score: float = 0.0
    scores_per_iteration: List[float] = field(default_factory=list)

    # Agent calls
    planner_calls: int = 0
    executor_calls: int = 0
    verifier_calls: int = 0

    # API usage
    api_calls: int = 0
    api_errors: int = 0

    # Resource usage
    peak_gpu_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_ram_mb: float = 0.0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finalize(self, end_time: datetime) -> None:
        """Finalize metrics after task completion."""
        self.end_time = end_time
        self.total_duration = (end_time - self.start_time).total_seconds()

        # Calculate average score
        if self.scores_per_iteration:
            self.average_score = sum(self.scores_per_iteration) / len(self.scores_per_iteration)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'status': self.status,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration': self.total_duration,
            'iterations_count': self.iterations_count,
            'strategies_generated': self.strategies_generated,
            'executions_total': self.executions_total,
            'executions_successful': self.executions_successful,
            'executions_failed': self.executions_failed,
            'executions_timeout': self.executions_timeout,
            'best_score': self.best_score,
            'average_score': self.average_score,
            'scores_per_iteration': self.scores_per_iteration,
            'planner_calls': self.planner_calls,
            'executor_calls': self.executor_calls,
            'verifier_calls': self.verifier_calls,
            'api_calls': self.api_calls,
            'api_errors': self.api_errors,
            'peak_gpu_memory_mb': self.peak_gpu_memory_mb,
            'peak_cpu_percent': self.peak_cpu_percent,
            'peak_ram_mb': self.peak_ram_mb,
            'metadata': self.metadata
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple tasks."""

    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    partial_success_tasks: int = 0

    total_iterations: int = 0
    total_strategies: int = 0
    total_executions: int = 0

    average_iterations_per_task: float = 0.0
    average_duration_per_task: float = 0.0
    average_best_score: float = 0.0

    total_api_calls: int = 0
    total_api_errors: int = 0

    success_rate: float = 0.0
    execution_success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'partial_success_tasks': self.partial_success_tasks,
            'total_iterations': self.total_iterations,
            'total_strategies': self.total_strategies,
            'total_executions': self.total_executions,
            'average_iterations_per_task': self.average_iterations_per_task,
            'average_duration_per_task': self.average_duration_per_task,
            'average_best_score': self.average_best_score,
            'total_api_calls': self.total_api_calls,
            'total_api_errors': self.total_api_errors,
            'success_rate': self.success_rate,
            'execution_success_rate': self.execution_success_rate
        }


class MetricsCollector:
    """
    Centralized metrics collection and aggregation.

    Tracks task-level metrics, agent calls, resource usage,
    and provides aggregate statistics.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.start_task("task_123", "classification")
        >>> collector.record_iteration("task_123", strategies=3)
        >>> collector.record_execution("task_123", success=True, score=0.95)
        >>> collector.end_task("task_123", "success")
        >>> stats = collector.get_aggregate_stats()
    """

    def __init__(self, metrics_dir: Optional[Path] = None):
        """
        Initialize metrics collector.

        Args:
            metrics_dir: Directory to save metrics (default: ./metrics)
        """
        self.metrics_dir = metrics_dir or Path("./metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.completed_tasks: List[TaskMetrics] = []

        self.logger = logging.getLogger("mle_star.metrics")
        self.logger.info(f"MetricsCollector initialized: {self.metrics_dir}")

    def start_task(
        self,
        task_id: str,
        task_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Start tracking a new task.

        Args:
            task_id: Unique task identifier
            task_type: Type of task (classification, regression, etc.)
            metadata: Additional task metadata
        """
        metrics = TaskMetrics(
            task_id=task_id,
            task_type=task_type,
            status='active',
            start_time=datetime.utcnow(),
            metadata=metadata or {}
        )

        self.task_metrics[task_id] = metrics
        self.logger.info(f"Started tracking task {task_id}")

    def end_task(self, task_id: str, status: str) -> None:
        """
        End task tracking.

        Args:
            task_id: Task identifier
            status: Final status (success, failed, partial_success)
        """
        if task_id not in self.task_metrics:
            self.logger.warning(f"Task {task_id} not found in active metrics")
            return

        metrics = self.task_metrics[task_id]
        metrics.status = status
        metrics.finalize(datetime.utcnow())

        # Move to completed
        self.completed_tasks.append(metrics)
        del self.task_metrics[task_id]

        # Save to file
        self._save_task_metrics(metrics)

        self.logger.info(
            f"Task {task_id} completed: {status}, "
            f"duration={metrics.total_duration:.1f}s, "
            f"best_score={metrics.best_score:.2f}"
        )

    def record_iteration(
        self,
        task_id: str,
        strategies: int = 0,
        best_score: float = 0.0
    ) -> None:
        """
        Record an iteration.

        Args:
            task_id: Task identifier
            strategies: Number of strategies generated
            best_score: Best score in this iteration
        """
        if task_id not in self.task_metrics:
            return

        metrics = self.task_metrics[task_id]
        metrics.iterations_count += 1
        metrics.strategies_generated += strategies
        metrics.scores_per_iteration.append(best_score)

        if best_score > metrics.best_score:
            metrics.best_score = best_score

    def record_execution(
        self,
        task_id: str,
        success: bool,
        score: float = 0.0,
        timeout: bool = False
    ) -> None:
        """
        Record a code execution.

        Args:
            task_id: Task identifier
            success: Whether execution succeeded
            score: Verification score
            timeout: Whether execution timed out
        """
        if task_id not in self.task_metrics:
            return

        metrics = self.task_metrics[task_id]
        metrics.executions_total += 1

        if timeout:
            metrics.executions_timeout += 1
        elif success:
            metrics.executions_successful += 1
        else:
            metrics.executions_failed += 1

    def record_agent_call(self, task_id: str, agent_role: str) -> None:
        """
        Record an agent call.

        Args:
            task_id: Task identifier
            agent_role: Agent role (planner, executor, verifier)
        """
        if task_id not in self.task_metrics:
            return

        metrics = self.task_metrics[task_id]

        if agent_role == 'planner':
            metrics.planner_calls += 1
        elif agent_role == 'executor':
            metrics.executor_calls += 1
        elif agent_role == 'verifier':
            metrics.verifier_calls += 1

    def record_api_call(self, task_id: str, success: bool = True) -> None:
        """
        Record an API call.

        Args:
            task_id: Task identifier
            success: Whether the call succeeded
        """
        if task_id not in self.task_metrics:
            return

        metrics = self.task_metrics[task_id]
        metrics.api_calls += 1

        if not success:
            metrics.api_errors += 1

    def record_resource_usage(
        self,
        task_id: str,
        gpu_memory_mb: float = 0.0,
        cpu_percent: float = 0.0,
        ram_mb: float = 0.0
    ) -> None:
        """
        Record resource usage.

        Args:
            task_id: Task identifier
            gpu_memory_mb: GPU memory in MB
            cpu_percent: CPU usage percent
            ram_mb: RAM usage in MB
        """
        if task_id not in self.task_metrics:
            return

        metrics = self.task_metrics[task_id]

        if gpu_memory_mb > metrics.peak_gpu_memory_mb:
            metrics.peak_gpu_memory_mb = gpu_memory_mb

        if cpu_percent > metrics.peak_cpu_percent:
            metrics.peak_cpu_percent = cpu_percent

        if ram_mb > metrics.peak_ram_mb:
            metrics.peak_ram_mb = ram_mb

    def get_task_metrics(self, task_id: str) -> Optional[TaskMetrics]:
        """
        Get metrics for a specific task.

        Args:
            task_id: Task identifier

        Returns:
            TaskMetrics or None
        """
        # Check active tasks
        if task_id in self.task_metrics:
            return self.task_metrics[task_id]

        # Check completed tasks
        for metrics in self.completed_tasks:
            if metrics.task_id == task_id:
                return metrics

        return None

    def get_aggregate_stats(self) -> AggregateMetrics:
        """
        Get aggregate statistics across all completed tasks.

        Returns:
            AggregateMetrics
        """
        stats = AggregateMetrics()

        if not self.completed_tasks:
            return stats

        stats.total_tasks = len(self.completed_tasks)

        total_duration = 0.0
        total_best_scores = 0.0
        total_exec_success = 0
        total_exec_total = 0

        for metrics in self.completed_tasks:
            # Status counts
            if metrics.status == 'success':
                stats.successful_tasks += 1
            elif metrics.status == 'failed':
                stats.failed_tasks += 1
            elif metrics.status == 'partial_success':
                stats.partial_success_tasks += 1

            # Aggregates
            stats.total_iterations += metrics.iterations_count
            stats.total_strategies += metrics.strategies_generated
            stats.total_executions += metrics.executions_total
            stats.total_api_calls += metrics.api_calls
            stats.total_api_errors += metrics.api_errors

            total_duration += metrics.total_duration
            total_best_scores += metrics.best_score
            total_exec_success += metrics.executions_successful
            total_exec_total += metrics.executions_total

        # Averages
        stats.average_iterations_per_task = stats.total_iterations / stats.total_tasks
        stats.average_duration_per_task = total_duration / stats.total_tasks
        stats.average_best_score = total_best_scores / stats.total_tasks

        # Rates
        stats.success_rate = stats.successful_tasks / stats.total_tasks

        if total_exec_total > 0:
            stats.execution_success_rate = total_exec_success / total_exec_total

        return stats

    def _save_task_metrics(self, metrics: TaskMetrics) -> None:
        """Save task metrics to file."""
        try:
            file_path = self.metrics_dir / f"{metrics.task_id}.json"

            with open(file_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)

            self.logger.debug(f"Saved metrics for task {metrics.task_id}")

        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def save_aggregate_stats(self, file_path: Optional[Path] = None) -> None:
        """
        Save aggregate statistics to file.

        Args:
            file_path: Path to save (default: metrics/aggregate_stats.json)
        """
        file_path = file_path or (self.metrics_dir / "aggregate_stats.json")

        try:
            stats = self.get_aggregate_stats()

            with open(file_path, 'w') as f:
                json.dump(stats.to_dict(), f, indent=2)

            self.logger.info(f"Saved aggregate stats to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save aggregate stats: {e}")

    def load_historical_metrics(self) -> None:
        """Load historical metrics from files."""
        try:
            for file_path in self.metrics_dir.glob("*.json"):
                if file_path.name == "aggregate_stats.json":
                    continue

                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Reconstruct TaskMetrics
                metrics = TaskMetrics(
                    task_id=data['task_id'],
                    task_type=data['task_type'],
                    status=data['status'],
                    start_time=datetime.fromisoformat(data['start_time']),
                    end_time=datetime.fromisoformat(data['end_time']) if data['end_time'] else None,
                    total_duration=data['total_duration'],
                    iterations_count=data['iterations_count'],
                    strategies_generated=data['strategies_generated'],
                    executions_total=data['executions_total'],
                    executions_successful=data['executions_successful'],
                    executions_failed=data['executions_failed'],
                    executions_timeout=data['executions_timeout'],
                    best_score=data['best_score'],
                    average_score=data['average_score'],
                    scores_per_iteration=data['scores_per_iteration'],
                    planner_calls=data['planner_calls'],
                    executor_calls=data['executor_calls'],
                    verifier_calls=data['verifier_calls'],
                    api_calls=data['api_calls'],
                    api_errors=data['api_errors'],
                    peak_gpu_memory_mb=data['peak_gpu_memory_mb'],
                    peak_cpu_percent=data['peak_cpu_percent'],
                    peak_ram_mb=data['peak_ram_mb'],
                    metadata=data['metadata']
                )

                self.completed_tasks.append(metrics)

            self.logger.info(f"Loaded {len(self.completed_tasks)} historical metrics")

        except Exception as e:
            self.logger.error(f"Failed to load historical metrics: {e}")

    def get_summary(self) -> str:
        """
        Get a human-readable summary of metrics.

        Returns:
            Summary string
        """
        stats = self.get_aggregate_stats()

        summary = f"""
MLE-STAR Metrics Summary
========================

Tasks:
  Total: {stats.total_tasks}
  Successful: {stats.successful_tasks}
  Failed: {stats.failed_tasks}
  Partial Success: {stats.partial_success_tasks}
  Success Rate: {stats.success_rate:.1%}

Iterations:
  Total: {stats.total_iterations}
  Average per Task: {stats.average_iterations_per_task:.1f}

Strategies:
  Total Generated: {stats.total_strategies}

Executions:
  Total: {stats.total_executions}
  Success Rate: {stats.execution_success_rate:.1%}

Performance:
  Average Duration: {stats.average_duration_per_task:.1f}s
  Average Best Score: {stats.average_best_score:.2f}

API Usage:
  Total Calls: {stats.total_api_calls}
  Errors: {stats.total_api_errors}
"""

        return summary.strip()

    def __str__(self) -> str:
        """String representation."""
        return (
            f"MetricsCollector("
            f"active_tasks={len(self.task_metrics)}, "
            f"completed_tasks={len(self.completed_tasks)}"
            f")"
        )
