"""Monitoring and protection components for MLE-STAR framework."""

from .metrics import (
    MetricsCollector,
    TaskMetrics,
    AggregateMetrics,
    MetricType
)

from .resource_monitor import (
    ResourceMonitor,
    ResourceSnapshot,
    get_gpu_info,
    check_gpu_available
)

from .logger import (
    MLEStarLogger,
    StructuredFormatter,
    ColoredConsoleFormatter,
    TaskLogContext,
    PerformanceLogger,
    ExecutionTimer,
    get_logger,
    setup_logging
)

from .guardrails import (
    GuardrailsManager,
    InputGuardrails,
    OutputGuardrails,
    RateLimiter,
    BudgetTracker,
    ValidationResult
)


__all__ = [
    # Metrics
    'MetricsCollector',
    'TaskMetrics',
    'AggregateMetrics',
    'MetricType',

    # Resource monitoring
    'ResourceMonitor',
    'ResourceSnapshot',
    'get_gpu_info',
    'check_gpu_available',

    # Logging
    'MLEStarLogger',
    'StructuredFormatter',
    'ColoredConsoleFormatter',
    'TaskLogContext',
    'PerformanceLogger',
    'ExecutionTimer',
    'get_logger',
    'setup_logging',

    # Guardrails
    'GuardrailsManager',
    'InputGuardrails',
    'OutputGuardrails',
    'RateLimiter',
    'BudgetTracker',
    'ValidationResult',
]
