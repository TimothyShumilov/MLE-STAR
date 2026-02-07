"""Structured logging for MLE-STAR framework."""

import logging
import logging.config
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import sys


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for log records.

    Outputs log records as JSON with structured fields for easy parsing
    and analysis.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON string
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'task_id'):
            log_data['task_id'] = record.task_id

        if hasattr(record, 'agent'):
            log_data['agent'] = record.agent

        if hasattr(record, 'iteration'):
            log_data['iteration'] = record.iteration

        if hasattr(record, 'metrics'):
            log_data['metrics'] = record.metrics

        return json.dumps(log_data)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Colored console formatter for human-readable output.

    Uses ANSI color codes for different log levels.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m'    # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.

        Args:
            record: Log record to format

        Returns:
            Formatted string with ANSI colors
        """
        # Get color for level
        color = self.COLORS.get(record.levelname, '')

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')

        # Format message
        message = super().format(record)

        # Add color
        return f"{color}[{timestamp}] {record.levelname:8s} {record.name:25s}{self.RESET} {message}"


class MLEStarLogger:
    """
    Centralized logging configuration for MLE-STAR framework.

    Provides structured logging with both JSON file output and
    human-readable console output.

    Example:
        >>> logger_manager = MLEStarLogger()
        >>> logger_manager.setup_logging(log_dir="./logs", level="INFO")
        >>> logger = logging.getLogger("mle_star.workflow")
        >>> logger.info("Task started", extra={'task_id': '123'})
    """

    DEFAULT_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': 'mle_star.monitoring.logger.StructuredFormatter'
            },
            'console': {
                '()': 'mle_star.monitoring.logger.ColoredConsoleFormatter',
                'format': '%(message)s'
            },
            'file': {
                'format': '[%(asctime)s] %(levelname)-8s %(name)-25s %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'console',
                'stream': 'ext://sys.stdout'
            },
            'file_json': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'json',
                'filename': 'logs/mle_star.jsonl',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            },
            'file_text': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'file',
                'filename': 'logs/mle_star.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'file',
                'filename': 'logs/errors.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            'mle_star': {
                'level': 'DEBUG',
                'handlers': ['console', 'file_json', 'file_text', 'error_file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        }
    }

    def __init__(self):
        """Initialize logger manager."""
        self.config = self.DEFAULT_CONFIG.copy()

    def setup_logging(
        self,
        log_dir: Optional[Path] = None,
        level: str = "INFO",
        config_file: Optional[Path] = None
    ) -> None:
        """
        Setup logging configuration.

        Args:
            log_dir: Directory for log files (default: ./logs)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            config_file: Optional YAML config file to override defaults
        """
        log_dir = log_dir or Path("./logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Load custom config if provided
        if config_file and config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                custom_config = yaml.safe_load(f)
                self.config.update(custom_config)

        # Update log file paths
        for handler_name, handler_config in self.config['handlers'].items():
            if 'filename' in handler_config:
                filename = Path(handler_config['filename'])
                handler_config['filename'] = str(log_dir / filename.name)

        # Update log level
        self.config['loggers']['mle_star']['level'] = level.upper()
        self.config['handlers']['console']['level'] = level.upper()

        # Apply configuration
        logging.config.dictConfig(self.config)

        # Log startup message
        logger = logging.getLogger("mle_star")
        logger.info(f"Logging initialized: level={level}, log_dir={log_dir}")

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get logger for a specific component.

        Args:
            name: Logger name (e.g., "mle_star.workflow")

        Returns:
            Logger instance
        """
        return logging.getLogger(name)

    def add_file_handler(
        self,
        name: str,
        filename: str,
        level: str = "INFO",
        formatter: str = "file"
    ) -> None:
        """
        Add a custom file handler.

        Args:
            name: Handler name
            filename: Log file name
            level: Log level
            formatter: Formatter to use
        """
        handler = logging.handlers.RotatingFileHandler(
            filename=filename,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )

        handler.setLevel(getattr(logging, level.upper()))

        # Get formatter
        if formatter in self.config['formatters']:
            formatter_config = self.config['formatters'][formatter]
            if '()' in formatter_config:
                # Custom formatter class
                formatter_class = formatter_config['()']
                # This is simplified - in production use proper import
                handler.setFormatter(logging.Formatter())
            else:
                handler.setFormatter(
                    logging.Formatter(
                        fmt=formatter_config.get('format'),
                        datefmt=formatter_config.get('datefmt')
                    )
                )

        # Add to root logger
        logging.getLogger('mle_star').addHandler(handler)


# Convenience functions

def get_logger(name: str) -> logging.Logger:
    """
    Get logger for a component.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    config_file: Optional[Path] = None
) -> None:
    """
    Setup logging (convenience function).

    Args:
        log_dir: Directory for log files
        level: Logging level
        config_file: Optional config file
    """
    manager = MLEStarLogger()
    manager.setup_logging(log_dir=log_dir, level=level, config_file=config_file)


# Context manager for task logging

class TaskLogContext:
    """
    Context manager for task-specific logging.

    Automatically adds task_id to all log records within the context.

    Example:
        >>> with TaskLogContext(task_id="task_123"):
        ...     logger.info("Processing task")  # Includes task_id automatically
    """

    def __init__(
        self,
        task_id: str,
        logger_name: str = "mle_star",
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize task log context.

        Args:
            task_id: Task identifier
            logger_name: Logger name
            extra: Additional context fields
        """
        self.task_id = task_id
        self.logger = logging.getLogger(logger_name)
        self.extra = extra or {}
        self.extra['task_id'] = task_id
        self.old_factory = None

    def __enter__(self):
        """Enter context."""
        # Save old factory
        self.old_factory = logging.getLogRecordFactory()

        # Create new factory that adds task_id
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.extra.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        # Restore old factory
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)


# Performance logging utilities

class PerformanceLogger:
    """
    Logger for performance metrics.

    Provides convenient methods for logging execution times and performance data.

    Example:
        >>> perf_logger = PerformanceLogger()
        >>> with perf_logger.log_execution("model_inference"):
        ...     # Your code here
        ...     pass
    """

    def __init__(self, logger_name: str = "mle_star.performance"):
        """
        Initialize performance logger.

        Args:
            logger_name: Logger name
        """
        self.logger = logging.getLogger(logger_name)

    def log_execution(self, operation: str):
        """
        Context manager for logging execution time.

        Args:
            operation: Operation name

        Example:
            >>> with perf_logger.log_execution("training"):
            ...     model.fit(X, y)
        """
        return ExecutionTimer(operation, self.logger)

    def log_metric(self, name: str, value: float, unit: str = "") -> None:
        """
        Log a performance metric.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
        """
        self.logger.info(
            f"Metric: {name} = {value:.4f} {unit}",
            extra={'metric_name': name, 'metric_value': value, 'metric_unit': unit}
        )


class ExecutionTimer:
    """
    Context manager for timing code execution.

    Example:
        >>> with ExecutionTimer("training", logger):
        ...     model.fit(X, y)
    """

    def __init__(self, operation: str, logger: logging.Logger):
        """
        Initialize execution timer.

        Args:
            operation: Operation name
            logger: Logger to use
        """
        self.operation = operation
        self.logger = logger
        self.start_time = None

    def __enter__(self):
        """Start timer."""
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log."""
        import time
        elapsed = time.time() - self.start_time
        self.logger.info(
            f"{self.operation} completed in {elapsed:.2f}s",
            extra={'operation': self.operation, 'elapsed_time': elapsed}
        )


# Initialize default logging on import
def _init_default_logging():
    """Initialize basic logging if not configured."""
    if not logging.getLogger('mle_star').handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)-8s %(name)-25s %(message)s',
            datefmt='%H:%M:%S'
        )


_init_default_logging()
