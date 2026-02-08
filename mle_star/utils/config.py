"""Configuration management for MLE-STAR framework."""

from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import yaml
import logging


@dataclass
class Config:
    """
    Main configuration for the MLE-STAR framework.

    Configuration must be created directly by passing all parameters.

    This class consolidates all configuration settings including:
    - API keys and credentials
    - Model configurations for each agent
    - Workflow parameters
    - Resource limits
    - Monitoring settings
    - Security settings

    Example:
        >>> config = Config(
        ...     openrouter_api_key='sk-or-...',
        ...     kaggle_username='your_username',
        ...     kaggle_key='your_api_key',
        ...     planner_model={
        ...         'model_name': 'meta-llama/llama-3.3-70b-instruct:free',
        ...         'temperature': 0.8,
        ...         'max_tokens': 2000,
        ...     },
        ...     executor_model={
        ...         'model_name': 'Qwen/Qwen2.5-Coder-32B-Instruct',
        ...         'temperature': 0.2,
        ...         'max_tokens': 4000,
        ...         'load_in_4bit': True,
        ...         'estimated_memory_gb': 10.0,
        ...     },
        ...     verifier_model={
        ...         'model_name': 'Qwen/Qwen2.5-Coder-14B-Instruct',
        ...         'temperature': 0.1,
        ...         'max_tokens': 1500,
        ...         'load_in_4bit': True,
        ...         'estimated_memory_gb': 4.0,
        ...     },
        ...     max_iterations=5,
        ...     parallel_strategies=3,
        ... )
    """

    # API Keys
    openrouter_api_key: str
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None

    # Model configurations
    planner_model: Dict[str, Any] = field(default_factory=dict)
    executor_model: Dict[str, Any] = field(default_factory=dict)
    verifier_model: Dict[str, Any] = field(default_factory=dict)

    # Workflow settings
    max_iterations: int = 5
    parallel_strategies: int = 3

    # Resource limits
    max_gpu_memory_gb: float = 28.0
    max_execution_time: int = 300
    max_memory_mb: int = 4096

    # Monitoring
    enable_monitoring: bool = True
    metrics_dir: Path = field(default_factory=lambda: Path("./metrics"))
    logs_dir: Path = field(default_factory=lambda: Path("./logs"))

    # Security
    enable_sandbox: bool = True
    allow_network: bool = False
    allow_file_io: bool = False

    # Development
    debug: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        """Post-initialization: convert string paths to Path objects."""
        if isinstance(self.metrics_dir, str):
            self.metrics_dir = Path(self.metrics_dir)
        if isinstance(self.logs_dir, str):
            self.logs_dir = Path(self.logs_dir)

        # Create directories if they don't exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            'openrouter_api_key': '***' if self.openrouter_api_key else None,  # Redacted
            'kaggle_username': '***' if self.kaggle_username else None,  # Redacted
            'kaggle_key': '***' if self.kaggle_key else None,  # Redacted
            'planner_model': self.planner_model,
            'executor_model': self.executor_model,
            'verifier_model': self.verifier_model,
            'max_iterations': self.max_iterations,
            'parallel_strategies': self.parallel_strategies,
            'max_gpu_memory_gb': self.max_gpu_memory_gb,
            'max_execution_time': self.max_execution_time,
            'max_memory_mb': self.max_memory_mb,
            'enable_monitoring': self.enable_monitoring,
            'metrics_dir': str(self.metrics_dir),
            'logs_dir': str(self.logs_dir),
            'enable_sandbox': self.enable_sandbox,
            'allow_network': self.allow_network,
            'allow_file_io': self.allow_file_io,
            'debug': self.debug,
            'log_level': self.log_level,
        }

    def save(self, path: Path) -> None:
        """
        Save configuration to YAML file.

        Note: API keys are redacted in the saved file for security.

        Args:
            path: Path to save configuration

        Example:
            >>> config = Config(...)
            >>> config.save(Path("config.yaml"))
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate configuration values.

        Returns:
            Tuple of (is_valid, error_message)

        Example:
            >>> config = Config(...)
            >>> is_valid, error = config.validate()
            >>> if not is_valid:
            ...     print(f"Invalid config: {error}")
        """
        # Validate API key
        if not self.openrouter_api_key:
            return False, "OpenRouter API key is required"

        # Validate model configs
        required_model_fields = ['model_name', 'temperature', 'max_tokens']

        for model_name, model_config in [
            ('planner', self.planner_model),
            ('executor', self.executor_model),
            ('verifier', self.verifier_model)
        ]:
            for field in required_model_fields:
                if field not in model_config:
                    return False, f"Missing '{field}' in {model_name}_model configuration"

        # Validate ranges
        if not (1 <= self.max_iterations <= 100):
            return False, "max_iterations must be between 1 and 100"

        if not (1 <= self.parallel_strategies <= 10):
            return False, "parallel_strategies must be between 1 and 10"

        if not (0.0 <= self.max_gpu_memory_gb <= 1000.0):
            return False, "max_gpu_memory_gb must be between 0 and 1000"

        if not (0 < self.max_execution_time <= 3600):
            return False, "max_execution_time must be between 1 and 3600 seconds"

        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_levels:
            return False, f"log_level must be one of {valid_levels}"

        return True, None

    def setup_logging(self) -> None:
        """
        Configure logging based on configuration settings.

        Sets up:
        - Log level
        - File handlers
        - Format
        """
        log_level = getattr(logging, self.log_level, logging.INFO)

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Setup root logger
        root_logger = logging.getLogger('mle_star')
        root_logger.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(console_handler)

        # File handler
        log_file = self.logs_dir / 'mle_star.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging configured: level={self.log_level}, file={log_file}")

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(iterations={self.max_iterations}, strategies={self.parallel_strategies})"

    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return (
            f"Config("
            f"max_iterations={self.max_iterations}, "
            f"parallel_strategies={self.parallel_strategies}, "
            f"max_gpu_memory_gb={self.max_gpu_memory_gb}, "
            f"enable_monitoring={self.enable_monitoring}"
            f")"
        )
