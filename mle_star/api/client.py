"""High-level client API for MLE-STAR framework."""

from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
import logging

from ..core.workflow import STARWorkflow
from ..core.state_manager import StateManager
from ..core.base_agent import AgentConfig, AgentRole
from ..agents.planner import PlannerAgent
from ..agents.executor import ExecutorAgent
from ..agents.verifier import VerifierAgent
from ..models.openrouter_client import OpenRouterClient
from ..models.local_model import LocalModel
from ..models.model_pool import ModelPool
from ..execution.sandbox import CodeSandbox
from ..execution.validator import CodeValidator
from ..utils.config import Config
from ..tasks.task import Task
from ..tasks.kaggle_task import KaggleTask


class MLEStarClient:
    """
    High-level client for MLE-STAR framework.

    This class provides a simple, user-friendly interface to the MLE-STAR
    framework. It handles all initialization, configuration, and orchestration.

    Example:
        >>> # Create config directly
        >>> config = Config(
        ...     openrouter_api_key='sk-or-...',
        ...     kaggle_username='username',
        ...     kaggle_key='key',
        ...     planner_model={'model_name': '...', ...},
        ...     # ... other parameters
        ... )
        >>> client = MLEStarClient(config)
        >>> await client.initialize()
        >>>
        >>> # Execute a task
        >>> task = Task(description="Train a classifier...", ...)
        >>> result = await client.execute_task(task)
        >>>
        >>> # Check result
        >>> if result['status'] == 'success':
        ...     print("Task completed!")
        >>>
        >>> # Execute Kaggle competition
        >>> result = await client.execute_kaggle_competition("titanic")
    """

    def __init__(self, config: Config):
        """
        Initialize MLE-STAR client.

        Args:
            config: Configuration object
        """
        self.config = config
        self._initialized = False

        # Components (initialized later)
        self.model_pool = None
        self.planner_model = None
        self.planner = None
        self.executor = None
        self.verifier = None
        self.workflow = None
        self.state_manager = None
        self.sandbox = None
        self.validator = None

        self.logger = logging.getLogger("mle_star.client")
        self.logger.info("MLEStarClient created")

    async def initialize(self) -> None:
        """
        Initialize all components (models, agents, workflow).

        This must be called before executing tasks. It loads models
        which can take 30-60 seconds on first run.

        Example:
            >>> config = Config(...)
            >>> client = MLEStarClient(config)
            >>> await client.initialize()  # Loads models
            >>> # Now ready to execute tasks
        """
        if self._initialized:
            self.logger.warning("Client already initialized")
            return

        self.logger.info("Initializing MLE-STAR client...")

        # Initialize model pool
        self.logger.info("Setting up model pool...")
        self.model_pool = ModelPool(
            max_gpu_memory_gb=self.config.max_gpu_memory_gb
        )

        # Register local models (Executor and Verifier)
        self.logger.info("Registering models...")
        self.model_pool.register_model('executor', self.config.executor_model)
        self.model_pool.register_model('verifier', self.config.verifier_model)

        # Initialize OpenRouter client for Planner
        self.logger.info("Initializing OpenRouter client...")
        self.planner_model = OpenRouterClient(
            api_key=self.config.openrouter_api_key,
            model_id=self.config.planner_model['model_name']
        )

        # Initialize code sandbox and validator
        self.logger.info("Setting up code sandbox...")
        self.sandbox = CodeSandbox(
            max_execution_time=self.config.max_execution_time,
            max_memory_mb=self.config.max_memory_mb,
            enable_network=self.config.allow_network
        )

        self.validator = CodeValidator(
            allow_file_io=self.config.allow_file_io,
            allow_network=self.config.allow_network
        )

        # Initialize agents
        self.logger.info("Initializing agents...")

        # Planner Agent
        planner_config = AgentConfig(
            role=AgentRole.PLANNER,
            model_config=self.config.planner_model,
            temperature=self.config.planner_model.get('temperature', 0.8),
            max_tokens=self.config.planner_model.get('max_tokens', 2000)
        )
        self.planner = PlannerAgent(planner_config, self.planner_model)

        # Executor Agent (model loaded lazily)
        executor_config = AgentConfig(
            role=AgentRole.EXECUTOR,
            model_config=self.config.executor_model,
            temperature=self.config.executor_model.get('temperature', 0.2),
            max_tokens=self.config.executor_model.get('max_tokens', 4000)
        )
        # Note: Model will be loaded on first use
        self.executor = ExecutorAgent(
            executor_config,
            None,  # Model set later
            self.sandbox
        )

        # Verifier Agent (model loaded lazily)
        verifier_config = AgentConfig(
            role=AgentRole.VERIFIER,
            model_config=self.config.verifier_model,
            temperature=self.config.verifier_model.get('temperature', 0.1),
            max_tokens=self.config.verifier_model.get('max_tokens', 1500)
        )
        self.verifier = VerifierAgent(
            verifier_config,
            None  # Model set later
        )

        # Initialize state manager
        self.logger.info("Setting up state manager...")
        self.state_manager = StateManager(
            persist_dir=self.config.metrics_dir / "states"
        )

        # Initialize STAR workflow
        self.logger.info("Setting up STAR workflow...")
        self.workflow = STARWorkflow(
            planner=self.planner,
            executor=self.executor,
            verifier=self.verifier,
            state_manager=self.state_manager,
            max_iterations=self.config.max_iterations,
            parallel_strategies=self.config.parallel_strategies
        )

        self._initialized = True
        self.logger.info("✓ MLE-STAR client initialized successfully")

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task using the STAR workflow.

        Args:
            task: Task object to execute

        Returns:
            Result dictionary with status and outcome

        Raises:
            RuntimeError: If client not initialized

        Example:
            >>> from mle_star.tasks.task import Task, TaskType
            >>> task = Task(
            ...     description="Train a classifier on Iris dataset",
            ...     task_type=TaskType.CLASSIFICATION,
            ...     success_criteria=["Accuracy > 0.95"]
            ... )
            >>> result = await client.execute_task(task)
            >>> print(f"Status: {result['status']}")
        """
        if not self._initialized:
            raise RuntimeError(
                "Client not initialized. Call await client.initialize() first."
            )

        self.logger.info(f"Executing task: {task.description[:50]}...")

        # Load models for executor and verifier (lazy loading)
        if self.executor.model is None:
            self.logger.info("Loading executor model (Qwen2.5-Coder 32B)...")
            self.executor.model = self.model_pool.get_model('executor')

        if self.verifier.model is None:
            self.logger.info("Loading verifier model (Qwen2.5-Coder 14B)...")
            self.verifier.model = self.model_pool.get_model('verifier')

        # Execute workflow
        result = await self.workflow.execute(task)

        self.logger.info(f"Task completed with status: {result['status']}")

        return result

    async def execute_kaggle_competition(
        self,
        competition_name: str,
        data_dir: Optional[Path] = None,
        evaluation_metric: Optional[str] = None,
        description: Optional[str] = None,
        auto_enrich: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a Kaggle competition task with full automation.

        Automatically downloads data, detects evaluation metric, and detects
        submission format when not provided. Requires Kaggle API credentials
        for auto-download feature (~/.kaggle/kaggle.json).

        Args:
            competition_name: Name of the competition (e.g., "titanic")
            data_dir: Directory containing competition data. If None, auto-downloads
                     to ~/.cache/mle_star/competitions/{competition_name}/
            evaluation_metric: Competition evaluation metric. If None, auto-detected
                              from Kaggle API metadata.
            description: Optional task description (overrides auto-generation)
            auto_enrich: If True, automatically fetch competition metadata
                        and profile dataset. If False, use minimal description.

        Returns:
            Result dictionary with status, result, and metadata

        Examples:
            >>> # Minimal usage - just competition name! (recommended)
            >>> result = await client.execute_kaggle_competition("titanic")
            >>>
            >>> # With manual data directory (backward compatible)
            >>> result = await client.execute_kaggle_competition(
            ...     "titanic",
            ...     data_dir=Path("./my_data/titanic")
            ... )
            >>>
            >>> # With manual metric override
            >>> result = await client.execute_kaggle_competition(
            ...     "titanic",
            ...     evaluation_metric="f1"
            ... )
            >>>
            >>> # With custom description
            >>> result = await client.execute_kaggle_competition(
            ...     "titanic",
            ...     description="Custom task description",
            ...     auto_enrich=False
            ... )
        """
        if not self._initialized:
            raise RuntimeError(
                "Client not initialized. Call await client.initialize() first."
            )

        # Create Kaggle task with auto-enrichment
        task = KaggleTask(
            competition_name=competition_name,
            data_dir=data_dir,
            evaluation_metric=evaluation_metric,
            description=description,
            auto_enrich=auto_enrich,
            kaggle_username=self.config.kaggle_username,
            kaggle_key=self.config.kaggle_key
        )

        # Validate data files
        file_status = task.validate_data_files()
        missing = [name for name, exists in file_status.items() if not exists]

        if missing:
            self.logger.warning(f"Missing data files: {missing}")

        # Execute task
        return await self.execute_task(task)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the client and workflow.

        Returns:
            Dictionary with statistics

        Example:
            >>> stats = client.get_stats()
            >>> print(f"Tasks processed: {stats['workflow']['planner_stats']['messages_processed']}")
        """
        if not self._initialized:
            return {'error': 'Client not initialized'}

        stats = {
            'initialized': self._initialized,
            'config': {
                'max_iterations': self.config.max_iterations,
                'parallel_strategies': self.config.parallel_strategies,
                'max_gpu_memory_gb': self.config.max_gpu_memory_gb
            },
            'models': {},
            'workflow': {}
        }

        # Model pool stats
        if self.model_pool:
            stats['models'] = self.model_pool.get_memory_stats()

        # Workflow stats
        if self.workflow:
            stats['workflow'] = self.workflow.get_stats()

        return stats

    async def cleanup(self) -> None:
        """
        Cleanup resources (unload models, clear cache).

        Call this when done with the client to free up GPU memory.

        Example:
            >>> await client.cleanup()
        """
        self.logger.info("Cleaning up resources...")

        if self.model_pool:
            self.model_pool.unload_all()

        self._initialized = False
        self.logger.info("✓ Cleanup complete")

    def __str__(self) -> str:
        """String representation."""
        status = "initialized" if self._initialized else "not initialized"
        return f"MLEStarClient({status})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"MLEStarClient("
            f"initialized={self._initialized}, "
            f"max_iterations={self.config.max_iterations}"
            f")"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
