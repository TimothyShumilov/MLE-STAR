"""Task data models for MLE-STAR framework."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import uuid


class TaskType(Enum):
    """Types of ML tasks supported by the framework."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    RECOMMENDATION = "recommendation"
    CUSTOM = "custom"


class TaskStatus(Enum):
    """Status of a task during execution."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """
    Generic ML task representation.

    This class represents a machine learning task that can be solved
    by the MLE-STAR framework. It includes all necessary information
    for the agents to understand and solve the problem.

    Example:
        >>> task = Task(
        ...     description="Train a classifier on Iris dataset",
        ...     task_type=TaskType.CLASSIFICATION,
        ...     success_criteria=["Accuracy > 0.95"],
        ...     target_metric="accuracy"
        ... )
    """

    description: str
    task_type: TaskType

    # Optional fields
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    constraints: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    data_path: Optional[str] = None
    target_metric: Optional[str] = None
    baseline_score: Optional[float] = None

    # Status
    status: TaskStatus = TaskStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary representation.

        Returns:
            Dictionary containing all task fields
        """
        return {
            'task_id': self.task_id,
            'description': self.description,
            'type': self.task_type.value,
            'constraints': self.constraints,
            'success_criteria': self.success_criteria,
            'data_path': self.data_path,
            'target_metric': self.target_metric,
            'baseline_score': self.baseline_score,
            'status': self.status.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """
        Create Task instance from dictionary.

        Args:
            data: Dictionary containing task data

        Returns:
            Task instance

        Raises:
            ValueError: If required fields are missing
        """
        try:
            return cls(
                task_id=data.get('task_id', str(uuid.uuid4())),
                description=data['description'],
                task_type=TaskType(data['type']),
                constraints=data.get('constraints', {}),
                success_criteria=data.get('success_criteria', []),
                data_path=data.get('data_path'),
                target_metric=data.get('target_metric'),
                baseline_score=data.get('baseline_score'),
                status=TaskStatus(data.get('status', 'pending'))
            )
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid task data: {e}")

    def add_constraint(self, key: str, value: Any) -> None:
        """
        Add a constraint to the task.

        Args:
            key: Constraint name
            value: Constraint value
        """
        self.constraints[key] = value

    def add_success_criterion(self, criterion: str) -> None:
        """
        Add a success criterion.

        Args:
            criterion: Success criterion description
        """
        if criterion not in self.success_criteria:
            self.success_criteria.append(criterion)

    def __str__(self) -> str:
        """String representation of the task."""
        return (
            f"Task(id={self.task_id[:8]}..., "
            f"type={self.task_type.value}, "
            f"status={self.status.value})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the task."""
        return (
            f"Task("
            f"task_id='{self.task_id}', "
            f"description='{self.description[:50]}...', "
            f"task_type={self.task_type}, "
            f"status={self.status}"
            f")"
        )


@dataclass
class Subtask:
    """
    Subtask within a larger task.

    Subtasks represent individual steps that need to be completed
    as part of executing a strategy.
    """

    subtask_id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    estimated_time: Optional[int] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert subtask to dictionary."""
        return {
            'subtask_id': self.subtask_id,
            'description': self.description,
            'dependencies': self.dependencies,
            'estimated_time': self.estimated_time,
            'status': self.status.value,
            'result': self.result
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Subtask':
        """Create Subtask from dictionary."""
        return cls(
            subtask_id=data['subtask_id'],
            description=data['description'],
            dependencies=data.get('dependencies', []),
            estimated_time=data.get('estimated_time'),
            status=TaskStatus(data.get('status', 'pending')),
            result=data.get('result')
        )

    def mark_completed(self, result: Dict[str, Any]) -> None:
        """
        Mark subtask as completed with result.

        Args:
            result: Result dictionary
        """
        self.status = TaskStatus.COMPLETED
        self.result = result

    def mark_failed(self, error: str) -> None:
        """
        Mark subtask as failed with error.

        Args:
            error: Error message
        """
        self.status = TaskStatus.FAILED
        self.result = {'error': error}

    def is_ready(self, completed_subtasks: set) -> bool:
        """
        Check if subtask is ready to execute (dependencies met).

        Args:
            completed_subtasks: Set of completed subtask IDs

        Returns:
            True if ready, False otherwise
        """
        return all(dep in completed_subtasks for dep in self.dependencies)


@dataclass
class MLTask(Task):
    """
    ML-specific task with additional ML-related fields.

    This extends the base Task class with fields specific to
    machine learning tasks.
    """

    # ML-specific fields
    dataset_info: Optional[Dict[str, Any]] = None
    feature_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    test_size: float = 0.2
    random_state: int = 42
    cross_validation: bool = False
    cv_folds: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including ML-specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'dataset_info': self.dataset_info,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'cross_validation': self.cross_validation,
            'cv_folds': self.cv_folds
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLTask':
        """Create MLTask from dictionary."""
        # Extract base task fields
        base_task = Task.from_dict(data)

        # Create MLTask with additional fields
        return cls(
            task_id=base_task.task_id,
            description=base_task.description,
            task_type=base_task.task_type,
            constraints=base_task.constraints,
            success_criteria=base_task.success_criteria,
            data_path=base_task.data_path,
            target_metric=base_task.target_metric,
            baseline_score=base_task.baseline_score,
            status=base_task.status,
            dataset_info=data.get('dataset_info'),
            feature_columns=data.get('feature_columns'),
            target_column=data.get('target_column'),
            test_size=data.get('test_size', 0.2),
            random_state=data.get('random_state', 42),
            cross_validation=data.get('cross_validation', False),
            cv_folds=data.get('cv_folds', 5)
        )


def create_task_from_description(
    description: str,
    task_type: Optional[TaskType] = None,
    **kwargs
) -> Task:
    """
    Convenience function to create a task from a description.

    Args:
        description: Task description
        task_type: Task type (auto-detected if None)
        **kwargs: Additional task parameters

    Returns:
        Task instance

    Example:
        >>> task = create_task_from_description(
        ...     "Train a random forest classifier on iris data",
        ...     success_criteria=["Accuracy > 0.9"]
        ... )
    """
    # Auto-detect task type if not provided
    if task_type is None:
        description_lower = description.lower()

        if any(word in description_lower for word in ['classif', 'predict class']):
            task_type = TaskType.CLASSIFICATION
        elif any(word in description_lower for word in ['regress', 'predict value']):
            task_type = TaskType.REGRESSION
        elif any(word in description_lower for word in ['cluster', 'group']):
            task_type = TaskType.CLUSTERING
        elif any(word in description_lower for word in ['time series', 'forecast']):
            task_type = TaskType.TIME_SERIES
        elif any(word in description_lower for word in ['text', 'nlp', 'language']):
            task_type = TaskType.NLP
        elif any(word in description_lower for word in ['image', 'vision', 'cnn']):
            task_type = TaskType.COMPUTER_VISION
        else:
            task_type = TaskType.CUSTOM

    return Task(
        description=description,
        task_type=task_type,
        **kwargs
    )
