"""State management for tasks and workflow execution."""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json
import logging

from .message import Message


class StateManager:
    """
    Manages task state and execution context throughout the STAR workflow.

    This class provides persistent storage for task state, including:
    - Task metadata and configuration
    - Iteration history
    - Message exchanges
    - Metrics and results

    State is persisted to disk as JSON files for debugging and recovery.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        """
        Initialize the state manager.

        Args:
            persist_dir: Directory for persisting state files.
                If None, state is only kept in memory.
        """
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.logger = logging.getLogger("mle_star.state_manager")

        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"State persistence enabled: {self.persist_dir}")

    def create_task(self, task: Any) -> str:
        """
        Create a new task state.

        Args:
            task: Task object (must have task_id and to_dict() method)

        Returns:
            Task ID

        Example:
            >>> task = Task(description="Train model", task_type=TaskType.CLASSIFICATION)
            >>> task_id = state_manager.create_task(task)
        """
        task_id = task.task_id
        self.tasks[task_id] = {
            'task': task.to_dict(),
            'status': 'active',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'iterations': [],
            'messages': [],
            'metrics': {},
            'result': None,
            'completed_at': None
        }

        self._persist(task_id)
        self.logger.info(f"Created task state: {task_id}")
        return task_id

    def update_task(self, task_id: str, **kwargs) -> None:
        """
        Update task state with arbitrary key-value pairs.

        Args:
            task_id: Task ID
            **kwargs: Key-value pairs to update

        Raises:
            KeyError: If task_id doesn't exist

        Example:
            >>> state_manager.update_task(
            ...     task_id,
            ...     status='completed',
            ...     score=0.95
            ... )
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        self.tasks[task_id].update(kwargs)
        self.tasks[task_id]['updated_at'] = datetime.utcnow().isoformat()
        self._persist(task_id)

    def update_iteration(self, task_id: str, iteration_data: Dict[str, Any]) -> None:
        """
        Add data for a new iteration.

        Args:
            task_id: Task ID
            iteration_data: Dictionary containing iteration information

        Raises:
            KeyError: If task_id doesn't exist

        Example:
            >>> state_manager.update_iteration(task_id, {
            ...     'iteration': 1,
            ...     'strategies': [...],
            ...     'best_score': 0.85
            ... })
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        iteration_data['timestamp'] = datetime.utcnow().isoformat()
        self.tasks[task_id]['iterations'].append(iteration_data)
        self.tasks[task_id]['updated_at'] = datetime.utcnow().isoformat()
        self._persist(task_id)

        self.logger.debug(
            f"Updated iteration {len(self.tasks[task_id]['iterations'])} for task {task_id}"
        )

    def add_message(self, task_id: str, message: Message) -> None:
        """
        Add a message to the task's message history.

        Args:
            task_id: Task ID
            message: Message instance

        Raises:
            KeyError: If task_id doesn't exist
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        self.tasks[task_id]['messages'].append(message.to_dict())
        self.tasks[task_id]['updated_at'] = datetime.utcnow().isoformat()

        # Persist only every 10 messages to reduce I/O
        if len(self.tasks[task_id]['messages']) % 10 == 0:
            self._persist(task_id)

    def update_metrics(self, task_id: str, metrics: Dict[str, Any]) -> None:
        """
        Update task metrics.

        Args:
            task_id: Task ID
            metrics: Dictionary of metrics to update/add

        Raises:
            KeyError: If task_id doesn't exist
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        self.tasks[task_id]['metrics'].update(metrics)
        self.tasks[task_id]['updated_at'] = datetime.utcnow().isoformat()
        self._persist(task_id)

    def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        Mark task as completed with final result.

        Args:
            task_id: Task ID
            result: Final result dictionary

        Raises:
            KeyError: If task_id doesn't exist
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        self.tasks[task_id]['status'] = 'completed'
        self.tasks[task_id]['result'] = result
        self.tasks[task_id]['completed_at'] = datetime.utcnow().isoformat()
        self.tasks[task_id]['updated_at'] = datetime.utcnow().isoformat()
        self._persist(task_id)

        self.logger.info(f"Task completed: {task_id}")

    def fail_task(self, task_id: str, reason: str) -> None:
        """
        Mark task as failed with reason.

        Args:
            task_id: Task ID
            reason: Failure reason

        Raises:
            KeyError: If task_id doesn't exist
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        self.tasks[task_id]['status'] = 'failed'
        self.tasks[task_id]['failure_reason'] = reason
        self.tasks[task_id]['completed_at'] = datetime.utcnow().isoformat()
        self.tasks[task_id]['updated_at'] = datetime.utcnow().isoformat()
        self._persist(task_id)

        self.logger.warning(f"Task failed: {task_id} - {reason}")

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve task state.

        Args:
            task_id: Task ID

        Returns:
            Task state dictionary or None if not found
        """
        return self.tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        Get task status.

        Args:
            task_id: Task ID

        Returns:
            Task status string or None if not found
        """
        task = self.tasks.get(task_id)
        return task['status'] if task else None

    def get_iterations(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get all iterations for a task.

        Args:
            task_id: Task ID

        Returns:
            List of iteration dictionaries

        Raises:
            KeyError: If task_id doesn't exist
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        return self.tasks[task_id]['iterations']

    def get_messages(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages for a task.

        Args:
            task_id: Task ID

        Returns:
            List of message dictionaries

        Raises:
            KeyError: If task_id doesn't exist
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        return self.tasks[task_id]['messages']

    def list_tasks(self, status: Optional[str] = None) -> List[str]:
        """
        List all task IDs, optionally filtered by status.

        Args:
            status: Optional status filter ('active', 'completed', 'failed')

        Returns:
            List of task IDs
        """
        if status:
            return [
                task_id for task_id, task_data in self.tasks.items()
                if task_data['status'] == status
            ]
        return list(self.tasks.keys())

    def delete_task(self, task_id: str) -> None:
        """
        Delete a task and its persisted state.

        Args:
            task_id: Task ID

        Raises:
            KeyError: If task_id doesn't exist
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        # Delete from memory
        del self.tasks[task_id]

        # Delete persisted file
        if self.persist_dir:
            file_path = self.persist_dir / f"{task_id}.json"
            if file_path.exists():
                file_path.unlink()

        self.logger.info(f"Deleted task: {task_id}")

    def _persist(self, task_id: str) -> None:
        """
        Persist task state to disk.

        Args:
            task_id: Task ID to persist
        """
        if not self.persist_dir:
            return

        if task_id not in self.tasks:
            return

        file_path = self.persist_dir / f"{task_id}.json"

        try:
            with open(file_path, 'w') as f:
                json.dump(self.tasks[task_id], f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to persist task {task_id}: {e}")

    def load_task(self, task_id: str) -> bool:
        """
        Load task state from disk.

        Args:
            task_id: Task ID to load

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.persist_dir:
            self.logger.warning("No persist_dir configured, cannot load task")
            return False

        file_path = self.persist_dir / f"{task_id}.json"

        if not file_path.exists():
            self.logger.warning(f"Task file not found: {file_path}")
            return False

        try:
            with open(file_path, 'r') as f:
                self.tasks[task_id] = json.load(f)
            self.logger.info(f"Loaded task from disk: {task_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load task {task_id}: {e}")
            return False

    def load_all_tasks(self) -> int:
        """
        Load all tasks from persist directory.

        Returns:
            Number of tasks loaded
        """
        if not self.persist_dir:
            self.logger.warning("No persist_dir configured, cannot load tasks")
            return 0

        if not self.persist_dir.exists():
            return 0

        loaded = 0
        for file_path in self.persist_dir.glob("*.json"):
            task_id = file_path.stem
            if self.load_task(task_id):
                loaded += 1

        self.logger.info(f"Loaded {loaded} tasks from disk")
        return loaded

    def clear_all(self) -> None:
        """Clear all tasks from memory (does not delete persisted files)."""
        self.tasks.clear()
        self.logger.info("Cleared all tasks from memory")

    def __len__(self) -> int:
        """Return number of tasks in memory."""
        return len(self.tasks)

    def __contains__(self, task_id: str) -> bool:
        """Check if task exists in memory."""
        return task_id in self.tasks
