"""Pytest configuration and shared fixtures for MLE-STAR tests."""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
from typing import Generator, Dict, Any

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mle_star.core.message import Message, MessageType
from mle_star.core.state_manager import StateManager
from mle_star.tasks.task import Task, TaskType, Subtask
from mle_star.monitoring.metrics import MetricsCollector


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests that require models")


# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Temporary directory fixtures
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp(prefix="mle_star_test_"))
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def state_dir(temp_dir: Path) -> Path:
    """Create temporary state directory."""
    state_path = temp_dir / "state"
    state_path.mkdir(parents=True, exist_ok=True)
    return state_path


@pytest.fixture
def metrics_dir(temp_dir: Path) -> Path:
    """Create temporary metrics directory."""
    metrics_path = temp_dir / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    return metrics_path


# State manager fixture
@pytest.fixture
def state_manager(state_dir: Path) -> StateManager:
    """Create StateManager instance."""
    return StateManager(state_dir=state_dir)


# Metrics collector fixture
@pytest.fixture
def metrics_collector(metrics_dir: Path) -> MetricsCollector:
    """Create MetricsCollector instance."""
    return MetricsCollector(metrics_dir=metrics_dir)


# Message fixtures
@pytest.fixture
def sample_message() -> Message:
    """Create sample message for testing."""
    return Message(
        msg_type=MessageType.TASK_REQUEST,
        sender="test_sender",
        receiver="test_receiver",
        content={"test": "data"}
    )


@pytest.fixture
def task_request_message() -> Message:
    """Create task request message."""
    return Message(
        msg_type=MessageType.TASK_REQUEST,
        sender="workflow",
        receiver="planner",
        content={
            'task': {
                'description': 'Train a classifier on Iris dataset',
                'task_type': 'classification'
            },
            'iteration': 0,
            'num_strategies': 3
        }
    )


# Task fixtures
@pytest.fixture
def simple_task() -> Task:
    """Create simple classification task."""
    return Task(
        description="Train a classifier on Iris dataset",
        task_type=TaskType.CLASSIFICATION,
        success_criteria=[
            "Accuracy > 0.90",
            "Model trains successfully"
        ]
    )


@pytest.fixture
def complex_task() -> Task:
    """Create complex task with subtasks."""
    subtasks = [
        Subtask(
            description="Load and explore Iris dataset",
            order=1,
            dependencies=[]
        ),
        Subtask(
            description="Train RandomForest classifier",
            order=2,
            dependencies=[1]
        ),
        Subtask(
            description="Evaluate model performance",
            order=3,
            dependencies=[2]
        )
    ]

    return Task(
        description="Complete ML pipeline for Iris classification",
        task_type=TaskType.CLASSIFICATION,
        subtasks=subtasks,
        success_criteria=["Accuracy > 0.95"],
        constraints=["Use scikit-learn only"]
    )


@pytest.fixture
def sample_strategy() -> Dict[str, Any]:
    """Create sample strategy for testing."""
    return {
        'name': 'RandomForest Baseline',
        'approach': 'Use RandomForestClassifier with default parameters',
        'subtasks': [
            {
                'description': 'Load Iris dataset',
                'order': 1,
                'implementation': 'Use sklearn.datasets.load_iris()'
            },
            {
                'description': 'Train RandomForest',
                'order': 2,
                'implementation': 'Fit RandomForestClassifier'
            },
            {
                'description': 'Evaluate',
                'order': 3,
                'implementation': 'Compute accuracy_score'
            }
        ],
        'expected_outcome': 'Accuracy around 0.95'
    }


@pytest.fixture
def sample_code() -> str:
    """Create sample Python code for testing."""
    return """
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
"""


@pytest.fixture
def dangerous_code() -> str:
    """Create dangerous code for security testing."""
    return """
import subprocess
import os

# This should be blocked!
subprocess.run(['rm', '-rf', '/'])
os.system('echo dangerous')
eval('__import__("os").system("ls")')
"""


@pytest.fixture
def safe_code_with_warnings() -> str:
    """Create safe code that should trigger warnings."""
    return """
import pandas as pd

# File operations (should warn)
with open('data.csv', 'r') as f:
    data = f.read()

df = pd.DataFrame({'a': [1, 2, 3]})
print(df)
"""


# Mock configuration
@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Create mock configuration."""
    return {
        'planner': {
            'role': 'planner',
            'model_type': 'openrouter',
            'model_config': {
                'model_id': 'meta-llama/llama-3.3-70b-instruct:free',
                'temperature': 0.8,
                'max_tokens': 2000
            }
        },
        'executor': {
            'role': 'executor',
            'model_type': 'local',
            'model_config': {
                'model_name': 'Qwen/Qwen2.5-Coder-32B-Instruct',
                'temperature': 0.2,
                'max_tokens': 4000
            }
        },
        'verifier': {
            'role': 'verifier',
            'model_type': 'local',
            'model_config': {
                'model_name': 'Qwen/Qwen2.5-Coder-14B-Instruct',
                'temperature': 0.1,
                'max_tokens': 1500
            }
        }
    }


# Skip markers for optional dependencies
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip markers."""
    for item in items:
        # Skip slow tests unless --slow flag is passed
        if "slow" in item.keywords and not config.getoption("--slow", default=False):
            item.add_marker(pytest.mark.skip(reason="Slow test, use --slow to run"))


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests that require model loading"
    )
