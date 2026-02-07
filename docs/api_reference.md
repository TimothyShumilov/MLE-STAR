# MLE-STAR API Reference

Complete API reference for the MLE-STAR framework.

## Table of Contents

1. [Public API](#public-api)
2. [Task Models](#task-models)
3. [Core Components](#core-components)
4. [Agents](#agents)
5. [Monitoring](#monitoring)
6. [Configuration](#configuration)

---

## Public API

### MLEStarClient

**File:** `mle_star/api/client.py`

High-level client for executing ML tasks.

#### Initialization

```python
from mle_star.api.client import MLEStarClient

# From environment variables (.env file)
async with MLEStarClient.from_env() as client:
    # Use client
    pass

# From config file
async with MLEStarClient.from_config("configs/agents.yaml") as client:
    # Use client
    pass

# Manual initialization
client = MLEStarClient(
    config_path="configs/agents.yaml",
    state_dir="./state",
    metrics_dir="./metrics"
)
await client.initialize()
```

#### Methods

##### `async initialize()`

Initialize the client (load models, setup agents).

```python
await client.initialize()
```

**Note:** Automatically called when using context manager.

---

##### `async execute_task(task: Task) -> Dict[str, Any]`

Execute a generic ML task.

```python
from mle_star.tasks.task import Task, TaskType

task = Task(
    description="Train a classifier on Iris dataset",
    task_type=TaskType.CLASSIFICATION,
    success_criteria=["Accuracy > 0.95"]
)

result = await client.execute_task(task)
```

**Parameters:**
- `task` (Task): Task to execute

**Returns:**
- `Dict[str, Any]`: Result dictionary with:
  - `status` (str): 'success', 'failed', or 'partial_success'
  - `result` (Dict): Best result from workflow
  - `iterations` (int): Number of iterations used
  - `task_id` (str): Task identifier

**Example Result:**
```python
{
    'status': 'success',
    'result': {
        'strategy': {...},
        'execution': {...},
        'verification': {'score': 0.95, ...},
        'code': "..."
    },
    'iterations': 3,
    'task_id': 'uuid-here'
}
```

---

##### `async execute_kaggle_competition(...) -> Dict[str, Any]`

Execute a Kaggle competition task.

```python
result = await client.execute_kaggle_competition(
    competition_name="titanic",
    data_dir="./data/titanic",
    evaluation_metric="accuracy"
)
```

**Parameters:**
- `competition_name` (str): Name of the competition
- `data_dir` (str): Directory containing competition data
- `evaluation_metric` (str): Metric to optimize
- `description` (str, optional): Custom task description
- `success_criteria` (List[str], optional): Custom criteria

**Returns:**
- `Dict[str, Any]`: Same as `execute_task()`

---

##### `async close()`

Close client and cleanup resources.

```python
await client.close()
```

**Note:** Automatically called when using context manager.

---

#### Usage Examples

**Basic Example:**
```python
import asyncio
from mle_star.api.client import MLEStarClient
from mle_star.tasks.task import Task, TaskType

async def main():
    async with MLEStarClient.from_env() as client:
        task = Task(
            description="Train a classifier on Iris dataset",
            task_type=TaskType.CLASSIFICATION,
            success_criteria=["Accuracy > 0.95"]
        )

        result = await client.execute_task(task)

        if result['status'] == 'success':
            print(f"Success! Score: {result['result']['verification']['score']}")
            print(f"Code:\n{result['result']['code']}")
        else:
            print(f"Failed: {result.get('reason', 'Unknown')}")

asyncio.run(main())
```

**Kaggle Example:**
```python
async def kaggle_example():
    async with MLEStarClient.from_env() as client:
        result = await client.execute_kaggle_competition(
            competition_name="titanic",
            data_dir="./data/titanic",
            evaluation_metric="accuracy"
        )

        print(f"Status: {result['status']}")
        print(f"Score: {result['result']['verification']['score']}")
```

---

## Task Models

### Task

**File:** `mle_star/tasks/task.py`

Generic task model.

#### Constructor

```python
from mle_star.tasks.task import Task, TaskType, TaskStatus, Subtask

task = Task(
    description: str,
    task_type: TaskType,
    subtasks: Optional[List[Subtask]] = None,
    success_criteria: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
)
```

**Parameters:**
- `description` (str): Task description
- `task_type` (TaskType): Type of ML task
- `subtasks` (List[Subtask], optional): List of subtasks
- `success_criteria` (List[str], optional): Success criteria
- `constraints` (List[str], optional): Task constraints
- `context` (Dict, optional): Additional context
- `metadata` (Dict, optional): Metadata

#### Task Types

```python
class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENERAL = "general"
```

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert task to dictionary.

```python
task_dict = task.to_dict()
```

##### `from_dict(data: Dict[str, Any]) -> Task`

Create task from dictionary.

```python
task = Task.from_dict(task_dict)
```

#### Example

```python
task = Task(
    description="Predict house prices based on features",
    task_type=TaskType.REGRESSION,
    success_criteria=[
        "RMSE < 50000",
        "R² > 0.85"
    ],
    constraints=[
        "Use only numerical features",
        "No data leakage"
    ]
)
```

---

### Subtask

```python
@dataclass
class Subtask:
    description: str
    order: int
    dependencies: List[int] = field(default_factory=list)
    estimated_duration: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Example:**
```python
subtasks = [
    Subtask(description="Load and explore data", order=1),
    Subtask(description="Feature engineering", order=2, dependencies=[1]),
    Subtask(description="Train model", order=3, dependencies=[2]),
    Subtask(description="Evaluate", order=4, dependencies=[3])
]
```

---

### MLTask

**File:** `mle_star/tasks/task.py`

Extended task model with ML-specific fields.

```python
from mle_star.tasks.task import MLTask

task = MLTask(
    description="Train classifier",
    task_type=TaskType.CLASSIFICATION,
    dataset_info={
        'name': 'iris',
        'samples': 150,
        'features': 4,
        'classes': 3
    },
    target_metric="accuracy",
    baseline_performance=0.90,
    expected_performance=0.95
)
```

---

### KaggleTask

**File:** `mle_star/tasks/kaggle_task.py`

Kaggle-specific task model.

```python
from mle_star.tasks.kaggle_task import KaggleTask

task = KaggleTask(
    competition_name="titanic",
    data_dir="./data/titanic",
    evaluation_metric="accuracy",
    submission_format="csv"
)
```

---

## Core Components

### Message

**File:** `mle_star/core/message.py`

Message for agent communication.

```python
from mle_star.core.message import Message, MessageType

msg = Message(
    msg_type=MessageType.TASK_REQUEST,
    sender="workflow",
    receiver="planner",
    content={'task': task.to_dict()}
)
```

#### Message Types

```python
class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_DECOMPOSITION = "task_decomposition"
    EXECUTION_REQUEST = "execution_request"
    EXECUTION_RESULT = "execution_result"
    VERIFICATION_REQUEST = "verification_request"
    VERIFICATION_RESULT = "verification_result"
    ERROR = "error"
    RESULT = "result"
```

#### Methods

##### `create_reply(msg_type: MessageType, content: Dict) -> Message`

Create reply message.

```python
reply = msg.create_reply(
    msg_type=MessageType.TASK_DECOMPOSITION,
    content={'strategies': [...]}
)
```

##### `to_dict() -> Dict[str, Any]`

Serialize to dictionary.

##### `from_dict(data: Dict[str, Any]) -> Message`

Deserialize from dictionary.

---

### StateManager

**File:** `mle_star/core/state_manager.py`

Manages task state persistence.

```python
from mle_star.core.state_manager import StateManager

state_manager = StateManager(state_dir="./state")
```

#### Methods

##### `create_task(task: Task) -> str`

Create new task and return ID.

```python
task_id = state_manager.create_task(task)
```

##### `update_iteration(task_id: str, data: Dict[str, Any])`

Update iteration data.

```python
state_manager.update_iteration(task_id, {
    'iteration': 0,
    'strategies_count': 3,
    'results': [...]
})
```

##### `complete_task(task_id: str, result: Dict[str, Any])`

Mark task as completed.

```python
state_manager.complete_task(task_id, result)
```

##### `fail_task(task_id: str, error: str)`

Mark task as failed.

```python
state_manager.fail_task(task_id, "Error message")
```

##### `load_task(task_id: str) -> Optional[Dict[str, Any]]`

Load task state.

```python
task_data = state_manager.load_task(task_id)
```

---

## Agents

### BaseAgent

**File:** `mle_star/core/base_agent.py`

Abstract base class for agents.

```python
from mle_star.core.base_agent import BaseAgent, AgentRole

class MyAgent(BaseAgent):
    async def process(self, message: Message) -> Message:
        # Implementation
        pass

    def validate_input(self, message: Message) -> bool:
        # Validation
        return True
```

#### Agent Roles

```python
class AgentRole(Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    WORKFLOW = "workflow"
```

---

## Monitoring

### MetricsCollector

**File:** `mle_star/monitoring/metrics.py`

Collect and aggregate metrics.

```python
from mle_star.monitoring.metrics import MetricsCollector

collector = MetricsCollector(metrics_dir="./metrics")
```

#### Methods

##### `start_task(task_id: str, task_type: str, metadata: Optional[Dict] = None)`

Start tracking a task.

```python
collector.start_task("task_123", "classification")
```

##### `record_iteration(task_id: str, strategies: int, best_score: float)`

Record an iteration.

```python
collector.record_iteration("task_123", strategies=3, best_score=0.85)
```

##### `record_execution(task_id: str, success: bool, score: float = 0.0, timeout: bool = False)`

Record code execution.

```python
collector.record_execution("task_123", success=True, score=0.90)
```

##### `record_agent_call(task_id: str, agent_role: str)`

Record agent call.

```python
collector.record_agent_call("task_123", "planner")
```

##### `end_task(task_id: str, status: str)`

End task tracking.

```python
collector.end_task("task_123", "success")
```

##### `get_aggregate_stats() -> AggregateMetrics`

Get aggregate statistics.

```python
stats = collector.get_aggregate_stats()
print(f"Success rate: {stats.success_rate:.1%}")
```

---

### ResourceMonitor

**File:** `mle_star/monitoring/resource_monitor.py`

Monitor system resources.

```python
from mle_star.monitoring.resource_monitor import ResourceMonitor

monitor = ResourceMonitor(
    gpu_memory_threshold_mb=28000,
    cpu_threshold_percent=90.0
)
```

#### Methods

##### `start(interval: float = 10.0)`

Start background monitoring.

```python
monitor.start(interval=10.0)  # Every 10 seconds
```

##### `stop()`

Stop monitoring.

```python
monitor.stop()
```

##### `get_current_snapshot() -> ResourceSnapshot`

Get current resource usage.

```python
snapshot = monitor.get_current_snapshot()
print(f"GPU: {snapshot.gpu_memory_allocated_mb}MB")
print(f"CPU: {snapshot.cpu_percent}%")
```

##### `get_peak_usage() -> Dict[str, float]`

Get peak resource usage.

```python
peaks = monitor.get_peak_usage()
```

##### `clear_gpu_cache()`

Clear GPU cache.

```python
monitor.clear_gpu_cache()
```

---

### GuardrailsManager

**File:** `mle_star/monitoring/guardrails.py`

Safety guardrails and validation.

```python
from mle_star.monitoring.guardrails import GuardrailsManager

guardrails = GuardrailsManager(
    strict_mode=False,
    max_api_calls_per_day=50
)
```

#### Methods

##### `validate_task_input(description: str, config: Optional[Dict] = None) -> ValidationResult`

Validate task input.

```python
result = guardrails.validate_task_input(
    "Train a classifier on Iris dataset"
)

if not result.valid:
    print(f"Validation failed: {result.issues}")
```

##### `validate_code_output(code: str) -> ValidationResult`

Validate generated code.

```python
result = guardrails.validate_code_output(code)
```

##### `check_rate_limit(operation: str) -> bool`

Check rate limit.

```python
if not guardrails.check_rate_limit('api_call'):
    raise Exception("Rate limit exceeded")
```

##### `get_status() -> Dict[str, Any]`

Get guardrails status.

```python
status = guardrails.get_status()
print(f"API calls remaining: {status['rate_limits']['api_calls_remaining']}")
```

---

## Configuration

### Config

**File:** `mle_star/utils/config.py`

Configuration management.

```python
from mle_star.utils.config import Config

# From YAML file
config = Config.from_yaml("configs/agents.yaml")

# From environment
config = Config.from_env()
```

#### Environment Variables

```bash
# .env file
OPENROUTER_API_KEY=your_key_here

# Optional
MAX_GPU_MEMORY_GB=28
MAX_ITERATIONS=5
PARALLEL_STRATEGIES=3
```

---

## Error Handling

### Common Exceptions

```python
from mle_star.core.exceptions import (
    TaskExecutionError,
    ValidationError,
    SecurityError,
    ResourceLimitError
)

try:
    result = await client.execute_task(task)
except ValidationError as e:
    print(f"Validation failed: {e}")
except ResourceLimitError as e:
    print(f"Resource limit exceeded: {e}")
except TaskExecutionError as e:
    print(f"Execution failed: {e}")
```

---

## Best Practices

### 1. Always Use Context Managers

```python
# Good
async with MLEStarClient.from_env() as client:
    result = await client.execute_task(task)

# Bad - must manually close
client = MLEStarClient.from_env()
await client.initialize()
result = await client.execute_task(task)
await client.close()  # Easy to forget!
```

### 2. Set Appropriate Success Criteria

```python
# Good - specific and measurable
task = Task(
    description="...",
    success_criteria=[
        "Accuracy > 0.95",
        "Training completes without errors",
        "Code runs in < 60 seconds"
    ]
)

# Bad - vague
task = Task(
    description="...",
    success_criteria=["Good performance"]
)
```

### 3. Handle Failures Gracefully

```python
result = await client.execute_task(task)

if result['status'] == 'success':
    print("Task completed successfully")
elif result['status'] == 'partial_success':
    print(f"Partial success after {result['iterations']} iterations")
    print(f"Best score: {result['result']['verification']['score']}")
else:
    print(f"Task failed: {result.get('reason', 'Unknown')}")
```

### 4. Monitor Resource Usage

```python
from mle_star.monitoring.resource_monitor import ResourceMonitor

monitor = ResourceMonitor()
monitor.start()

# Execute tasks
result = await client.execute_task(task)

# Check resources
monitor.log_summary()
monitor.stop()
```

### 5. Use Metrics for Analysis

```python
from mle_star.monitoring.metrics import MetricsCollector

collector = MetricsCollector()

# After running tasks
stats = collector.get_aggregate_stats()
print(f"Success rate: {stats.success_rate:.1%}")
print(f"Average iterations: {stats.average_iterations_per_task:.1f}")
```

---

## Type Hints

All public APIs include full type hints:

```python
from typing import Dict, Any, Optional, List

async def execute_task(
    self,
    task: Task
) -> Dict[str, Any]:
    ...
```

Use type checkers like `mypy` for validation:

```bash
mypy your_code.py
```

---

## Async/Await Best Practices

```python
import asyncio

# Good - run in asyncio event loop
async def main():
    async with MLEStarClient.from_env() as client:
        result = await client.execute_task(task)
    return result

# Run
result = asyncio.run(main())

# Good - multiple tasks concurrently
async def run_multiple_tasks(tasks: List[Task]):
    async with MLEStarClient.from_env() as client:
        results = await asyncio.gather(*[
            client.execute_task(task)
            for task in tasks
        ])
    return results
```

---

## Further Reading

- [Architecture Documentation](architecture.md)
- [Security Guide](security.md)
- [Examples](../examples/)
- [Test Documentation](../tests/README.md)
