# MLE-STAR Framework Architecture

This document describes the architecture and design of the MLE-STAR (Multi-Agent ML Engineering with Search and Targeted Refinement) framework.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Agent Architecture](#agent-architecture)
5. [STAR Workflow](#star-workflow)
6. [Data Flow](#data-flow)
7. [Security Architecture](#security-architecture)
8. [Monitoring Architecture](#monitoring-architecture)
9. [Design Decisions](#design-decisions)

---

## Overview

MLE-STAR is a multi-agent framework for automating machine learning workflows through iterative refinement. The framework uses three specialized agents working together in a STAR (Search, Test, And Refine) loop to solve ML tasks.

### Key Characteristics

- **Multi-Agent:** Three specialized agents (Planner, Executor, Verifier)
- **Iterative:** STAR workflow with up to 5 refinement iterations
- **Secure:** Sandboxed code execution with validation
- **Monitored:** Comprehensive metrics and resource tracking
- **Hybrid:** API-based and local model deployment

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MLE-STAR Client                          │
│                     (High-Level Public API)                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       STAR Workflow                              │
│              (Orchestrates Agent Collaboration)                  │
└───┬──────────────┬──────────────┬──────────────┬────────────────┘
    │              │              │              │
    ▼              ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐
│ Planner │  │ Executor │  │ Verifier │  │ State Manager  │
│ Agent   │  │  Agent   │  │  Agent   │  │                │
└────┬────┘  └────┬─────┘  └────┬─────┘  └────────────────┘
     │            │             │
     ▼            ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Llama   │  │  Qwen   │  │  Qwen   │
│ 3.3 70B │  │ 2.5-32B │  │ 2.5-14B │
│(OpenRtr)│  │ (Local) │  │ (Local) │
└─────────┘  └─────────┘  └─────────┘
     │            │             │
     └────────────┴─────────────┘
                  │
                  ▼
        ┌──────────────────┐
        │  Model Pool      │
        │ (GPU Management) │
        └──────────────────┘
```

### Layer Architecture

**Layer 1: Public API**
- `MLEStarClient` - High-level interface
- Simple async/await API
- Factory methods for easy initialization

**Layer 2: Orchestration**
- `STARWorkflow` - Coordinates agents
- `StateManager` - Persists task state
- `MessageHistory` - Tracks communication

**Layer 3: Agents**
- `PlannerAgent` - Strategy generation
- `ExecutorAgent` - Code implementation
- `VerifierAgent` - Result validation

**Layer 4: Models**
- `OpenRouterClient` - API-based models
- `LocalModel` - HuggingFace models
- `ModelPool` - Resource management

**Layer 5: Execution & Security**
- `CodeSandbox` - Isolated execution
- `CodeValidator` - Security checks

**Layer 6: Monitoring**
- `MetricsCollector` - Performance tracking
- `ResourceMonitor` - GPU/CPU/RAM monitoring
- `GuardrailsManager` - Safety checks

---

## Core Components

### 1. Message Protocol

**File:** `mle_star/core/message.py`

All inter-agent communication uses structured messages:

```python
@dataclass
class Message:
    msg_type: MessageType
    sender: str
    receiver: str
    content: Dict[str, Any]
    msg_id: str  # UUID
    parent_msg_id: Optional[str]  # For threading
    timestamp: datetime
```

**Message Types:**
- `TASK_REQUEST` - Initial task submission
- `TASK_DECOMPOSITION` - Strategy generation
- `EXECUTION_REQUEST` - Code generation request
- `EXECUTION_RESULT` - Code execution output
- `VERIFICATION_REQUEST` - Result verification
- `VERIFICATION_RESULT` - Verification outcome
- `ERROR` - Error notifications
- `RESULT` - Final result

**Threading:** Messages form conversation threads via `parent_msg_id`, enabling feedback loops.

### 2. Base Agent

**File:** `mle_star/core/base_agent.py`

Abstract base class for all agents:

```python
class BaseAgent(ABC):
    @abstractmethod
    async def process(self, message: Message) -> Message:
        """Process incoming message and return response."""
        pass

    @abstractmethod
    def validate_input(self, message: Message) -> bool:
        """Validate message before processing."""
        pass
```

**Common Features:**
- Message validation
- Retry logic with exponential backoff
- Statistics tracking
- Error handling
- Conversation history

### 3. State Manager

**File:** `mle_star/core/state_manager.py`

Manages task state persistence:

```python
class StateManager:
    def create_task(self, task: Task) -> str
    def update_iteration(self, task_id: str, data: Dict)
    def complete_task(self, task_id: str, result: Dict)
    def load_task(self, task_id: str) -> Dict
```

**State Structure:**
```json
{
  "task_id": "uuid",
  "task": { ... },
  "status": "active|completed|failed",
  "iterations": [
    {
      "iteration": 0,
      "strategies": [...],
      "results": [...]
    }
  ],
  "final_result": { ... }
}
```

**Persistence:** JSON files in `state_dir/`

---

## Agent Architecture

### Planner Agent

**File:** `mle_star/agents/planner.py`
**Model:** Llama 3.3 70B (OpenRouter API)
**Role:** Strategy generation and task decomposition

**Process:**
1. Receives `TASK_REQUEST` with task description
2. Analyzes task requirements
3. Generates 3 alternative strategies
4. Returns `TASK_DECOMPOSITION` with strategies

**Strategy Structure:**
```python
{
  "name": "Strategy name",
  "approach": "High-level approach",
  "subtasks": [
    {"description": "...", "order": 1},
    ...
  ],
  "expected_outcome": "What to expect"
}
```

**Temperature:** 0.8 (creative exploration)

### Executor Agent

**File:** `mle_star/agents/executor.py`
**Model:** Qwen2.5-Coder-32B (Local, 4-bit quantized)
**Role:** Code generation and execution

**Process:**
1. Receives `EXECUTION_REQUEST` with strategy
2. Generates Python code
3. Validates code with `CodeValidator`
4. Executes in `CodeSandbox`
5. Returns `EXECUTION_RESULT`

**Code Generation:**
- Temperature: 0.2 (deterministic)
- Max tokens: 4000
- Extracts code from markdown blocks

**Execution:**
- Timeout: 300s default
- Memory limit: 4GB
- Isolated filesystem

**Temperature:** 0.2 (deterministic code)

### Verifier Agent

**File:** `mle_star/agents/verifier.py`
**Model:** Qwen2.5-Coder-14B (Local, 4-bit quantized)
**Role:** Result verification and quality assessment

**Process:**
1. Receives `VERIFICATION_REQUEST` with execution result
2. Analyzes code quality
3. Checks success criteria
4. Computes objective metrics
5. Returns `VERIFICATION_RESULT` with score

**Scoring:**
```python
score = base_score
  + (criteria_met_bonus * criteria_weight)
  - (issues_penalty * issues_weight)
  + (metrics_bonus * metrics_weight)

# Score range: 0.0 - 1.0
```

**Temperature:** 0.1 (consistent evaluation)

---

## STAR Workflow

**File:** `mle_star/core/workflow.py`

Three-phase iterative workflow:

### Phase 1: Search

**Objective:** Generate multiple solution strategies

```python
strategies = await planner.generate_strategies(
    task=task,
    iteration=i,
    num_strategies=3
)
```

**Output:** List of 3 alternative approaches

### Phase 2: Evaluation

**Objective:** Execute and verify each strategy

```python
for strategy in strategies:
    # Execute
    code = await executor.generate_code(strategy)
    result = await executor.execute(code)

    # Verify
    verification = await verifier.verify(result, strategy)

    results.append({
        'strategy': strategy,
        'execution': result,
        'verification': verification
    })
```

**Output:** List of evaluated results with scores

### Phase 3: Refinement

**Objective:** Decide next action

```python
best_result = max(results, key=lambda r: r['verification']['score'])

if best_result['verification']['score'] >= 0.8:
    return {'action': 'complete', 'result': best_result}
elif best_result['verification']['score'] >= 0.5:
    return {'action': 'refine', 'feedback': ...}
else:
    return {'action': 'refine', 'major_refinement': True}
```

**Decisions:**
- **Complete:** Score ≥ 0.8 → Task done
- **Refine:** 0.5 ≤ Score < 0.8 → Continue iteration
- **Major Refine:** Score < 0.5 → Significant changes needed

**Max Iterations:** 5

---

## Data Flow

### End-to-End Task Execution

```
User → Client → Workflow
                   │
                   ├─ Iteration 1
                   │   ├─ Planner: Generate strategies
                   │   ├─ For each strategy:
                   │   │   ├─ Executor: Generate & execute code
                   │   │   └─ Verifier: Verify results
                   │   └─ Refinement: Best score < 0.8
                   │
                   ├─ Iteration 2
                   │   └─ (same pattern)
                   │
                   └─ Iteration N
                       └─ Best score ≥ 0.8 → Complete
                           └─ Return result to user
```

### Message Flow Example

```
Workflow → Planner: TASK_REQUEST
           {task: {...}, iteration: 0}

Planner → Workflow: TASK_DECOMPOSITION
          {strategies: [{...}, {...}, {...}]}

Workflow → Executor: EXECUTION_REQUEST
           {strategy: {...}}

Executor → Workflow: EXECUTION_RESULT
           {code: "...", result: {...}, status: "success"}

Workflow → Verifier: VERIFICATION_REQUEST
           {strategy: {...}, result: {...}}

Verifier → Workflow: VERIFICATION_RESULT
           {score: 0.85, feedback: {...}, status: "success"}
```

---

## Security Architecture

### Multi-Layer Security

**Layer 1: Input Validation**
- Task description sanitization
- File path validation
- Configuration validation

**Layer 2: Code Validation (AST)**
```python
# Pre-execution validation
validator = CodeValidator()
result = validator.validate_code(code)

if not result['valid']:
    raise SecurityError(result['issues'])
```

**Checks:**
- Forbidden imports (subprocess, eval, socket, etc.)
- Dangerous function calls
- Syntax errors
- Cyclomatic complexity

**Layer 3: Sandbox Execution**
```python
sandbox = CodeSandbox(
    max_execution_time=300,
    max_memory_mb=4096
)

result = await sandbox.execute(code)
```

**Isolation:**
- Temporary isolated filesystem
- Resource limits (CPU, memory, time)
- Subprocess isolation
- Optional: Docker containers

**Layer 4: Runtime Limits**

On Unix/Linux/Mac:
```python
import resource

resource.setrlimit(resource.RLIMIT_AS, (4GB, 4GB))
resource.setrlimit(resource.RLIMIT_CPU, (300s, 300s))
resource.setrlimit(resource.RLIMIT_NPROC, (100, 100))
```

**Layer 5: Output Validation**
- Execution result sanitization
- Error message filtering
- Result size limits

### Security Limitations

**Current Implementation:**
- Basic subprocess isolation (production: use Docker)
- No network restrictions (can be added)
- Windows: limited resource controls

**Production Recommendations:**
- Deploy in Docker containers
- Add network restrictions
- Implement audit logging
- Regular security reviews

---

## Monitoring Architecture

### Metrics Collection

**File:** `mle_star/monitoring/metrics.py`

**Per-Task Metrics:**
```python
@dataclass
class TaskMetrics:
    iterations_count: int
    strategies_generated: int
    executions_successful: int
    best_score: float
    total_duration: float
    api_calls: int
    peak_gpu_memory_mb: float
    # ... and more
```

**Aggregate Statistics:**
```python
@dataclass
class AggregateMetrics:
    total_tasks: int
    successful_tasks: int
    success_rate: float
    average_iterations_per_task: float
    # ... and more
```

**Storage:** JSON files per task + aggregate stats

### Resource Monitoring

**File:** `mle_star/monitoring/resource_monitor.py`

**Real-Time Monitoring:**
```python
monitor = ResourceMonitor()
monitor.start(interval=10.0)  # Every 10 seconds

snapshot = monitor.get_current_snapshot()
# → GPU memory, CPU %, RAM %, disk usage
```

**Peak Tracking:**
- Peak GPU memory
- Peak CPU usage
- Peak RAM usage

**Alerts:** Threshold-based callbacks

### Guardrails

**File:** `mle_star/monitoring/guardrails.py`

**Input Guardrails:**
- Dangerous pattern detection
- Suspicious keyword alerts
- Path traversal prevention

**Output Guardrails:**
- Code safety validation
- Import whitelist/blacklist
- Code length limits

**Rate Limiting:**
- API calls: 50/day (free tier)
- Task starts: 10/hour

**Budget Tracking:**
- Cost estimation
- Daily budget limits
- Quota management

---

## Design Decisions

### 1. Hybrid Model Deployment

**Decision:** API-based Planner + Local Executor/Verifier

**Rationale:**
- Planner needs strong reasoning (70B model)
- Executor/Verifier can use smaller models locally
- Balances cost and performance
- Fits within VRAM constraints (32GB)

**Alternatives Considered:**
- All API: Too expensive, rate-limited
- All local: 70B doesn't fit in VRAM
- All local with smaller models: Lower quality planning

### 2. Message-Based Communication

**Decision:** Structured JSON messages with UUIDs

**Rationale:**
- Type-safe and traceable
- Enables conversation threading
- Easy to serialize/deserialize
- Supports debugging and auditing

**Alternatives Considered:**
- Direct function calls: Less traceable
- Event bus: More complex, unnecessary
- gRPC: Overkill for local agents

### 3. 4-bit Quantization

**Decision:** Use BitsAndBytes NF4 quantization

**Rationale:**
- Reduces VRAM ~70% (32B: 64GB → ~10GB)
- Minimal quality loss
- Faster loading
- Enables dual 32B+14B on 32GB VRAM

**Trade-offs:**
- Slightly slower inference
- Quantization overhead

### 4. Subprocess Sandbox

**Decision:** Subprocess + resource limits (production: Docker)

**Rationale:**
- Simple to implement
- Works on Unix/Linux/Mac
- Good enough for development/research
- Clear path to Docker for production

**Alternatives Considered:**
- RestrictedPython: Limited capabilities
- VM-based: Too heavyweight
- Docker from start: Over-engineered for MVP

### 5. JSON State Persistence

**Decision:** File-based JSON storage

**Rationale:**
- Simple and human-readable
- No database setup required
- Easy debugging
- Sufficient for research use

**Future Migration Path:**
- SQLite for single-user scaling
- PostgreSQL for multi-user production

### 6. Async/Await API

**Decision:** Fully async with asyncio

**Rationale:**
- Natural for I/O-bound operations (API calls)
- Enables concurrent strategy execution (future)
- Modern Python best practice

**Trade-off:**
- Slightly more complex than sync
- Requires Python 3.7+

### 7. Three Separate Agents

**Decision:** Specialized agents vs. single agent

**Rationale:**
- Separation of concerns
- Different models optimized for each role
- Parallel evolution of capabilities
- Clearer debugging

**Alternative:**
- Single agent with role-switching: Simpler but less specialized

---

## Performance Characteristics

### Timing (Approximate)

| Operation | Duration |
|-----------|----------|
| Model loading (first time) | 30-60s |
| Strategy generation | 10-30s |
| Code generation | 20-40s |
| Code execution | Varies (up to 300s) |
| Result verification | 15-25s |
| **Full iteration** | **2-6 minutes** |
| **Complete task (3-5 iter)** | **10-25 minutes** |

### Resource Usage

| Component | VRAM | RAM |
|-----------|------|-----|
| Executor (Qwen 32B, 4-bit) | ~10GB | ~4GB |
| Verifier (Qwen 14B, 4-bit) | ~4GB | ~2GB |
| **Total** | **~14GB** | **~6GB** |

**Headroom:** 18GB on 2x16GB GPUs

### API Usage

- **Free Tier:** 50 requests/day
- **Planner calls:** ~3 per iteration
- **Estimated capacity:** ~16 tasks/day (conservative)

---

## Extensibility

### Adding New Agents

1. Inherit from `BaseAgent`
2. Implement `process()` and `validate_input()`
3. Register message types
4. Integrate into workflow

### Adding New Models

1. Inherit from `BaseModel`
2. Implement `generate()` method
3. Add to `ModelPool` if local
4. Configure in `configs/models.yaml`

### Custom Task Types

1. Extend `Task` class
2. Add task-specific fields
3. Update agent prompts
4. Implement task-specific validation

### Monitoring Extensions

1. Implement custom `MetricType`
2. Add to `MetricsCollector`
3. Create visualization tools
4. Export to monitoring systems

---

## Future Enhancements

### Short-term
- Docker-based sandbox
- Parallel strategy execution
- SQLite state storage
- Web UI for monitoring

### Long-term
- Multi-task batch processing
- Active learning from failures
- Custom model fine-tuning
- Distributed execution
- Integration with MLOps platforms

---

## References

- **MLE-STAR Paper:** [arXiv:2506.15692](https://arxiv.org/pdf/2506.15692)
- **Llama 3.3 70B:** [OpenRouter](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct:free)
- **Qwen2.5-Coder:** [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
- **BitsAndBytes:** [HuggingFace Blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
