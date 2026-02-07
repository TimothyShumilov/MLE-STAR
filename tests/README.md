# MLE-STAR Tests

Comprehensive test suite for the MLE-STAR framework.

## Structure

```
tests/
├── conftest.py                    # Shared fixtures and pytest configuration
├── unit/                          # Unit tests (fast, isolated)
│   ├── test_message.py           # Message protocol tests
│   ├── test_security.py          # Sandbox and validator tests
│   └── test_monitoring.py        # Monitoring components tests
├── integration/                   # Integration tests (slower, multiple components)
│   └── test_workflow_integration.py
└── fixtures/                      # Test data and fixtures
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test categories
```bash
# Unit tests only (fast)
pytest -m unit

# Integration tests
pytest -m integration

# Include slow tests (require model loading)
pytest --slow
```

### Run specific test files
```bash
# Test message protocol
pytest tests/unit/test_message.py

# Test security components
pytest tests/unit/test_security.py

# Test monitoring
pytest tests/unit/test_monitoring.py
```

### Run with coverage
```bash
pytest --cov=mle_star --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Run specific test
```bash
pytest tests/unit/test_message.py::TestMessage::test_message_creation
```

## Test Markers

Tests are marked with different categories:

- `@pytest.mark.unit` - Fast unit tests, no external dependencies
- `@pytest.mark.integration` - Integration tests, multiple components
- `@pytest.mark.slow` - Slow tests requiring model loading (skipped by default)

## Writing Tests

### Unit Test Example

```python
import pytest
from mle_star.core.message import Message, MessageType


@pytest.mark.unit
class TestMyComponent:
    """Test MyComponent class."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        component = MyComponent()

        # Act
        result = component.do_something()

        # Assert
        assert result is not None
```

### Integration Test Example

```python
import pytest


@pytest.mark.integration
class TestIntegration:
    """Test component integration."""

    @pytest.mark.asyncio
    async def test_async_workflow(self):
        """Test async workflow."""
        # Test implementation
        pass
```

### Using Fixtures

```python
def test_with_fixtures(sample_message, temp_dir):
    """Test using shared fixtures."""
    # sample_message and temp_dir are available from conftest.py
    assert sample_message.msg_type == MessageType.TASK_REQUEST
    assert temp_dir.exists()
```

## Available Fixtures

See `conftest.py` for all available fixtures:

- `temp_dir` - Temporary directory (auto-cleanup)
- `state_manager` - StateManager instance
- `metrics_collector` - MetricsCollector instance
- `simple_task` - Simple classification task
- `complex_task` - Complex task with subtasks
- `sample_code` - Safe Python code for testing
- `dangerous_code` - Dangerous code for security testing
- And more...

## Test Coverage Goals

Target coverage by component:

- **Core (message, base_agent, state_manager):** 100%
- **Security (sandbox, validator):** 100%
- **Monitoring (metrics, resource_monitor, guardrails):** 90%
- **Agents (planner, executor, verifier):** 80% (excluding model calls)
- **Workflow:** 85%
- **Overall:** >80%

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pytest -m "unit or integration" --cov=mle_star
```

## Troubleshooting

### Tests fail with "No module named 'mle_star'"
Ensure you're running pytest from the project root:
```bash
cd /path/to/project
pytest
```

### Async tests fail
Make sure pytest-asyncio is installed:
```bash
pip install pytest-asyncio
```

### Slow tests timeout
Increase timeout in pytest.ini or skip slow tests:
```bash
pytest -m "not slow"
```

## Notes

- **Slow tests:** Tests marked with `@pytest.mark.slow` require loading actual models and are skipped by default. Use `--slow` to run them.
- **Temp files:** All temporary files are automatically cleaned up after tests.
- **Parallelization:** For faster execution, install pytest-xdist and run `pytest -n auto`
