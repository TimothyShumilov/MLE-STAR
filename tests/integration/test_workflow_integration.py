"""Integration tests for STAR workflow."""

import pytest
import asyncio
from pathlib import Path

from mle_star.core.workflow import STARWorkflow
from mle_star.core.state_manager import StateManager
from mle_star.tasks.task import Task, TaskType
from mle_star.monitoring.metrics import MetricsCollector


@pytest.mark.integration
class TestWorkflowIntegration:
    """Integration tests for full workflow."""

    @pytest.fixture
    def workflow_components(self, temp_dir):
        """Create workflow components (without real models)."""
        # Note: This is a mock setup - real tests would need actual models
        # or mocked agents

        state_manager = StateManager(state_dir=temp_dir / "state")
        metrics_collector = MetricsCollector(metrics_dir=temp_dir / "metrics")

        return {
            'state_manager': state_manager,
            'metrics_collector': metrics_collector,
            'temp_dir': temp_dir
        }

    def test_state_manager_task_lifecycle(self, workflow_components):
        """Test task lifecycle in state manager."""
        state_manager = workflow_components['state_manager']

        # Create task
        task = Task(
            description="Test task",
            task_type=TaskType.CLASSIFICATION
        )

        # Create task in state manager
        task_id = state_manager.create_task(task)
        assert task_id is not None

        # Update iteration
        state_manager.update_iteration(task_id, {
            'iteration': 0,
            'strategies_count': 3,
            'results': []
        })

        # Get iterations
        iterations = state_manager.get_iterations(task_id)
        assert len(iterations) == 1

        # Complete task
        result = {'status': 'success', 'score': 0.95}
        state_manager.complete_task(task_id, result)

        # Load task
        loaded_task = state_manager.load_task(task_id)
        assert loaded_task is not None
        assert loaded_task['status'] == 'completed'

    def test_metrics_collector_integration(self, workflow_components):
        """Test metrics collector integration."""
        metrics = workflow_components['metrics_collector']

        # Simulate task execution
        task_id = "integration_test_task"

        metrics.start_task(task_id, "classification")

        # Simulate iterations
        for i in range(3):
            metrics.record_iteration(task_id, strategies=3, best_score=0.7 + i * 0.1)
            metrics.record_agent_call(task_id, "planner")
            metrics.record_agent_call(task_id, "executor")
            metrics.record_agent_call(task_id, "verifier")
            metrics.record_execution(task_id, success=True, score=0.7 + i * 0.1)

        # End task
        metrics.end_task(task_id, "success")

        # Check aggregate stats
        stats = metrics.get_aggregate_stats()
        assert stats.total_tasks == 1
        assert stats.successful_tasks == 1
        assert stats.total_iterations == 3

    def test_task_serialization_roundtrip(self, simple_task, complex_task):
        """Test task serialization and deserialization."""
        # Simple task
        task_dict = simple_task.to_dict()
        restored = Task.from_dict(task_dict)

        assert restored.description == simple_task.description
        assert restored.task_type == simple_task.task_type
        assert restored.success_criteria == simple_task.success_criteria

        # Complex task
        task_dict = complex_task.to_dict()
        restored = Task.from_dict(task_dict)

        assert len(restored.subtasks) == len(complex_task.subtasks)
        assert restored.constraints == complex_task.constraints


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndMock:
    """End-to-end tests with mocked components."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        from mle_star.core.base_agent import BaseAgent, AgentRole
        from mle_star.core.message import Message, MessageType

        class MockAgent(BaseAgent):
            """Mock agent for testing."""

            def __init__(self, role: AgentRole):
                super().__init__(role=role, model_type="mock")

            async def process(self, message: Message) -> Message:
                """Process message (mock implementation)."""
                if message.msg_type == MessageType.TASK_REQUEST:
                    # Return mock strategies
                    return message.create_reply(
                        msg_type=MessageType.TASK_DECOMPOSITION,
                        content={
                            'strategies': [
                                {
                                    'name': 'Mock Strategy',
                                    'approach': 'Mock approach',
                                    'subtasks': []
                                }
                            ]
                        }
                    )

                return message.create_reply(
                    msg_type=MessageType.RESULT,
                    content={'result': 'mock'}
                )

            def validate_input(self, message: Message) -> bool:
                """Validate input."""
                return True

        return MockAgent

    @pytest.mark.asyncio
    async def test_mock_agent_communication(self, mock_agent):
        """Test communication between mock agents."""
        from mle_star.core.message import Message, MessageType, AgentRole

        planner = mock_agent(AgentRole.PLANNER)
        executor = mock_agent(AgentRole.EXECUTOR)

        # Create request
        request = Message(
            msg_type=MessageType.TASK_REQUEST,
            sender="test",
            receiver="planner",
            content={'task': {'description': 'Test'}}
        )

        # Process with planner
        response = await planner.process(request)

        assert response.msg_type == MessageType.TASK_DECOMPOSITION
        assert 'strategies' in response.content


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security components."""

    @pytest.mark.asyncio
    async def test_validator_and_sandbox_integration(self, sample_code):
        """Test code validator and sandbox working together."""
        from mle_star.execution.validator import CodeValidator
        from mle_star.execution.sandbox import CodeSandbox

        validator = CodeValidator()
        sandbox = CodeSandbox(max_execution_time=10)

        # Validate code
        validation = validator.validate_code(sample_code)
        assert validation['valid'] is True

        # Execute validated code
        result = await sandbox.execute(sample_code)
        assert result.status == 'success'

    @pytest.mark.asyncio
    async def test_block_dangerous_code(self, dangerous_code):
        """Test that dangerous code is blocked."""
        from mle_star.execution.validator import CodeValidator
        from mle_star.execution.sandbox import CodeSandbox

        validator = CodeValidator()

        # Validation should fail
        validation = validator.validate_code(dangerous_code)
        assert validation['valid'] is False

        # Even if we execute, sandbox should protect us
        sandbox = CodeSandbox(max_execution_time=5)
        result = await sandbox.execute(dangerous_code)

        # Should error (import subprocess should fail or be caught)
        assert result.status in ['error', 'timeout']


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for monitoring system."""

    def test_full_monitoring_stack(self, temp_dir):
        """Test full monitoring stack together."""
        from mle_star.monitoring.metrics import MetricsCollector
        from mle_star.monitoring.resource_monitor import ResourceMonitor
        from mle_star.monitoring.guardrails import GuardrailsManager

        # Initialize all components
        metrics = MetricsCollector(metrics_dir=temp_dir / "metrics")
        resource_monitor = ResourceMonitor()
        guardrails = GuardrailsManager(max_api_calls_per_day=50)

        # Simulate task
        task_id = "monitor_test"

        # Validate input
        validation = guardrails.validate_task_input(
            "Train a classifier",
            config={}
        )
        assert validation.valid is True

        # Check rate limit
        assert guardrails.check_rate_limit('task_start') is True

        # Start tracking
        metrics.start_task(task_id, "classification")

        # Get resource snapshot
        snapshot = resource_monitor.get_current_snapshot()
        assert snapshot.cpu_percent >= 0

        # Record metrics
        metrics.record_agent_call(task_id, "planner")
        metrics.record_execution(task_id, success=True, score=0.90)

        # End task
        metrics.end_task(task_id, "success")

        # Verify all worked
        stats = metrics.get_aggregate_stats()
        assert stats.total_tasks == 1

        status = guardrails.get_status()
        assert 'rate_limits' in status


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_config_loading(self, temp_dir):
        """Test configuration loading."""
        from mle_star.utils.config import Config
        import yaml

        # Create test config file
        config_data = {
            'planner': {
                'role': 'planner',
                'model_type': 'openrouter',
                'model_config': {
                    'model_id': 'test-model',
                    'temperature': 0.8
                }
            }
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Load config
        config = Config.from_yaml(config_file)

        assert config is not None
        # Config validation would happen here


@pytest.mark.integration
class TestStateManagement:
    """Integration tests for state management."""

    def test_state_persistence(self, state_manager, simple_task):
        """Test that state persists across operations."""
        # Create task
        task_id = state_manager.create_task(simple_task)

        # Update multiple times
        for i in range(3):
            state_manager.update_iteration(task_id, {
                'iteration': i,
                'strategies_count': 3,
                'best_score': 0.8 + i * 0.05
            })

        # Load task
        task_data = state_manager.load_task(task_id)

        assert task_data is not None
        assert len(task_data['iterations']) == 3

        # Complete task
        state_manager.complete_task(task_id, {'status': 'success'})

        # Reload and verify
        task_data = state_manager.load_task(task_id)
        assert task_data['status'] == 'completed'
