"""Planner agent implementation."""

from typing import Dict, Any, List
from ..core.base_agent import BaseAgent, AgentConfig, AgentRole
from ..core.message import Message, MessageType
from ..utils.prompt_templates import PlannerPrompts, PromptFormatter


class PlannerAgent(BaseAgent):
    """
    Planner Agent - Uses Llama 3.3 70B via OpenRouter.

    Responsibilities:
    - Decompose ML tasks into actionable subtasks
    - Generate multiple solution strategies (exploration)
    - Create execution plans
    - Provide refinement guidance based on feedback

    Example:
        >>> planner = PlannerAgent(config, openrouter_client)
        >>> response = await planner.handle_message(task_request)
    """

    def __init__(self, config: AgentConfig, model_interface):
        """
        Initialize Planner agent.

        Args:
            config: Agent configuration
            model_interface: OpenRouter client for Llama 3.3 70B
        """
        super().__init__(config, model_interface)
        self.prompts = PlannerPrompts()
        self.formatter = PromptFormatter()

        self.logger.info("Planner agent initialized")

    async def process(self, message: Message) -> Message:
        """
        Process incoming message.

        Args:
            message: Incoming message

        Returns:
            Response message

        Raises:
            ValueError: If message type not supported
        """
        if message.msg_type == MessageType.TASK_REQUEST:
            return await self._decompose_task(message)
        elif message.msg_type == MessageType.FEEDBACK:
            return await self._refine_strategy(message)
        else:
            raise ValueError(
                f"Planner does not support message type: {message.msg_type}"
            )

    async def _decompose_task(self, message: Message) -> Message:
        """
        Decompose task into multiple strategies.

        Args:
            message: Task request message

        Returns:
            Task decomposition message with strategies
        """
        task = message.content['task']
        iteration = message.content.get('iteration', 0)
        num_strategies = message.content.get('num_strategies', 3)

        self.logger.info(
            f"Decomposing task (iteration {iteration}): "
            f"{task.get('description', 'N/A')[:50]}..."
        )

        # Build prompt
        prompt = self.prompts.task_decomposition(
            task_description=task.get('description', ''),
            task_type=task.get('type', 'custom'),
            constraints=task.get('constraints', {}),
            iteration=iteration,
            num_strategies=num_strategies
        )

        # Generate strategies
        try:
            response = await self._generate_response(
                prompt,
                temperature=0.8,  # Higher for creativity
                max_tokens=3000
            )

            # Parse strategies from response
            strategies = self._parse_strategies(response)

            if not strategies:
                raise ValueError("No strategies generated")

            self.logger.info(f"Generated {len(strategies)} strategies")

            return message.create_reply(
                msg_type=MessageType.TASK_DECOMPOSITION,
                content={'strategies': strategies},
                metadata={
                    'iteration': iteration,
                    'num_strategies': len(strategies)
                }
            )

        except Exception as e:
            self.logger.error(f"Task decomposition failed: {e}")
            raise RuntimeError(f"Failed to decompose task: {e}")

    async def _refine_strategy(self, message: Message) -> Message:
        """
        Refine strategies based on feedback.

        Args:
            message: Feedback message

        Returns:
            Refined strategies message
        """
        results = message.content.get('results', [])
        task = message.content.get('task', {})

        self.logger.info("Refining strategies based on feedback")

        # Build refinement prompt
        prompt = self.prompts.strategy_refinement(
            task_description=task.get('description', ''),
            previous_results=results,
            feedback=message.content
        )

        # Generate refined strategies
        try:
            response = await self._generate_response(
                prompt,
                temperature=0.7,  # Slightly lower for focused refinement
                max_tokens=3000
            )

            strategies = self._parse_strategies(response)

            if not strategies:
                raise ValueError("No refined strategies generated")

            self.logger.info(f"Generated {len(strategies)} refined strategies")

            return message.create_reply(
                msg_type=MessageType.TASK_DECOMPOSITION,
                content={'strategies': strategies},
                metadata={'refined': True}
            )

        except Exception as e:
            self.logger.error(f"Strategy refinement failed: {e}")
            raise RuntimeError(f"Failed to refine strategies: {e}")

    def _parse_strategies(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse strategies from LLM response.

        Args:
            response: LLM response text

        Returns:
            List of strategy dictionaries

        Raises:
            ValueError: If parsing fails
        """
        # Try to extract JSON from response
        strategies = self.formatter.extract_json_from_response(response)

        if strategies is None:
            self.logger.warning("Could not extract JSON from response")
            # Fallback: create a single strategy from the text
            strategies = [{
                'id': 'strategy_fallback',
                'name': 'Generated Strategy',
                'approach': response[:500],
                'subtasks': [
                    {
                        'id': 'subtask_1',
                        'description': 'Implement solution based on generated approach',
                        'dependencies': [],
                        'estimated_time': 300
                    }
                ],
                'requirements': ['pandas', 'numpy', 'scikit-learn'],
                'complexity': 'medium',
                'success_criteria': ['Code executes without errors']
            }]

        # Ensure strategies is a list
        if not isinstance(strategies, list):
            strategies = [strategies]

        # Validate and enrich strategies
        for i, strategy in enumerate(strategies):
            # Ensure required fields
            strategy.setdefault('id', f'strategy_{i+1}')
            strategy.setdefault('name', f'Strategy {i+1}')
            strategy.setdefault('approach', 'No description')
            strategy.setdefault('subtasks', [])
            strategy.setdefault('requirements', [])
            strategy.setdefault('complexity', 'medium')
            strategy.setdefault('success_criteria', [])

            # Validate subtasks
            for j, subtask in enumerate(strategy['subtasks']):
                subtask.setdefault('id', f"subtask_{j+1}")
                subtask.setdefault('description', 'No description')
                subtask.setdefault('dependencies', [])
                subtask.setdefault('estimated_time', 60)

        return strategies

    def validate_input(self, message: Message) -> bool:
        """
        Validate input message.

        Args:
            message: Message to validate

        Returns:
            True if valid, False otherwise
        """
        if message.msg_type == MessageType.TASK_REQUEST:
            return 'task' in message.content

        elif message.msg_type == MessageType.FEEDBACK:
            return 'task' in message.content or 'results' in message.content

        return False

    def validate_output(self, response: Message) -> bool:
        """
        Validate output message.

        Args:
            response: Response message

        Returns:
            True if valid, False otherwise
        """
        if response.msg_type == MessageType.TASK_DECOMPOSITION:
            if 'strategies' not in response.content:
                return False

            strategies = response.content['strategies']
            if not isinstance(strategies, list) or len(strategies) == 0:
                return False

            return True

        return response.msg_type == MessageType.ERROR
