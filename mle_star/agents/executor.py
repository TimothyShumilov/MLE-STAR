"""Executor agent implementation."""

from typing import Dict, Any, Optional
from ..core.base_agent import BaseAgent, AgentConfig, AgentRole
from ..core.message import Message, MessageType
from ..utils.prompt_templates import ExecutorPrompts, PromptFormatter


class ExecutorAgent(BaseAgent):
    """
    Executor Agent - Uses Qwen2.5-Coder-32B locally.

    Responsibilities:
    - Generate code implementations from strategies
    - Execute code in sandboxed environment
    - Handle errors and retries
    - Report execution results

    Example:
        >>> executor = ExecutorAgent(config, local_model, sandbox)
        >>> response = await executor.handle_message(execution_request)
    """

    def __init__(self, config: AgentConfig, model_interface, sandbox=None):
        """
        Initialize Executor agent.

        Args:
            config: Agent configuration
            model_interface: Local model (Qwen2.5-Coder-32B)
            sandbox: CodeSandbox instance (optional, for testing)
        """
        super().__init__(config, model_interface)
        self.sandbox = sandbox
        self.prompts = ExecutorPrompts()
        self.formatter = PromptFormatter()

        self.logger.info("Executor agent initialized")

    async def process(self, message: Message) -> Message:
        """
        Process execution request.

        Args:
            message: Execution request message

        Returns:
            Execution result message

        Raises:
            ValueError: If message type not supported
        """
        if message.msg_type != MessageType.EXECUTION_REQUEST:
            raise ValueError(
                f"Executor only supports EXECUTION_REQUEST, got {message.msg_type}"
            )

        strategy = message.content['strategy']

        self.logger.info(f"Executing strategy: {strategy.get('name', 'Unknown')}")

        # Generate code
        code = await self._generate_code(strategy)

        # Execute in sandbox
        execution_result = await self._execute_code(code, strategy)

        return message.create_reply(
            msg_type=MessageType.EXECUTION_RESULT,
            content={
                'code': code,
                'result': execution_result,
                'strategy_id': strategy.get('id')
            }
        )

    async def _generate_code(self, strategy: Dict[str, Any]) -> str:
        """
        Generate code implementation from strategy.

        Args:
            strategy: Strategy dictionary with subtasks

        Returns:
            Generated Python code

        Raises:
            RuntimeError: If code generation fails
        """
        self.logger.debug("Generating code for strategy")

        # Build prompt
        prompt = self.prompts.code_generation(
            subtasks=strategy.get('subtasks', []),
            requirements=strategy.get('requirements', []),
            context=strategy.get('context', {})
        )

        try:
            # Generate code
            response = await self._generate_response(
                prompt,
                temperature=0.2,  # Lower for deterministic code
                max_tokens=4000
            )

            # Extract code from response
            code = self._extract_code(response)

            if not code:
                raise ValueError("No code generated")

            self.logger.debug(f"Generated {len(code)} characters of code")

            return code

        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            raise RuntimeError(f"Failed to generate code: {e}")

    async def _execute_code(
        self,
        code: str,
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute code in sandbox.

        Args:
            code: Python code to execute
            strategy: Strategy dict with execution parameters

        Returns:
            Execution result dictionary
        """
        if self.sandbox is None:
            # If no sandbox, return mock success
            self.logger.warning("No sandbox configured, returning mock result")
            return {
                'status': 'success',
                'output': 'Sandbox not configured - code not executed',
                'error': '',
                'execution_time': 0.0
            }

        try:
            self.logger.debug("Executing code in sandbox")

            result = await self.sandbox.execute(
                code,
                timeout=strategy.get('timeout', 300),
                memory_limit=strategy.get('memory_limit', '4G')
            )

            return {
                'status': 'success' if result.status == 'success' else 'error',
                'output': result.stdout,
                'error': result.stderr,
                'execution_time': result.execution_time,
                'exit_code': result.exit_code
            }

        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            return {
                'status': 'error',
                'output': '',
                'error': str(e),
                'execution_time': 0.0,
                'exit_code': -1
            }

    def _extract_code(self, response: str) -> Optional[str]:
        """
        Extract code from LLM response.

        Args:
            response: LLM response text

        Returns:
            Extracted code or None
        """
        code = self.formatter.extract_code_from_response(response, 'python')

        if code is None:
            # Try without language specifier
            code = self.formatter.extract_code_from_response(response)

        if code is None:
            # Fallback: use entire response
            self.logger.warning("Could not extract code block, using entire response")
            code = response

        return code

    async def fix_code(
        self,
        code: str,
        error_message: str,
        strategy: Dict[str, Any]
    ) -> str:
        """
        Attempt to fix code that produced an error.

        Args:
            code: Original code
            error_message: Error message
            strategy: Strategy context

        Returns:
            Fixed code

        Raises:
            RuntimeError: If fixing fails
        """
        self.logger.info("Attempting to fix code")

        prompt = self.prompts.code_fixing(
            original_code=code,
            error_message=error_message,
            context={'strategy': strategy}
        )

        try:
            response = await self._generate_response(
                prompt,
                temperature=0.2,
                max_tokens=4000
            )

            fixed_code = self._extract_code(response)

            if not fixed_code:
                raise ValueError("No fixed code generated")

            self.logger.debug("Code fix generated")
            return fixed_code

        except Exception as e:
            self.logger.error(f"Code fixing failed: {e}")
            raise RuntimeError(f"Failed to fix code: {e}")

    def validate_input(self, message: Message) -> bool:
        """
        Validate input message.

        Args:
            message: Message to validate

        Returns:
            True if valid, False otherwise
        """
        if message.msg_type != MessageType.EXECUTION_REQUEST:
            return False

        return 'strategy' in message.content

    def validate_output(self, response: Message) -> bool:
        """
        Validate output message.

        Args:
            response: Response message

        Returns:
            True if valid, False otherwise
        """
        if response.msg_type == MessageType.EXECUTION_RESULT:
            required = ['code', 'result']
            return all(k in response.content for k in required)

        return response.msg_type == MessageType.ERROR
