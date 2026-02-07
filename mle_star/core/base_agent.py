"""Base agent class for MLE-STAR framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from .message import Message, MessageType, MessageValidator


class AgentRole(Enum):
    """Roles for different agent types in the MLE-STAR framework."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    WORKFLOW = "workflow"


@dataclass
class AgentConfig:
    """
    Configuration for an agent.

    Attributes:
        role: Agent's role in the system
        model_config: Model-specific configuration (model name, temperature, etc.)
        max_retries: Maximum number of retries for failed operations
        timeout: Operation timeout in seconds
        temperature: Temperature for model generation (0.0 - 1.0)
        max_tokens: Maximum tokens for model generation
    """

    role: AgentRole
    model_config: Dict[str, Any]
    max_retries: int = 3
    timeout: int = 300
    temperature: float = 0.7
    max_tokens: int = 2000


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the MLE-STAR framework.

    This class defines the common interface and functionality that all agents
    (Planner, Executor, Verifier) must implement. It handles:
    - Message processing and validation
    - Interaction history tracking
    - Error handling and retries
    - Model interaction abstraction

    Subclasses must implement:
    - process(): Main message processing logic
    - validate_input(): Input message validation
    - validate_output(): Output message validation
    """

    def __init__(self, config: AgentConfig, model_interface):
        """
        Initialize the agent.

        Args:
            config: Agent configuration
            model_interface: Model interface for generation (BaseModel instance)
        """
        self.config = config
        self.model = model_interface
        self.history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(f"mle_star.{config.role.value}")

        # Statistics
        self.stats = {
            'messages_processed': 0,
            'messages_failed': 0,
            'total_tokens': 0,
            'total_time': 0.0
        }

    @abstractmethod
    async def process(self, message: Message) -> Message:
        """
        Process an incoming message and return a response.

        This is the main entry point for agent logic. Subclasses must
        implement their specific message processing logic here.

        Args:
            message: Incoming message to process

        Returns:
            Response message

        Raises:
            ValueError: If message is invalid
            RuntimeError: If processing fails
        """
        pass

    @abstractmethod
    def validate_input(self, message: Message) -> bool:
        """
        Validate an incoming message.

        Subclasses should implement their specific validation logic here,
        checking for required fields and valid content structure.

        Args:
            message: Message to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def validate_output(self, response: Message) -> bool:
        """
        Validate an outgoing message.

        Subclasses should implement their specific validation logic here,
        ensuring the response meets expected format requirements.

        Args:
            response: Response message to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    async def _generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using the model.

        This method provides a uniform interface for model interaction,
        handling retries and error logging.

        Args:
            prompt: Input prompt for the model
            **kwargs: Additional arguments for model generation
                (temperature, max_tokens, etc.)

        Returns:
            Generated text response

        Raises:
            RuntimeError: If generation fails after max retries
        """
        # Merge default config with kwargs
        gen_kwargs = {
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            **kwargs
        }

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(
                    f"Generating response (attempt {attempt + 1}/{self.config.max_retries})"
                )
                response = await self.model.generate(prompt, **gen_kwargs)
                return response
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Generation failed on attempt {attempt + 1}: {e}"
                )

        # All retries failed
        self.logger.error(f"Generation failed after {self.config.max_retries} attempts")
        raise RuntimeError(
            f"Model generation failed after {self.config.max_retries} retries: {last_error}"
        )

    def add_to_history(self, message: Message) -> None:
        """
        Add a message to the agent's history.

        Args:
            message: Message to add to history
        """
        self.history.append({
            'timestamp': message.timestamp.isoformat(),
            'msg_id': message.msg_id,
            'msg_type': message.msg_type.value,
            'sender': message.sender,
            'receiver': message.receiver,
            'content_summary': self._summarize_content(message.content)
        })

        # Keep only last 100 messages to prevent memory issues
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def _summarize_content(self, content: Dict[str, Any], max_len: int = 200) -> str:
        """
        Create a summary of message content for logging.

        Args:
            content: Message content dictionary
            max_len: Maximum length of summary

        Returns:
            Truncated string representation of content
        """
        content_str = str(content)
        if len(content_str) > max_len:
            return content_str[:max_len] + "..."
        return content_str

    def update_stats(self, **kwargs) -> None:
        """
        Update agent statistics.

        Args:
            **kwargs: Stat name-value pairs to update
        """
        for key, value in kwargs.items():
            if key in self.stats:
                if isinstance(value, (int, float)):
                    self.stats[key] += value
                else:
                    self.stats[key] = value

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            'role': self.config.role.value,
            'history_size': len(self.history)
        }

    async def handle_message(self, message: Message) -> Message:
        """
        Handle an incoming message with full validation and error handling.

        This is a wrapper around process() that provides:
        - Input validation
        - Error handling
        - Output validation
        - History tracking
        - Statistics updates

        Args:
            message: Incoming message

        Returns:
            Response message

        Raises:
            ValueError: If message validation fails
            RuntimeError: If processing fails
        """
        import time

        start_time = time.time()

        # Validate message structure
        is_valid, error = MessageValidator.validate(message)
        if not is_valid:
            raise ValueError(f"Invalid message structure: {error}")

        # Validate message content
        if not self.validate_input(message):
            raise ValueError(
                f"Invalid input message for {self.config.role.value}: "
                f"content does not match expected format"
            )

        # Add to history
        self.add_to_history(message)

        try:
            # Process message
            response = await self.process(message)

            # Validate response
            if not self.validate_output(response):
                raise ValueError(
                    f"Invalid output message from {self.config.role.value}: "
                    f"response does not match expected format"
                )

            # Add response to history
            self.add_to_history(response)

            # Update statistics
            processing_time = time.time() - start_time
            self.update_stats(
                messages_processed=1,
                total_time=processing_time
            )

            self.logger.info(
                f"Processed message {message.msg_id[:8]}... "
                f"in {processing_time:.2f}s"
            )

            return response

        except Exception as e:
            # Update failure statistics
            self.update_stats(messages_failed=1)
            self.logger.error(
                f"Failed to process message {message.msg_id[:8]}...: {e}"
            )

            # Create error response
            error_response = message.create_reply(
                msg_type=MessageType.ERROR,
                content={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'agent_role': self.config.role.value
                },
                metadata={'processing_time': time.time() - start_time}
            )

            return error_response

    def reset_history(self) -> None:
        """Clear the agent's message history."""
        self.history.clear()
        self.logger.info(f"History cleared for {self.config.role.value} agent")

    def reset_stats(self) -> None:
        """Reset agent statistics to initial values."""
        self.stats = {
            'messages_processed': 0,
            'messages_failed': 0,
            'total_tokens': 0,
            'total_time': 0.0
        }
        self.logger.info(f"Statistics reset for {self.config.role.value} agent")

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(role={self.config.role.value})"

    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (
            f"{self.__class__.__name__}("
            f"role={self.config.role.value}, "
            f"model={self.config.model_config.get('model_name', 'unknown')}, "
            f"messages_processed={self.stats['messages_processed']}"
            f")"
        )
