"""Base model interface for MLE-STAR framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging


class BaseModel(ABC):
    """
    Abstract base class for all model interfaces.

    This class defines the common interface that all model implementations
    (OpenRouter API, local models, etc.) must implement.
    """

    def __init__(self):
        """Initialize the model."""
        self.logger = logging.getLogger(f"mle_star.{self.__class__.__name__}")

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion from prompt.

        Args:
            prompt: Input prompt string
            **kwargs: Model-specific generation parameters
                - temperature: Sampling temperature (0.0 - 1.0)
                - max_tokens: Maximum tokens to generate
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - stop: Stop sequences

        Returns:
            Generated text string

        Raises:
            RuntimeError: If generation fails

        Example:
            >>> model = SomeModel()
            >>> response = await model.generate(
            ...     "Write a function to calculate fibonacci",
            ...     temperature=0.7,
            ...     max_tokens=1000
            ... )
        """
        pass

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics (if applicable).

        Returns:
            Dictionary with memory statistics (in GB).
            Empty dict if not applicable.

        Example:
            >>> usage = model.get_memory_usage()
            >>> print(f"Allocated: {usage.get('allocated_gb', 0):.2f} GB")
        """
        return {}

    def clear_cache(self) -> None:
        """
        Clear model cache (if applicable).

        This can help free up memory between operations.
        """
        pass

    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}()"
