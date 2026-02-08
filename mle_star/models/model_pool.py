"""Model pool for managing multiple local models with GPU memory constraints."""

from typing import Dict, Optional, Any
import torch
import logging

from .local_model import LocalModel


class ModelPool:
    """
    Manages multiple local models with GPU memory pooling.

    This class provides:
    - Lazy loading of models (load on first use)
    - Memory-aware model management
    - Automatic unloading when memory constrained
    - LRU-based model eviction

    Example:
        >>> pool = ModelPool(max_gpu_memory_gb=28.0)
        >>> pool.register_model("executor", {
        ...     "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
        ...     "load_in_4bit": True,
        ...     "estimated_memory_gb": 10.0
        ... })
        >>> executor_model = pool.get_model("executor")
    """

    def __init__(self, max_gpu_memory_gb: float = 28.0):
        """
        Initialize model pool.

        Args:
            max_gpu_memory_gb: Maximum GPU memory to use across all models
        """
        self.models: Dict[str, Dict[str, Any]] = {}
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.loaded_models: list[str] = []  # LRU list (most recent at end)
        self.logger = logging.getLogger("mle_star.model_pool")

        self.logger.info(f"Model pool initialized with {max_gpu_memory_gb}GB limit")

    def register_model(self, name: str, model_config: Dict[str, Any]) -> None:
        """
        Register a model (doesn't load it yet).

        Args:
            name: Unique name for the model (e.g., "executor", "verifier")
            model_config: Model configuration dictionary with keys:
                - model_name: HuggingFace model identifier
                - load_in_4bit: Whether to use 4-bit quantization
                - estimated_memory_gb: Estimated memory usage in GB
                - device_map: Optional device mapping
                - Other LocalModel parameters

        Example:
            >>> pool.register_model("executor", {
            ...     "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
            ...     "load_in_4bit": True,
            ...     "estimated_memory_gb": 10.0
            ... })
        """
        if name in self.models:
            self.logger.warning(f"Model '{name}' already registered, overwriting")

        self.models[name] = {
            'config': model_config,
            'instance': None,
            'memory_usage_gb': model_config.get('estimated_memory_gb', 10.0)
        }

        self.logger.info(
            f"Registered model '{name}': "
            f"{model_config.get('model_name', 'unknown')} "
            f"(est. {model_config.get('estimated_memory_gb', 10.0):.1f}GB)"
        )

    def get_model(self, name: str) -> LocalModel:
        """
        Get or load a model.

        If the model is already loaded, returns it immediately.
        Otherwise, checks memory availability and loads the model,
        potentially unloading other models if needed.

        Args:
            name: Model name

        Returns:
            LocalModel instance

        Raises:
            KeyError: If model not registered
            RuntimeError: If model cannot be loaded due to memory constraints

        Example:
            >>> model = pool.get_model("executor")
            >>> response = await model.generate("Hello")
        """
        if name not in self.models:
            raise KeyError(
                f"Model '{name}' not registered. "
                f"Available models: {list(self.models.keys())}"
            )

        model_info = self.models[name]

        # If already loaded, update LRU and return
        if model_info['instance'] is not None:
            self._update_lru(name)
            self.logger.debug(f"Using cached model '{name}'")
            return model_info['instance']

        # Check if we need to free memory
        required_memory = model_info['memory_usage_gb']
        self._ensure_memory_available(required_memory)

        # Load model
        try:
            self.logger.info(f"Loading model '{name}'...")

            model_info['instance'] = LocalModel(
                model_name=model_info['config']['model_name'],
                device_map=model_info['config'].get('device_map', 'auto'),
                load_in_4bit=model_info['config'].get('load_in_4bit', True),
                trust_remote_code=model_info['config'].get('trust_remote_code', True),
                max_memory=model_info['config'].get('max_memory')
            )

            # Update LRU list
            self.loaded_models.append(name)

            # Get actual memory usage
            if torch.cuda.is_available():
                actual_memory = model_info['instance'].get_memory_usage()
                self.logger.info(
                    f"Model '{name}' loaded: "
                    f"{actual_memory.get('allocated_gb', 0):.2f}GB allocated"
                )

            return model_info['instance']

        except Exception as e:
            self.logger.error(f"Failed to load model '{name}': {e}")
            raise RuntimeError(f"Failed to load model '{name}': {e}")

    def _update_lru(self, name: str) -> None:
        """
        Update LRU list when a model is accessed.

        Args:
            name: Model name
        """
        if name in self.loaded_models:
            self.loaded_models.remove(name)
        self.loaded_models.append(name)

    def _ensure_memory_available(self, required_gb: float) -> None:
        """
        Ensure enough GPU memory is available.

        Unloads least recently used models if necessary.

        Args:
            required_gb: Required memory in GB

        Raises:
            RuntimeError: If cannot free enough memory
        """
        current_usage = self._get_total_memory_usage()
        available = self.max_gpu_memory_gb - current_usage

        self.logger.debug(
            f"Memory check: {current_usage:.2f}GB used, "
            f"{available:.2f}GB available, "
            f"{required_gb:.2f}GB required"
        )

        # If enough memory available, done
        if available >= required_gb:
            return

        # Try to free memory by unloading models
        while available < required_gb and self.loaded_models:
            # Unload least recently used model
            lru_model = self.loaded_models[0]
            self.logger.info(
                f"Freeing memory by unloading '{lru_model}' "
                f"(need {required_gb:.2f}GB)"
            )
            self._unload_model(lru_model)

            current_usage = self._get_total_memory_usage()
            available = self.max_gpu_memory_gb - current_usage

        # Check if we freed enough
        if available < required_gb:
            raise RuntimeError(
                f"Cannot free enough memory. "
                f"Required: {required_gb:.2f}GB, "
                f"Available: {available:.2f}GB, "
                f"Max: {self.max_gpu_memory_gb:.2f}GB"
            )

    def _unload_model(self, name: str) -> None:
        """
        Unload a model from memory.

        Args:
            name: Model name
        """
        if name not in self.models:
            return

        if self.models[name]['instance'] is not None:
            self.logger.info(f"Unloading model '{name}'")

            # Delete model instance
            del self.models[name]['instance']
            self.models[name]['instance'] = None

            # Remove from LRU list
            if name in self.loaded_models:
                self.loaded_models.remove(name)

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.debug(f"Model '{name}' unloaded")

    def unload_all(self) -> None:
        """Unload all models from memory."""
        for name in list(self.loaded_models):
            self._unload_model(name)

        self.logger.info("All models unloaded")

    def _get_total_memory_usage(self) -> float:
        """
        Get total GPU memory usage in GB.

        Returns:
            Total memory usage in GB
        """
        if not torch.cuda.is_available():
            return 0.0

        return torch.cuda.memory_allocated() / 1e9

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with memory stats
        """
        stats = {
            'loaded_models': self.loaded_models.copy(),
            'total_memory_gb': self._get_total_memory_usage(),
            'max_memory_gb': self.max_gpu_memory_gb,
            'available_memory_gb': self.max_gpu_memory_gb - self._get_total_memory_usage(),
            'num_loaded': len(self.loaded_models),
            'num_registered': len(self.models)
        }

        # Add per-device stats if available
        if torch.cuda.is_available():
            stats['device_count'] = torch.cuda.device_count()
            stats['devices'] = []

            for i in range(torch.cuda.device_count()):
                device_stats = {
                    'device_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'allocated_gb': torch.cuda.memory_allocated(i) / 1e9,
                    'reserved_gb': torch.cuda.memory_reserved(i) / 1e9,
                    'total_gb': torch.cuda.get_device_properties(i).total_memory / 1e9
                }
                stats['devices'].append(device_stats)

        return stats

    def is_loaded(self, name: str) -> bool:
        """
        Check if a model is currently loaded.

        Args:
            name: Model name

        Returns:
            True if loaded, False otherwise
        """
        if name not in self.models:
            return False

        return self.models[name]['instance'] is not None

    def list_models(self, loaded_only: bool = False) -> list[str]:
        """
        List model names.

        Args:
            loaded_only: If True, only return loaded models

        Returns:
            List of model names
        """
        if loaded_only:
            return self.loaded_models.copy()

        return list(self.models.keys())

    def __len__(self) -> int:
        """Return number of registered models."""
        return len(self.models)

    def __contains__(self, name: str) -> bool:
        """Check if model is registered."""
        return name in self.models

    def __str__(self) -> str:
        """String representation."""
        return (
            f"ModelPool("
            f"models={len(self.models)}, "
            f"loaded={len(self.loaded_models)}, "
            f"memory={self._get_total_memory_usage():.2f}/{self.max_gpu_memory_gb:.2f}GB"
            f")"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return str(self)
