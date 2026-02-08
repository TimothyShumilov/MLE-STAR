"""Local model management with 4-bit quantization support."""

import torch
from typing import Dict, Any, Optional
import logging

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base_model import BaseModel


class LocalModel(BaseModel):
    """
    Local model manager with 4-bit quantization via BitsAndBytes.

    This class provides:
    - Efficient 4-bit quantization (NF4) for memory optimization
    - GPU memory tracking
    - Automatic model offloading
    - Optimized inference

    Memory footprint (approximate):
    - 32B model: ~10GB in 4-bit
    - 14B model: ~4GB in 4-bit
    - 7B model: ~2GB in 4-bit

    Example:
        >>> model = LocalModel(
        ...     model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        ...     load_in_4bit=True
        ... )
        >>> response = await model.generate("Write a function...")
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        load_in_4bit: bool = True,
        trust_remote_code: bool = True,
        max_memory: Optional[Dict[int, str]] = None
    ):
        """
        Initialize local model with quantization.

        Args:
            model_name: HuggingFace model identifier
            device_map: Device mapping strategy ('auto', 'balanced', or custom dict)
            load_in_4bit: Whether to use 4-bit quantization
            trust_remote_code: Whether to trust remote code in model repo
            max_memory: Optional dict mapping device id to max memory (e.g., {0: "15GB"})

        Raises:
            ImportError: If transformers or bitsandbytes not installed
            RuntimeError: If model loading fails
        """
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library not installed. "
                "Install with: pip install transformers>=4.35.0"
            )

        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_4bit = load_in_4bit

        self.logger.info(
            f"Loading model {model_name} "
            f"(4-bit={load_in_4bit}, device={self.device})"
        )

        # Configure quantization if enabled
        if load_in_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # NF4 quantization
                bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
                bnb_4bit_use_double_quant=True  # Double quantization for extra savings
            )
            self.logger.info("4-bit quantization enabled (NF4)")
        else:
            quantization_config = None
            if load_in_4bit:
                self.logger.warning(
                    "4-bit quantization requested but CUDA not available. "
                    "Loading in full precision."
                )

        try:
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                max_memory=max_memory,
                low_cpu_mem_usage=True
            )

            self.logger.info("Model loaded successfully")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.debug("Set pad_token to eos_token")

            # Set to evaluation mode
            self.model.eval()

            # Log memory usage
            if torch.cuda.is_available():
                memory_stats = self.get_memory_usage()
                self.logger.info(
                    f"Initial GPU memory: "
                    f"{memory_stats.get('allocated_gb', 0):.2f}GB allocated, "
                    f"{memory_stats.get('reserved_gb', 0):.2f}GB reserved"
                )

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate completion using local model.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 2000)
                - max_input_length: Maximum input tokens (default: 2048)
                - top_p: Nucleus sampling (default: 0.95)
                - top_k: Top-k sampling (default: 50)
                - do_sample: Whether to use sampling (default: True)

        Returns:
            Generated text

        Raises:
            RuntimeError: If generation fails
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=kwargs.get("max_input_length", 2048)
            ).to(self.device)

            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", 2000),
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": kwargs.get("do_sample", True),
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 50),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            # Add stop sequences if provided
            if "stop" in kwargs and kwargs["stop"]:
                # Convert stop strings to token IDs
                stop_ids = [
                    self.tokenizer.encode(stop, add_special_tokens=False)
                    for stop in kwargs["stop"]
                ]
                gen_kwargs["eos_token_id"] = stop_ids

            # Generate
            self.logger.debug(
                f"Generating with temperature={gen_kwargs['temperature']}, "
                f"max_tokens={gen_kwargs['max_new_tokens']}"
            )

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Decode only the new tokens (excluding input)
            input_length = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )

            self.logger.debug(f"Generated {len(response)} characters")

            return response

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get GPU memory usage statistics.

        Returns:
            Dictionary with memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {}

        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
            'device_count': torch.cuda.device_count(),
        }

    def clear_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        info = {
            'model_name': self.model_name,
            'device': str(self.device),
            'load_in_4bit': self.load_in_4bit,
            'vocab_size': len(self.tokenizer),
        }

        # Add parameter count if available
        try:
            num_params = sum(p.numel() for p in self.model.parameters())
            info['num_parameters'] = num_params
            info['num_parameters_b'] = num_params / 1e9
        except:
            pass

        # Add memory usage if on CUDA
        if torch.cuda.is_available():
            info.update(self.get_memory_usage())

        return info

    def __str__(self) -> str:
        """String representation."""
        return f"LocalModel(model={self.model_name}, 4bit={self.load_in_4bit})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        memory = self.get_memory_usage()
        memory_str = f", mem={memory.get('allocated_gb', 0):.2f}GB" if memory else ""
        return (
            f"LocalModel("
            f"model={self.model_name}, "
            f"4bit={self.load_in_4bit}, "
            f"device={self.device}"
            f"{memory_str}"
            f")"
        )

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
