"""OpenRouter API client for remote model inference."""

import asyncio
import aiohttp
from typing import Dict, Any, Optional
import logging

from .base_model import BaseModel


class OpenRouterClient(BaseModel):
    """
    OpenRouter API client for accessing Llama 3.3 70B and other models.

    This client provides:
    - OpenAI-compatible API interface
    - Rate limiting (50 requests/day on free tier)
    - Automatic retry with exponential backoff
    - Request tracking

    Example:
        >>> client = OpenRouterClient(
        ...     api_key="your_key",
        ...     model_id="meta-llama/llama-3.3-70b-instruct:free"
        ... )
        >>> response = await client.generate("Hello, world!")
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "meta-llama/llama-3.3-70b-instruct:free",
        max_requests_per_day: int = 50,
        timeout: int = 120
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model_id: Model identifier (e.g., "meta-llama/llama-3.3-70b-instruct:free")
            max_requests_per_day: Maximum requests allowed per day
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = "https://openrouter.ai/api/v1"
        self.requests_count = 0
        self.max_requests_per_day = max_requests_per_day
        self.timeout = timeout

        self.logger.info(
            f"OpenRouter client initialized: model={model_id}, "
            f"rate_limit={max_requests_per_day}/day"
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate completion using OpenRouter API.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens (default: 2000)
                - top_p: Nucleus sampling (default: 1.0)
                - top_k: Top-k sampling (default: 0)
                - stop: Stop sequences (default: None)

        Returns:
            Generated text

        Raises:
            RuntimeError: If API call fails or rate limit exceeded
        """
        # Check rate limit
        if self.requests_count >= self.max_requests_per_day:
            raise RuntimeError(
                f"Daily rate limit reached ({self.max_requests_per_day} requests). "
                "Please wait 24 hours or upgrade to paid tier."
            )

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mle-star/framework",
            "X-Title": "MLE-STAR Framework"
        }

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000),
            "top_p": kwargs.get("top_p", 1.0),
        }

        # Add optional parameters
        if "stop" in kwargs and kwargs["stop"]:
            payload["stop"] = kwargs["stop"]

        # Retry with exponential backoff
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.requests_count += 1

                            # Extract response
                            content = data['choices'][0]['message']['content']

                            self.logger.debug(
                                f"Generated response: {len(content)} chars, "
                                f"requests={self.requests_count}/{self.max_requests_per_day}"
                            )

                            return content

                        elif response.status == 429:
                            # Rate limited
                            self.logger.warning(f"Rate limited on attempt {attempt + 1}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            else:
                                raise RuntimeError("Rate limit exceeded")

                        elif response.status == 401:
                            raise RuntimeError("Invalid API key")

                        elif response.status == 402:
                            raise RuntimeError("Insufficient credits")

                        else:
                            error_text = await response.text()
                            raise RuntimeError(
                                f"API request failed with status {response.status}: {error_text}"
                            )

            except asyncio.TimeoutError:
                last_error = "Request timeout"
                self.logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Request timed out after {max_retries} attempts")

            except aiohttp.ClientError as e:
                last_error = str(e)
                self.logger.warning(f"Client error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Client error after {max_retries} attempts: {e}")

        raise RuntimeError(f"Generation failed after {max_retries} retries: {last_error}")

    def reset_daily_counter(self) -> None:
        """
        Reset daily request counter.

        Call this at the start of each day to reset the rate limit counter.
        """
        self.requests_count = 0
        self.logger.info("Daily request counter reset")

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.

        Returns:
            Dictionary with usage information
        """
        return {
            'requests_used': self.requests_count,
            'requests_remaining': self.max_requests_per_day - self.requests_count,
            'rate_limit': self.max_requests_per_day,
            'model_id': self.model_id
        }

    def __str__(self) -> str:
        """String representation."""
        return f"OpenRouterClient(model={self.model_id})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"OpenRouterClient("
            f"model={self.model_id}, "
            f"requests={self.requests_count}/{self.max_requests_per_day}"
            f")"
        )
