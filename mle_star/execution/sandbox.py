"""Secure code execution sandbox."""

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import subprocess
import logging
import time
import sys
import platform


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""

    status: str  # 'success', 'error', 'timeout'
    stdout: str
    stderr: str
    execution_time: float
    exit_code: int


class CodeSandbox:
    """
    Secure code execution sandbox.

    Security layers:
    1. Temporary isolated filesystem
    2. Resource limits (CPU, memory, time)
    3. Subprocess isolation
    4. File system restrictions

    For production: Consider Docker containers or firejail for stronger isolation.

    Example:
        >>> sandbox = CodeSandbox(max_execution_time=300)
        >>> result = await sandbox.execute("print('Hello, world!')")
        >>> print(result.stdout)
        Hello, world!
    """

    def __init__(
        self,
        max_execution_time: int = 300,
        max_memory_mb: int = 4096,
        enable_network: bool = True
    ):
        """
        Initialize code sandbox.

        Args:
            max_execution_time: Maximum execution time in seconds
            max_memory_mb: Maximum memory in MB
            enable_network: Whether to allow network access
        """
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.enable_network = enable_network
        self.temp_dir = None

        self.logger = logging.getLogger("mle_star.sandbox")
        self.logger.info(
            f"Sandbox initialized: "
            f"max_time={max_execution_time}s, "
            f"max_memory={max_memory_mb}MB"
        )

    async def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        memory_limit: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute code in sandbox.

        Args:
            code: Python code to execute
            timeout: Optional timeout override
            memory_limit: Optional memory limit (e.g., "4G")

        Returns:
            ExecutionResult with output and status

        Example:
            >>> result = await sandbox.execute("import pandas as pd\\nprint(pd.__version__)")
        """
        timeout = timeout or self.max_execution_time

        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="mle_star_sandbox_"))

        try:
            self.logger.debug(f"Executing code in: {self.temp_dir}")

            # Write code to file
            code_file = self.temp_dir / "main.py"
            code_file.write_text(code, encoding='utf-8')

            # Execute with restrictions
            result = await self._run_with_limits(code_file, timeout)

            return result

        except Exception as e:
            self.logger.error(f"Sandbox execution failed: {e}")
            return ExecutionResult(
                status='error',
                stdout='',
                stderr=str(e),
                execution_time=0.0,
                exit_code=-1
            )

        finally:
            # Cleanup
            self._cleanup()

    async def _run_with_limits(
        self,
        code_file: Path,
        timeout: int
    ) -> ExecutionResult:
        """
        Run code with resource limits.

        Args:
            code_file: Path to Python file
            timeout: Timeout in seconds

        Returns:
            ExecutionResult
        """
        start_time = time.time()

        try:
            # Prepare subprocess
            # Note: On Windows, resource limits via resource module don't work
            # For production, use Docker or OS-specific mechanisms

            if platform.system() == 'Windows':
                # Windows: basic subprocess without resource limits
                self.logger.warning(
                    "Running on Windows - resource limits not enforced"
                )
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(code_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.temp_dir
                )
            else:
                # Unix/Linux/Mac: use resource limits
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(code_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.temp_dir,
                    preexec_fn=self._set_limits
                )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                execution_time = time.time() - start_time

                return ExecutionResult(
                    status='success' if process.returncode == 0 else 'error',
                    stdout=stdout.decode('utf-8', errors='replace'),
                    stderr=stderr.decode('utf-8', errors='replace'),
                    execution_time=execution_time,
                    exit_code=process.returncode or 0
                )

            except asyncio.TimeoutError:
                # Timeout - kill process
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass

                return ExecutionResult(
                    status='timeout',
                    stdout='',
                    stderr=f'Execution timeout ({timeout}s exceeded)',
                    execution_time=timeout,
                    exit_code=-1
                )

        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return ExecutionResult(
                status='error',
                stdout='',
                stderr=str(e),
                execution_time=time.time() - start_time,
                exit_code=-1
            )

    def _set_limits(self):
        """
        Set resource limits for subprocess (Unix/Linux/Mac only).

        This function is called before the subprocess starts.
        """
        try:
            import resource

            # Memory limit (in bytes)
            max_memory_bytes = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_AS,
                (max_memory_bytes, max_memory_bytes)
            )

            # CPU time limit
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self.max_execution_time, self.max_execution_time)
            )

            # File size limit (1GB)
            max_file_size = 1024 * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (max_file_size, max_file_size)
            )

            # Number of processes
            resource.setrlimit(
                resource.RLIMIT_NPROC,
                (100, 100)
            )

        except ImportError:
            # resource module not available (Windows)
            pass
        except Exception as e:
            self.logger.warning(f"Could not set resource limits: {e}")

    def _cleanup(self):
        """Cleanup temporary directory."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.logger.debug(f"Cleaned up: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Cleanup failed: {e}")

    async def execute_with_files(
        self,
        code: str,
        input_files: Optional[Dict[str, bytes]] = None,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute code with additional input files.

        Args:
            code: Python code
            input_files: Dictionary mapping filenames to content
            timeout: Optional timeout

        Returns:
            ExecutionResult
        """
        timeout = timeout or self.max_execution_time

        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="mle_star_sandbox_"))

        try:
            # Write input files
            if input_files:
                for filename, content in input_files.items():
                    file_path = self.temp_dir / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_bytes(content)

            # Write and execute code
            code_file = self.temp_dir / "main.py"
            code_file.write_text(code, encoding='utf-8')

            result = await self._run_with_limits(code_file, timeout)

            return result

        finally:
            self._cleanup()

    def __str__(self) -> str:
        """String representation."""
        return (
            f"CodeSandbox("
            f"max_time={self.max_execution_time}s, "
            f"max_memory={self.max_memory_mb}MB"
            f")"
        )


# Production-ready alternative using Docker (example)
class DockerSandbox(CodeSandbox):
    """
    Docker-based sandbox for stronger isolation (production use).

    This is a placeholder for a Docker-based implementation.
    Requires Docker to be installed and configured.

    Note: Not fully implemented - use CodeSandbox for now.
    """

    def __init__(self, *args, **kwargs):
        """Initialize Docker sandbox."""
        super().__init__(*args, **kwargs)
        self.docker_image = "python:3.9-slim"
        self.logger.warning(
            "DockerSandbox is not fully implemented. "
            "Using basic CodeSandbox instead."
        )

    async def _run_with_limits(
        self,
        code_file: Path,
        timeout: int
    ) -> ExecutionResult:
        """
        Run code in Docker container.

        This would use Docker to run code in an isolated container
        with strict resource limits and network isolation.

        For now, falls back to parent implementation.
        """
        # TODO: Implement Docker execution
        # docker run --rm --memory=4g --cpus=1 --network=none \
        #            -v /path/to/code:/code python:3.9 python /code/main.py

        return await super()._run_with_limits(code_file, timeout)
