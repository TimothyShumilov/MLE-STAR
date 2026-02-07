"""Resource monitoring for GPU, CPU, and RAM usage."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import psutil

# Try to import torch for GPU monitoring
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time."""

    timestamp: float

    # GPU
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_count: int = 0

    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 0

    # RAM
    ram_used_mb: float = 0.0
    ram_available_mb: float = 0.0
    ram_percent: float = 0.0

    # Disk
    disk_used_gb: float = 0.0
    disk_available_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'gpu': {
                'memory_allocated_mb': self.gpu_memory_allocated_mb,
                'memory_reserved_mb': self.gpu_memory_reserved_mb,
                'memory_total_mb': self.gpu_memory_total_mb,
                'utilization_percent': self.gpu_utilization_percent,
                'gpu_count': self.gpu_count
            },
            'cpu': {
                'percent': self.cpu_percent,
                'count': self.cpu_count
            },
            'ram': {
                'used_mb': self.ram_used_mb,
                'available_mb': self.ram_available_mb,
                'percent': self.ram_percent
            },
            'disk': {
                'used_gb': self.disk_used_gb,
                'available_gb': self.disk_available_gb
            }
        }


class ResourceMonitor:
    """
    Monitor system resource usage (GPU, CPU, RAM).

    Provides real-time monitoring with optional callbacks for alerts
    when resource usage exceeds thresholds.

    Example:
        >>> monitor = ResourceMonitor()
        >>> monitor.start(interval=5.0)  # Monitor every 5 seconds
        >>> snapshot = monitor.get_current_snapshot()
        >>> print(f"GPU Memory: {snapshot.gpu_memory_allocated_mb}MB")
        >>> monitor.stop()
    """

    def __init__(
        self,
        gpu_memory_threshold_mb: float = 28000,  # 28GB default
        cpu_threshold_percent: float = 90.0,
        ram_threshold_percent: float = 90.0,
        on_threshold_exceeded: Optional[Callable] = None
    ):
        """
        Initialize resource monitor.

        Args:
            gpu_memory_threshold_mb: GPU memory alert threshold
            cpu_threshold_percent: CPU usage alert threshold
            ram_threshold_percent: RAM usage alert threshold
            on_threshold_exceeded: Callback when threshold exceeded
        """
        self.gpu_memory_threshold_mb = gpu_memory_threshold_mb
        self.cpu_threshold_percent = cpu_threshold_percent
        self.ram_threshold_percent = ram_threshold_percent
        self.on_threshold_exceeded = on_threshold_exceeded

        self.logger = logging.getLogger("mle_star.resource_monitor")

        # Monitoring state
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Peak values
        self.peak_gpu_memory_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.peak_ram_percent = 0.0

        # Check GPU availability
        self.has_gpu = HAS_TORCH and torch.cuda.is_available()

        if not HAS_TORCH:
            self.logger.warning("PyTorch not available - GPU monitoring disabled")
        elif not self.has_gpu:
            self.logger.warning("No CUDA GPU detected - GPU monitoring disabled")
        else:
            gpu_count = torch.cuda.device_count()
            self.logger.info(f"GPU monitoring enabled: {gpu_count} GPU(s) detected")

    def get_current_snapshot(self) -> ResourceSnapshot:
        """
        Get current resource usage snapshot.

        Returns:
            ResourceSnapshot with current metrics
        """
        snapshot = ResourceSnapshot(timestamp=time.time())

        # GPU metrics
        if self.has_gpu:
            try:
                snapshot.gpu_count = torch.cuda.device_count()

                # Sum across all GPUs
                total_allocated = 0
                total_reserved = 0
                total_memory = 0

                for i in range(snapshot.gpu_count):
                    allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MB
                    reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)  # MB
                    total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)  # MB

                    total_allocated += allocated
                    total_reserved += reserved
                    total_memory += total

                snapshot.gpu_memory_allocated_mb = total_allocated
                snapshot.gpu_memory_reserved_mb = total_reserved
                snapshot.gpu_memory_total_mb = total_memory

                # Utilization (estimated from memory usage)
                if total_memory > 0:
                    snapshot.gpu_utilization_percent = (total_allocated / total_memory) * 100

                # Update peak
                if total_allocated > self.peak_gpu_memory_mb:
                    self.peak_gpu_memory_mb = total_allocated

            except Exception as e:
                self.logger.warning(f"Failed to get GPU metrics: {e}")

        # CPU metrics
        try:
            snapshot.cpu_percent = psutil.cpu_percent(interval=0.1)
            snapshot.cpu_count = psutil.cpu_count()

            if snapshot.cpu_percent > self.peak_cpu_percent:
                self.peak_cpu_percent = snapshot.cpu_percent

        except Exception as e:
            self.logger.warning(f"Failed to get CPU metrics: {e}")

        # RAM metrics
        try:
            memory = psutil.virtual_memory()
            snapshot.ram_used_mb = memory.used / (1024 ** 2)
            snapshot.ram_available_mb = memory.available / (1024 ** 2)
            snapshot.ram_percent = memory.percent

            if snapshot.ram_percent > self.peak_ram_percent:
                self.peak_ram_percent = snapshot.ram_percent

        except Exception as e:
            self.logger.warning(f"Failed to get RAM metrics: {e}")

        # Disk metrics
        try:
            disk = psutil.disk_usage('/')
            snapshot.disk_used_gb = disk.used / (1024 ** 3)
            snapshot.disk_available_gb = disk.free / (1024 ** 3)

        except Exception as e:
            self.logger.warning(f"Failed to get disk metrics: {e}")

        return snapshot

    def check_thresholds(self, snapshot: ResourceSnapshot) -> None:
        """
        Check if resource usage exceeds thresholds.

        Args:
            snapshot: Current resource snapshot
        """
        alerts = []

        # GPU memory
        if snapshot.gpu_memory_allocated_mb > self.gpu_memory_threshold_mb:
            alerts.append(
                f"GPU memory exceeded threshold: "
                f"{snapshot.gpu_memory_allocated_mb:.0f}MB > "
                f"{self.gpu_memory_threshold_mb:.0f}MB"
            )

        # CPU
        if snapshot.cpu_percent > self.cpu_threshold_percent:
            alerts.append(
                f"CPU usage exceeded threshold: "
                f"{snapshot.cpu_percent:.1f}% > "
                f"{self.cpu_threshold_percent:.1f}%"
            )

        # RAM
        if snapshot.ram_percent > self.ram_threshold_percent:
            alerts.append(
                f"RAM usage exceeded threshold: "
                f"{snapshot.ram_percent:.1f}% > "
                f"{self.ram_threshold_percent:.1f}%"
            )

        # Trigger callback if any alerts
        if alerts:
            for alert in alerts:
                self.logger.warning(alert)

            if self.on_threshold_exceeded:
                try:
                    self.on_threshold_exceeded(snapshot, alerts)
                except Exception as e:
                    self.logger.error(f"Threshold callback failed: {e}")

    async def _monitor_loop(self, interval: float) -> None:
        """
        Background monitoring loop.

        Args:
            interval: Monitoring interval in seconds
        """
        self.logger.info(f"Resource monitoring started (interval={interval}s)")

        while self.monitoring:
            try:
                snapshot = self.get_current_snapshot()
                self.check_thresholds(snapshot)

                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(interval)

    def start(self, interval: float = 10.0) -> None:
        """
        Start background monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            self.logger.warning("Monitoring already running")
            return

        self.monitoring = True

        # Start background task
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.monitor_task = loop.create_task(self._monitor_loop(interval))

        self.logger.info("Resource monitoring started")

    def stop(self) -> None:
        """Stop background monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False

        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None

        self.logger.info("Resource monitoring stopped")

    def reset_peaks(self) -> None:
        """Reset peak values."""
        self.peak_gpu_memory_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.peak_ram_percent = 0.0

    def get_peak_usage(self) -> Dict[str, float]:
        """
        Get peak resource usage.

        Returns:
            Dictionary with peak values
        """
        return {
            'peak_gpu_memory_mb': self.peak_gpu_memory_mb,
            'peak_cpu_percent': self.peak_cpu_percent,
            'peak_ram_percent': self.peak_ram_percent
        }

    def get_gpu_memory_available(self) -> float:
        """
        Get available GPU memory in MB.

        Returns:
            Available GPU memory or 0 if no GPU
        """
        if not self.has_gpu:
            return 0.0

        try:
            snapshot = self.get_current_snapshot()
            return snapshot.gpu_memory_total_mb - snapshot.gpu_memory_allocated_mb
        except Exception as e:
            self.logger.warning(f"Failed to get available GPU memory: {e}")
            return 0.0

    def clear_gpu_cache(self) -> None:
        """Clear PyTorch GPU cache."""
        if not self.has_gpu:
            return

        try:
            torch.cuda.empty_cache()
            self.logger.info("GPU cache cleared")
        except Exception as e:
            self.logger.warning(f"Failed to clear GPU cache: {e}")

    def log_summary(self) -> None:
        """Log current resource summary."""
        snapshot = self.get_current_snapshot()

        summary = f"""
Resource Usage Summary:
  GPU: {snapshot.gpu_memory_allocated_mb:.0f}MB / {snapshot.gpu_memory_total_mb:.0f}MB ({snapshot.gpu_utilization_percent:.1f}%)
  CPU: {snapshot.cpu_percent:.1f}% ({snapshot.cpu_count} cores)
  RAM: {snapshot.ram_used_mb:.0f}MB ({snapshot.ram_percent:.1f}%)
  Disk: {snapshot.disk_used_gb:.1f}GB used, {snapshot.disk_available_gb:.1f}GB available

Peak Usage:
  GPU: {self.peak_gpu_memory_mb:.0f}MB
  CPU: {self.peak_cpu_percent:.1f}%
  RAM: {self.peak_ram_percent:.1f}%
"""

        self.logger.info(summary.strip())

    def __str__(self) -> str:
        """String representation."""
        status = "active" if self.monitoring else "inactive"
        return f"ResourceMonitor(status={status}, gpu={self.has_gpu})"

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


# Utility functions

def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information.

    Returns:
        Dictionary with GPU details or empty dict if no GPU
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {}

    try:
        gpu_count = torch.cuda.device_count()
        gpus = []

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                'index': i,
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024 ** 3),
                'compute_capability': f"{props.major}.{props.minor}"
            })

        return {
            'count': gpu_count,
            'gpus': gpus,
            'cuda_version': torch.version.cuda
        }

    except Exception as e:
        logging.getLogger("mle_star.resource_monitor").error(
            f"Failed to get GPU info: {e}"
        )
        return {}


def check_gpu_available(min_memory_gb: float = 8.0) -> bool:
    """
    Check if GPU is available with minimum memory.

    Args:
        min_memory_gb: Minimum required GPU memory in GB

    Returns:
        True if GPU available with sufficient memory
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return False

    try:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory_gb = props.total_memory / (1024 ** 3)

            if total_memory_gb >= min_memory_gb:
                return True

        return False

    except Exception:
        return False
