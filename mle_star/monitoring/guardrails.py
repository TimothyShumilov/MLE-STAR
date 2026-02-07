"""Safety guardrails and validation for MLE-STAR framework."""

import re
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of validation check."""

    valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, message: str) -> None:
        """Add a validation issue."""
        self.valid = False
        self.issues.append(message)

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)


class InputGuardrails:
    """
    Input validation guardrails.

    Validates task descriptions, configurations, and user inputs
    to prevent malicious or invalid data.

    Example:
        >>> guardrails = InputGuardrails()
        >>> result = guardrails.validate_task_description("Train a classifier on Iris dataset")
        >>> if not result.valid:
        ...     print(f"Validation failed: {result.issues}")
    """

    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        r'rm\s+-rf',  # Dangerous file deletion
        r'sudo\s+',   # Privilege escalation
        r'chmod\s+777',  # Dangerous permissions
        r'eval\(',    # Dangerous eval
        r'exec\(',    # Dangerous exec
        r'__import__\(',  # Dynamic imports
        r'system\(',  # System calls
        r'popen\(',   # Process spawning
    ]

    # Suspicious keywords
    SUSPICIOUS_KEYWORDS = {
        'password', 'secret', 'api_key', 'token',
        'credential', 'private_key', 'ssh_key',
        'delete', 'drop', 'truncate', 'remove'
    }

    # Maximum lengths
    MAX_TASK_DESCRIPTION_LENGTH = 5000
    MAX_SUBTASK_COUNT = 20
    MAX_CRITERIA_COUNT = 10

    def __init__(self, strict_mode: bool = False):
        """
        Initialize input guardrails.

        Args:
            strict_mode: Enable strict validation (more restrictive)
        """
        self.strict_mode = strict_mode
        self.logger = logging.getLogger("mle_star.guardrails.input")

    def validate_task_description(self, description: str) -> ValidationResult:
        """
        Validate task description.

        Args:
            description: Task description text

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        # Check length
        if len(description) > self.MAX_TASK_DESCRIPTION_LENGTH:
            result.add_issue(
                f"Task description too long: {len(description)} > "
                f"{self.MAX_TASK_DESCRIPTION_LENGTH} characters"
            )

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, description, re.IGNORECASE):
                result.add_issue(f"Dangerous pattern detected: {pattern}")

        # Check for suspicious keywords
        description_lower = description.lower()
        found_keywords = [
            kw for kw in self.SUSPICIOUS_KEYWORDS
            if kw in description_lower
        ]

        if found_keywords:
            result.add_warning(
                f"Suspicious keywords found: {', '.join(found_keywords)}"
            )

        # Check for empty or too short
        if len(description.strip()) < 10:
            result.add_issue("Task description too short (minimum 10 characters)")

        return result

    def validate_task_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate task configuration.

        Args:
            config: Task configuration dictionary

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        # Check subtask count
        if 'subtasks' in config:
            subtask_count = len(config['subtasks'])
            if subtask_count > self.MAX_SUBTASK_COUNT:
                result.add_issue(
                    f"Too many subtasks: {subtask_count} > {self.MAX_SUBTASK_COUNT}"
                )

        # Check success criteria count
        if 'success_criteria' in config:
            criteria_count = len(config['success_criteria'])
            if criteria_count > self.MAX_CRITERIA_COUNT:
                result.add_issue(
                    f"Too many success criteria: {criteria_count} > {self.MAX_CRITERIA_COUNT}"
                )

        # Check for required fields
        required_fields = ['description']
        for field in required_fields:
            if field not in config:
                result.add_issue(f"Missing required field: {field}")

        return result

    def validate_file_path(self, path: str) -> ValidationResult:
        """
        Validate file path for security issues.

        Args:
            path: File path string

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        # Check for path traversal
        if '..' in path:
            result.add_issue("Path traversal detected (..)")

        # Check for absolute paths to sensitive locations
        sensitive_paths = ['/etc', '/root', '/sys', '/proc', '~/.ssh']
        for sensitive in sensitive_paths:
            if path.startswith(sensitive):
                result.add_issue(f"Access to sensitive path: {sensitive}")

        return result


class OutputGuardrails:
    """
    Output validation guardrails.

    Validates generated code and outputs before execution.

    Example:
        >>> guardrails = OutputGuardrails()
        >>> result = guardrails.validate_generated_code(code_string)
        >>> if not result.valid:
        ...     print(f"Code validation failed: {result.issues}")
    """

    # Forbidden imports (in addition to validator.py)
    FORBIDDEN_IMPORTS = {
        'subprocess', 'os.system', 'eval', 'exec',
        'socket', 'urllib', 'requests', 'http',
        'ctypes', 'pickle', 'marshal', 'shelve',
        '__builtin__', 'builtins'
    }

    # Allowed imports (whitelist mode)
    ALLOWED_IMPORTS = {
        'numpy', 'pandas', 'sklearn', 'scipy',
        'matplotlib', 'seaborn', 'torch', 'tensorflow',
        'xgboost', 'lightgbm', 'catboost',
        'PIL', 'cv2', 'transformers',
        'math', 'random', 'collections', 'itertools',
        'datetime', 'json', 'csv', 'io', 're'
    }

    MAX_CODE_LENGTH = 50000  # 50KB

    def __init__(self, whitelist_mode: bool = False):
        """
        Initialize output guardrails.

        Args:
            whitelist_mode: Use whitelist for imports (more restrictive)
        """
        self.whitelist_mode = whitelist_mode
        self.logger = logging.getLogger("mle_star.guardrails.output")

    def validate_generated_code(self, code: str) -> ValidationResult:
        """
        Validate generated code.

        Args:
            code: Python code string

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        # Check length
        if len(code) > self.MAX_CODE_LENGTH:
            result.add_issue(
                f"Code too long: {len(code)} > {self.MAX_CODE_LENGTH} characters"
            )

        # Extract imports
        import_pattern = r'(?:from|import)\s+(\w+)'
        imports = set(re.findall(import_pattern, code))

        # Check against forbidden imports
        forbidden_found = imports & self.FORBIDDEN_IMPORTS
        if forbidden_found:
            result.add_issue(
                f"Forbidden imports detected: {', '.join(forbidden_found)}"
            )

        # Whitelist mode: check against allowed imports
        if self.whitelist_mode:
            disallowed = imports - self.ALLOWED_IMPORTS
            if disallowed:
                result.add_warning(
                    f"Non-whitelisted imports: {', '.join(disallowed)}"
                )

        # Check for dangerous function calls
        dangerous_calls = ['eval(', 'exec(', 'compile(', '__import__(']
        for call in dangerous_calls:
            if call in code:
                result.add_issue(f"Dangerous function call detected: {call}")

        # Check for file operations (warn only)
        file_ops = ['open(', 'file(', 'with open']
        found_file_ops = [op for op in file_ops if op in code]
        if found_file_ops:
            result.add_warning(
                f"File operations detected: {', '.join(found_file_ops)}"
            )

        return result

    def validate_execution_result(self, result: Dict[str, Any]) -> ValidationResult:
        """
        Validate execution result.

        Args:
            result: Execution result dictionary

        Returns:
            ValidationResult
        """
        validation = ValidationResult(valid=True)

        # Check for required fields
        if 'status' not in result:
            validation.add_issue("Missing 'status' field in result")

        # Check for errors
        if result.get('status') == 'error':
            error_msg = result.get('error', 'Unknown error')
            validation.add_warning(f"Execution error: {error_msg}")

        return validation


class RateLimiter:
    """
    Rate limiter for API calls and resource usage.

    Tracks and limits API calls, task executions, and other operations
    to prevent abuse and manage costs.

    Example:
        >>> limiter = RateLimiter(max_calls_per_day=50)
        >>> if limiter.check_and_increment('api_call'):
        ...     # Make API call
        ...     pass
        ... else:
        ...     print("Rate limit exceeded")
    """

    def __init__(
        self,
        max_calls_per_day: int = 50,
        max_tasks_per_hour: int = 10,
        max_gpu_memory_gb: float = 28.0
    ):
        """
        Initialize rate limiter.

        Args:
            max_calls_per_day: Maximum API calls per day
            max_tasks_per_hour: Maximum tasks per hour
            max_gpu_memory_gb: Maximum GPU memory allowed
        """
        self.max_calls_per_day = max_calls_per_day
        self.max_tasks_per_hour = max_tasks_per_hour
        self.max_gpu_memory_gb = max_gpu_memory_gb

        # Tracking
        self.api_calls: List[datetime] = []
        self.task_starts: List[datetime] = []

        self.logger = logging.getLogger("mle_star.guardrails.rate_limiter")

    def check_and_increment(self, operation: str) -> bool:
        """
        Check rate limit and increment counter if allowed.

        Args:
            operation: Operation type ('api_call', 'task_start', 'gpu_memory')

        Returns:
            True if operation allowed, False if rate limit exceeded
        """
        now = datetime.utcnow()

        if operation == 'api_call':
            # Clean old calls (older than 1 day)
            cutoff = now - timedelta(days=1)
            self.api_calls = [call for call in self.api_calls if call > cutoff]

            # Check limit
            if len(self.api_calls) >= self.max_calls_per_day:
                self.logger.warning(
                    f"API call rate limit exceeded: "
                    f"{len(self.api_calls)}/{self.max_calls_per_day} per day"
                )
                return False

            # Increment
            self.api_calls.append(now)
            return True

        elif operation == 'task_start':
            # Clean old tasks (older than 1 hour)
            cutoff = now - timedelta(hours=1)
            self.task_starts = [task for task in self.task_starts if task > cutoff]

            # Check limit
            if len(self.task_starts) >= self.max_tasks_per_hour:
                self.logger.warning(
                    f"Task start rate limit exceeded: "
                    f"{len(self.task_starts)}/{self.max_tasks_per_hour} per hour"
                )
                return False

            # Increment
            self.task_starts.append(now)
            return True

        return True

    def get_remaining_quota(self, operation: str) -> int:
        """
        Get remaining quota for an operation.

        Args:
            operation: Operation type

        Returns:
            Remaining quota count
        """
        now = datetime.utcnow()

        if operation == 'api_call':
            cutoff = now - timedelta(days=1)
            current_calls = [call for call in self.api_calls if call > cutoff]
            return max(0, self.max_calls_per_day - len(current_calls))

        elif operation == 'task_start':
            cutoff = now - timedelta(hours=1)
            current_tasks = [task for task in self.task_starts if task > cutoff]
            return max(0, self.max_tasks_per_hour - len(current_tasks))

        return 0

    def reset(self) -> None:
        """Reset all counters."""
        self.api_calls.clear()
        self.task_starts.clear()
        self.logger.info("Rate limiter counters reset")


class BudgetTracker:
    """
    Budget tracker for API costs and resource usage.

    Tracks estimated costs and resource consumption to prevent
    budget overruns.

    Example:
        >>> tracker = BudgetTracker(max_daily_cost=10.0)
        >>> tracker.record_api_call(model="gpt-4", tokens=1000)
        >>> if tracker.is_budget_exceeded():
        ...     print("Budget exceeded!")
    """

    # Estimated costs (dollars per 1000 tokens)
    COST_PER_1K_TOKENS = {
        'gpt-4': 0.03,
        'gpt-3.5-turbo': 0.002,
        'llama-3.3-70b': 0.0,  # Free tier
    }

    def __init__(self, max_daily_cost: float = 10.0):
        """
        Initialize budget tracker.

        Args:
            max_daily_cost: Maximum daily cost in dollars
        """
        self.max_daily_cost = max_daily_cost
        self.daily_costs: List[tuple[datetime, float]] = []

        self.logger = logging.getLogger("mle_star.guardrails.budget")

    def record_api_call(
        self,
        model: str,
        tokens: int,
        cost_override: Optional[float] = None
    ) -> None:
        """
        Record an API call and its cost.

        Args:
            model: Model name
            tokens: Token count
            cost_override: Override estimated cost
        """
        now = datetime.utcnow()

        # Calculate cost
        if cost_override is not None:
            cost = cost_override
        else:
            cost_per_1k = self.COST_PER_1K_TOKENS.get(model, 0.01)  # Default
            cost = (tokens / 1000.0) * cost_per_1k

        self.daily_costs.append((now, cost))

        self.logger.debug(
            f"Recorded API call: model={model}, tokens={tokens}, cost=${cost:.4f}"
        )

    def get_daily_cost(self) -> float:
        """
        Get total cost for current day.

        Returns:
            Total cost in dollars
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(days=1)

        # Filter to last 24 hours
        recent_costs = [cost for timestamp, cost in self.daily_costs if timestamp > cutoff]

        return sum(recent_costs)

    def is_budget_exceeded(self) -> bool:
        """
        Check if budget is exceeded.

        Returns:
            True if budget exceeded
        """
        daily_cost = self.get_daily_cost()
        exceeded = daily_cost >= self.max_daily_cost

        if exceeded:
            self.logger.warning(
                f"Budget exceeded: ${daily_cost:.2f} >= ${self.max_daily_cost:.2f}"
            )

        return exceeded

    def get_remaining_budget(self) -> float:
        """
        Get remaining budget for the day.

        Returns:
            Remaining budget in dollars
        """
        daily_cost = self.get_daily_cost()
        return max(0.0, self.max_daily_cost - daily_cost)


class GuardrailsManager:
    """
    Centralized guardrails management.

    Combines input validation, output validation, rate limiting,
    and budget tracking.

    Example:
        >>> guardrails = GuardrailsManager()
        >>> # Validate input
        >>> result = guardrails.validate_task_input(task_description)
        >>> # Check rate limit
        >>> if not guardrails.check_rate_limit('api_call'):
        ...     raise Exception("Rate limit exceeded")
    """

    def __init__(
        self,
        strict_mode: bool = False,
        max_api_calls_per_day: int = 50,
        max_daily_cost: float = 10.0
    ):
        """
        Initialize guardrails manager.

        Args:
            strict_mode: Enable strict validation
            max_api_calls_per_day: Maximum API calls per day
            max_daily_cost: Maximum daily cost
        """
        self.input_guardrails = InputGuardrails(strict_mode=strict_mode)
        self.output_guardrails = OutputGuardrails(whitelist_mode=strict_mode)
        self.rate_limiter = RateLimiter(max_calls_per_day=max_api_calls_per_day)
        self.budget_tracker = BudgetTracker(max_daily_cost=max_daily_cost)

        self.logger = logging.getLogger("mle_star.guardrails")

    def validate_task_input(
        self,
        description: str,
        config: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate task input.

        Args:
            description: Task description
            config: Optional task configuration

        Returns:
            ValidationResult
        """
        # Validate description
        result = self.input_guardrails.validate_task_description(description)

        # Validate config if provided
        if config:
            config_result = self.input_guardrails.validate_task_config(config)
            result.issues.extend(config_result.issues)
            result.warnings.extend(config_result.warnings)
            if not config_result.valid:
                result.valid = False

        return result

    def validate_code_output(self, code: str) -> ValidationResult:
        """
        Validate generated code.

        Args:
            code: Python code string

        Returns:
            ValidationResult
        """
        return self.output_guardrails.validate_generated_code(code)

    def check_rate_limit(self, operation: str) -> bool:
        """
        Check and increment rate limit.

        Args:
            operation: Operation type

        Returns:
            True if allowed
        """
        return self.rate_limiter.check_and_increment(operation)

    def check_budget(self) -> bool:
        """
        Check if budget is available.

        Returns:
            True if budget available
        """
        return not self.budget_tracker.is_budget_exceeded()

    def record_api_call(self, model: str, tokens: int) -> None:
        """
        Record API call for rate limiting and budget tracking.

        Args:
            model: Model name
            tokens: Token count
        """
        self.budget_tracker.record_api_call(model, tokens)

    def get_status(self) -> Dict[str, Any]:
        """
        Get guardrails status.

        Returns:
            Status dictionary
        """
        return {
            'rate_limits': {
                'api_calls_remaining': self.rate_limiter.get_remaining_quota('api_call'),
                'tasks_remaining': self.rate_limiter.get_remaining_quota('task_start')
            },
            'budget': {
                'daily_cost': self.budget_tracker.get_daily_cost(),
                'remaining': self.budget_tracker.get_remaining_budget(),
                'exceeded': self.budget_tracker.is_budget_exceeded()
            }
        }
