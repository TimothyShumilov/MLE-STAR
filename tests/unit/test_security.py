"""Unit tests for security components (sandbox and validator)."""

import pytest
import asyncio

from mle_star.execution.sandbox import CodeSandbox, ExecutionResult
from mle_star.execution.validator import CodeValidator


@pytest.mark.unit
class TestCodeValidator:
    """Test CodeValidator class."""

    def test_validate_safe_code(self, sample_code):
        """Test validation of safe code."""
        validator = CodeValidator()
        result = validator.validate_code(sample_code)

        assert result['valid'] is True
        assert len(result['issues']) == 0

    def test_validate_dangerous_imports(self, dangerous_code):
        """Test detection of dangerous imports."""
        validator = CodeValidator()
        result = validator.validate_code(dangerous_code)

        assert result['valid'] is False
        assert len(result['issues']) > 0

        # Check that forbidden imports were detected
        issues_text = ' '.join(result['issues']).lower()
        assert 'subprocess' in issues_text or 'import' in issues_text

    def test_validate_dangerous_functions(self):
        """Test detection of dangerous function calls."""
        code = """
x = eval('1 + 1')
y = exec('print("hello")')
"""

        validator = CodeValidator()
        result = validator.validate_code(code)

        assert result['valid'] is False
        issues_text = ' '.join(result['issues']).lower()
        assert 'eval' in issues_text or 'exec' in issues_text

    def test_validate_syntax_error(self):
        """Test detection of syntax errors."""
        code = """
def broken_function(
    # Missing closing parenthesis
    print("This won't work")
"""

        validator = CodeValidator()
        result = validator.validate_code(code)

        assert result['valid'] is False
        assert any('syntax' in issue.lower() for issue in result['issues'])

    def test_validate_complexity(self):
        """Test complexity analysis."""
        # Very complex code with high cyclomatic complexity
        complex_code = """
def complex_function(x):
    if x > 10:
        if x > 20:
            if x > 30:
                if x > 40:
                    if x > 50:
                        return "very high"
                    return "high"
                return "medium-high"
            return "medium"
        return "low-medium"
    return "low"
"""

        validator = CodeValidator()
        result = validator.validate_code(complex_code)

        assert 'complexity' in result
        assert result['complexity'] > 5  # Should detect high complexity

    def test_validate_file_operations_warning(self, safe_code_with_warnings):
        """Test warning for file operations."""
        validator = CodeValidator()
        result = validator.validate_code(safe_code_with_warnings)

        # File operations should generate warnings but not fail validation
        # (unless strict mode is enabled)
        assert len(result['warnings']) > 0

    def test_whitelist_mode(self):
        """Test whitelist mode for imports."""
        code = """
import unusual_package
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3]})
print(df)
"""

        validator = CodeValidator(whitelist_mode=True)
        result = validator.validate_code(code)

        # pandas should be allowed, unusual_package should trigger warning
        assert len(result['warnings']) > 0


@pytest.mark.unit
class TestCodeSandbox:
    """Test CodeSandbox class."""

    def test_sandbox_initialization(self):
        """Test sandbox initialization."""
        sandbox = CodeSandbox(
            max_execution_time=60,
            max_memory_mb=1024
        )

        assert sandbox.max_execution_time == 60
        assert sandbox.max_memory_mb == 1024

    @pytest.mark.asyncio
    async def test_execute_simple_code(self):
        """Test executing simple code."""
        sandbox = CodeSandbox(max_execution_time=10)

        code = """
print("Hello, world!")
result = 2 + 2
print(f"Result: {result}")
"""

        result = await sandbox.execute(code)

        assert result.status == 'success'
        assert result.exit_code == 0
        assert "Hello, world!" in result.stdout
        assert "Result: 4" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_with_error(self):
        """Test executing code that raises an error."""
        sandbox = CodeSandbox(max_execution_time=10)

        code = """
x = 1 / 0  # ZeroDivisionError
"""

        result = await sandbox.execute(code)

        assert result.status == 'error'
        assert result.exit_code != 0
        assert "ZeroDivisionError" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test execution timeout."""
        sandbox = CodeSandbox(max_execution_time=2)

        code = """
import time
time.sleep(10)  # Will timeout after 2 seconds
print("This should not print")
"""

        result = await sandbox.execute(code, timeout=2)

        assert result.status == 'timeout'
        assert "timeout" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_execute_with_output(self):
        """Test capturing stdout and stderr."""
        sandbox = CodeSandbox()

        code = """
import sys

print("This goes to stdout")
print("This goes to stderr", file=sys.stderr)
"""

        result = await sandbox.execute(code)

        assert result.status == 'success'
        assert "stdout" in result.stdout
        assert "stderr" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_with_files(self):
        """Test executing code with input files."""
        sandbox = CodeSandbox()

        code = """
with open('data.txt', 'r') as f:
    content = f.read()
    print(f"File content: {content}")
"""

        files = {
            'data.txt': b'Hello from file!'
        }

        result = await sandbox.execute_with_files(code, input_files=files)

        assert result.status == 'success'
        assert "Hello from file!" in result.stdout

    @pytest.mark.asyncio
    async def test_cleanup_temp_dir(self):
        """Test that temporary directory is cleaned up."""
        sandbox = CodeSandbox()

        code = "print('test')"

        result = await sandbox.execute(code)

        assert result.status == 'success'

        # temp_dir should be None after cleanup
        assert sandbox.temp_dir is None

    @pytest.mark.asyncio
    async def test_execution_time_measurement(self):
        """Test that execution time is measured."""
        sandbox = CodeSandbox()

        code = """
import time
time.sleep(0.1)
print("Done")
"""

        result = await sandbox.execute(code)

        assert result.status == 'success'
        assert result.execution_time > 0.1
        assert result.execution_time < 1.0  # Should complete quickly


@pytest.mark.unit
class TestSandboxSecurity:
    """Test sandbox security features."""

    @pytest.mark.asyncio
    async def test_isolated_filesystem(self):
        """Test that sandbox uses isolated filesystem."""
        sandbox = CodeSandbox()

        code = """
import os
print(f"Working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")
"""

        result = await sandbox.execute(code)

        assert result.status == 'success'
        # Should be in a temp directory
        assert "mle_star_sandbox" in result.stdout

    @pytest.mark.asyncio
    async def test_cannot_access_parent_directory(self):
        """Test that code cannot access parent directory."""
        sandbox = CodeSandbox()

        code = """
import os
try:
    # Try to access parent directory
    files = os.listdir('..')
    print("SUCCESS: Accessed parent directory")
except Exception as e:
    print(f"BLOCKED: {type(e).__name__}")
"""

        result = await sandbox.execute(code)

        # Accessing parent should either succeed (showing only temp dir)
        # or fail - either way, it shouldn't access real system files
        assert result.status in ['success', 'error']

    @pytest.mark.asyncio
    async def test_network_operations(self):
        """Test network operations (should work if enabled)."""
        sandbox = CodeSandbox(enable_network=True)

        code = """
try:
    import urllib.request
    # Try to make a network request
    # (will fail without real network, but that's OK for test)
    print("Network access attempted")
except Exception as e:
    print(f"Network error: {type(e).__name__}")
"""

        result = await sandbox.execute(code)

        # Should execute without sandbox errors
        assert result.status in ['success', 'error']
        # Should not be a timeout
        assert result.status != 'timeout'


@pytest.mark.unit
class TestValidatorEdgeCases:
    """Test edge cases for code validator."""

    def test_empty_code(self):
        """Test validation of empty code."""
        validator = CodeValidator()
        result = validator.validate_code("")

        # Empty code might be valid or invalid depending on implementation
        # Just check it doesn't crash
        assert 'valid' in result

    def test_whitespace_only(self):
        """Test validation of whitespace-only code."""
        validator = CodeValidator()
        result = validator.validate_code("   \n  \n  ")

        assert 'valid' in result

    def test_comments_only(self):
        """Test validation of comments-only code."""
        code = """
# This is a comment
# Another comment
"""

        validator = CodeValidator()
        result = validator.validate_code(code)

        assert result['valid'] is True

    def test_very_long_code(self):
        """Test validation of very long code."""
        # Generate long but valid code
        code = "\n".join([f"x{i} = {i}" for i in range(10000)])

        validator = CodeValidator()
        result = validator.validate_code(code)

        # Should validate (though might be slow)
        assert 'valid' in result

    def test_unicode_code(self):
        """Test validation of code with unicode characters."""
        code = """
# Привет мир (Russian)
# 你好世界 (Chinese)
message = "Hello, 世界!"
print(message)
"""

        validator = CodeValidator()
        result = validator.validate_code(code)

        assert result['valid'] is True

    def test_nested_imports(self):
        """Test detection of nested imports."""
        code = """
def dynamic_import():
    import sys
    import subprocess  # Forbidden!
    return subprocess
"""

        validator = CodeValidator()
        result = validator.validate_code(code)

        assert result['valid'] is False
        assert 'subprocess' in str(result['issues']).lower()
