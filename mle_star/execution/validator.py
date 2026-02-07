"""Code validation and security checks."""

import re
import ast
from typing import List, Dict, Any, Set, Tuple
import logging


class CodeValidator:
    """
    Validates generated code for security and safety.

    Security checks:
    - Syntax validation
    - Dangerous import detection
    - Dangerous pattern detection
    - Code complexity analysis
    - Forbidden operations

    Example:
        >>> validator = CodeValidator()
        >>> result = validator.validate_code("print('Hello')")
        >>> if result['valid']:
        ...     print("Code is safe!")
    """

    # Dangerous imports that should be blocked
    DANGEROUS_MODULES = {
        'subprocess', 'multiprocessing', 'threading',
        'socket', 'urllib', 'requests', 'urllib3',  # Network
        'ctypes', 'cffi',  # Low-level access
        '__builtin__', '__builtins__',  # Builtins manipulation
    }

    # Dangerous function calls
    DANGEROUS_CALLS = {
        'eval', 'exec', '__import__', 'compile',
        'open',  # File operations (can be dangerous)
        'input', 'raw_input',  # Interactive input
        'os.system', 'os.popen', 'os.spawn',
        'subprocess.run', 'subprocess.call', 'subprocess.Popen',
    }

    # Allowed safe imports
    SAFE_IMPORTS = {
        'pandas', 'numpy', 'sklearn', 'scikit-learn',
        'scipy', 'matplotlib', 'seaborn', 'plotly',
        'torch', 'tensorflow', 'keras',
        'xgboost', 'lightgbm', 'catboost',
        'statsmodels', 'nltk', 'spacy',
        'pillow', 'cv2', 'imageio',
        'json', 'csv', 'pickle', 'joblib',
        'datetime', 'time', 'random',
        'math', 'statistics', 'collections',
        're', 'string', 'itertools', 'functools',
    }

    def __init__(
        self,
        allow_file_io: bool = False,
        allow_network: bool = False,
        max_complexity: int = 100
    ):
        """
        Initialize code validator.

        Args:
            allow_file_io: Whether to allow file I/O operations
            allow_network: Whether to allow network operations
            max_complexity: Maximum cyclomatic complexity allowed
        """
        self.allow_file_io = allow_file_io
        self.allow_network = allow_network
        self.max_complexity = max_complexity

        self.logger = logging.getLogger("mle_star.validator")

    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate code and return detailed results.

        Args:
            code: Python code to validate

        Returns:
            Dictionary with validation results:
                - valid: bool
                - issues: List of issue strings
                - warnings: List of warning strings
                - complexity: int
                - forbidden_imports: List of forbidden imports found
                - forbidden_calls: List of forbidden calls found

        Example:
            >>> validator = CodeValidator()
            >>> result = validator.validate_code(code)
            >>> if not result['valid']:
            ...     print(f"Issues: {result['issues']}")
        """
        issues = []
        warnings = []
        forbidden_imports = []
        forbidden_calls = []

        # Check syntax
        syntax_valid, syntax_error = self._check_syntax(code)
        if not syntax_valid:
            issues.append(f"Syntax error: {syntax_error}")
            # Can't proceed with AST analysis if syntax is invalid
            return {
                'valid': False,
                'issues': issues,
                'warnings': warnings,
                'complexity': 0,
                'forbidden_imports': [],
                'forbidden_calls': []
            }

        # Parse code into AST
        try:
            tree = ast.parse(code)
        except:
            issues.append("Failed to parse code into AST")
            return {
                'valid': False,
                'issues': issues,
                'warnings': warnings,
                'complexity': 0,
                'forbidden_imports': [],
                'forbidden_calls': []
            }

        # Check imports
        forbidden_imports = self._check_imports(tree)
        if forbidden_imports:
            issues.append(f"Forbidden imports: {', '.join(forbidden_imports)}")

        # Check function calls
        forbidden_calls = self._check_function_calls(tree)
        if forbidden_calls:
            issues.append(f"Forbidden function calls: {', '.join(forbidden_calls)}")

        # Check dangerous patterns
        dangerous_patterns = self._check_patterns(code)
        if dangerous_patterns:
            issues.append(f"Dangerous patterns: {', '.join(dangerous_patterns)}")

        # Check file operations
        if not self.allow_file_io:
            file_ops = self._check_file_operations(tree)
            if file_ops:
                warnings.append(f"File operations detected: {', '.join(file_ops)}")

        # Check complexity
        complexity = self._calculate_complexity(tree)
        if complexity > self.max_complexity:
            warnings.append(
                f"High complexity: {complexity} (max: {self.max_complexity})"
            )

        # Determine validity
        valid = len(issues) == 0

        return {
            'valid': valid,
            'issues': issues,
            'warnings': warnings,
            'complexity': complexity,
            'forbidden_imports': forbidden_imports,
            'forbidden_calls': forbidden_calls
        }

    def _check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check Python syntax.

        Args:
            code: Code to check

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"{e.msg} at line {e.lineno}"
        except Exception as e:
            return False, str(e)

    def _check_imports(self, tree: ast.AST) -> List[str]:
        """
        Check for forbidden imports.

        Args:
            tree: AST tree

        Returns:
            List of forbidden imports found
        """
        forbidden = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in self.DANGEROUS_MODULES:
                        forbidden.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in self.DANGEROUS_MODULES:
                        forbidden.append(node.module)

        return forbidden

    def _check_function_calls(self, tree: ast.AST) -> List[str]:
        """
        Check for forbidden function calls.

        Args:
            tree: AST tree

        Returns:
            List of forbidden calls found
        """
        forbidden = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Get function name
                func_name = self._get_call_name(node.func)

                if func_name in self.DANGEROUS_CALLS:
                    forbidden.append(func_name)

        return forbidden

    def _get_call_name(self, node) -> str:
        """
        Extract function call name from AST node.

        Args:
            node: AST node

        Returns:
            Function name as string
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # For calls like os.system
            base = self._get_call_name(node.value)
            return f"{base}.{node.attr}"
        else:
            return ""

    def _check_patterns(self, code: str) -> List[str]:
        """
        Check for dangerous patterns using regex.

        Args:
            code: Source code

        Returns:
            List of dangerous patterns found
        """
        patterns = []

        # Check for eval/exec
        if re.search(r'\beval\s*\(', code):
            patterns.append('eval()')
        if re.search(r'\bexec\s*\(', code):
            patterns.append('exec()')

        # Check for __import__
        if re.search(r'__import__\s*\(', code):
            patterns.append('__import__()')

        # Check for shell commands
        if re.search(r'os\.system\s*\(', code):
            patterns.append('os.system()')

        return patterns

    def _check_file_operations(self, tree: ast.AST) -> List[str]:
        """
        Check for file operations.

        Args:
            tree: AST tree

        Returns:
            List of file operations found
        """
        file_ops = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node.func)

                if func_name == 'open':
                    file_ops.append('open()')

        return file_ops

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """
        Calculate cyclomatic complexity.

        Args:
            tree: AST tree

        Returns:
            Complexity score
        """
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            # Branch points
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            # Function/class definitions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity += 1

        return complexity

    def get_safe_imports_suggestions(self, code: str) -> List[str]:
        """
        Get suggestions for safe imports based on code content.

        Args:
            code: Source code

        Returns:
            List of suggested safe imports
        """
        suggestions = []

        # Simple heuristics
        if 'pd.' in code or 'DataFrame' in code:
            suggestions.append('import pandas as pd')

        if 'np.' in code or 'array' in code:
            suggestions.append('import numpy as np')

        if 'sklearn' in code or 'fit(' in code:
            suggestions.append('from sklearn import ...')

        if 'plt.' in code or 'plot(' in code:
            suggestions.append('import matplotlib.pyplot as plt')

        return suggestions

    def __str__(self) -> str:
        """String representation."""
        return (
            f"CodeValidator("
            f"file_io={self.allow_file_io}, "
            f"network={self.allow_network}, "
            f"max_complexity={self.max_complexity}"
            f")"
        )
