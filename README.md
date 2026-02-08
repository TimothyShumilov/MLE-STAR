# MLE-STAR: Multi-Agent ML Engineering Framework

A production-ready framework for automated ML engineering using the STAR (Search, Test, And Refine) methodology with three specialized agents.

## Overview

MLE-STAR implements a multi-agent system for automating machine learning engineering tasks:

- **Planner Agent** (Llama 3.3 70B): Decomposes tasks and generates solution strategies
- **Executor Agent** (Qwen2.5-Coder 32B): Implements solutions through code generation
- **Verifier Agent** (Qwen2.5-Coder 14B): Validates results and provides feedback

### Key Features

✅ **Specialized Agents**: Three agents with clear roles and responsibilities
✅ **STAR Workflow**: Iterative Search → Evaluate → Refine process
✅ **Hybrid Deployment**: API-based Planner + local Executor/Verifier
✅ **Memory Optimized**: 4-bit quantization fits in 2x16GB VRAM (~14GB total)
✅ **Secure Execution**: Sandboxed code execution with guardrails
✅ **Built-in Monitoring**: Comprehensive metrics and resource tracking
✅ **Auto-Enrichment**: Automatic Kaggle metadata fetch and dataset profiling
✅ **Generic Framework**: Adaptable to any ML task or Kaggle competition

## Architecture

### Agent Roles

**Planner Agent** (via OpenRouter API)
- Model: Llama 3.3 70B Instruct (free tier)
- Capabilities: Task decomposition, strategy generation, refinement guidance
- Benchmarks: MMLU 86.0, MMLU-Pro 68.9

**Executor Agent** (local, 4-bit)
- Model: Qwen2.5-Coder 32B Instruct
- Memory: ~10GB VRAM in 4-bit quantization
- Capabilities: Code generation, execution, error handling
- Benchmarks: Comparable to GPT-4o on code tasks

**Verifier Agent** (local, 4-bit)
- Model: Qwen2.5-Coder 14B Instruct
- Memory: ~4GB VRAM in 4-bit quantization
- Capabilities: Result validation, bug detection, scoring
- Benchmarks: Strong code understanding and evaluation

### STAR Workflow

1. **Search Phase**: Generate multiple solution strategies (3 parallel)
2. **Evaluation Phase**: Execute and verify each strategy
3. **Refinement Phase**: Iterate based on feedback (max 5 iterations)

## Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU (2x16GB VRAM recommended)
- 50GB+ disk space for models

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/mle-star.git
cd mle-star

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Get OpenRouter API Key

1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Navigate to API Keys section
3. Create a new API key
4. Add to `.env` file: `OPENROUTER_API_KEY=your_key_here`

### Kaggle API Setup (Optional but Recommended)

For automatic competition information retrieval and dataset profiling:

1. **Create Kaggle Account**: Sign up at [Kaggle](https://www.kaggle.com)
2. **Generate API Token**:
   - Go to Account settings → API section
   - Click "Create New Token"
   - This downloads `kaggle.json` to your computer
3. **Install Credentials**:
   ```bash
   # Create Kaggle directory
   mkdir -p ~/.kaggle

   # Move kaggle.json to correct location
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

   # Set proper permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

**What Auto-Enrichment Provides:**

With Kaggle API configured, MLE-STAR automatically:
- ✅ Fetches competition descriptions, evaluation metrics, and requirements
- ✅ Analyzes dataset structure (columns, data types, missing values, statistics)
- ✅ Identifies potential target columns and task type
- ✅ Generates rich task context for agents
- ✅ Profiles datasets in <5 seconds with intelligent sampling

**Without API credentials**, you can still:
- ✅ Use dataset profiling (works on local CSV files)
- ✅ Provide manual task descriptions
- ✅ Run all framework features normally

See [Kaggle API Documentation](https://www.kaggle.com/docs/api#authentication) for more details.

## Quick Start

**Kaggle Competition - Minimal Example:**

```python
import asyncio
from mle_star.api.client import MLEStarClient

async def main():
    async with MLEStarClient.from_env() as client:
        # Just competition name - everything else is automatic!
        result = await client.execute_kaggle_competition("titanic")

        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Score: {result['result']['verification']['score']:.4f}")

asyncio.run(main())
```

**Custom ML Task:**

```python
import asyncio
from mle_star.api.client import MLEStarClient
from mle_star.tasks.task import Task, TaskType

async def main():
    async with MLEStarClient.from_env() as client:
        task = Task(
            description="Train a classifier on the Iris dataset with accuracy > 0.95",
            task_type=TaskType.CLASSIFICATION,
            success_criteria=["Accuracy > 0.95", "Code runs without errors"]
        )

        result = await client.execute_task(task)

        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Score: {result['result']['verification']['score']}")

asyncio.run(main())
```

## Examples

### Kaggle Competition - Full Automation

**Zero manual setup!** Just provide the competition name:

```python
import asyncio
from mle_star.api.client import MLEStarClient

async def main():
    async with MLEStarClient.from_env() as client:
        # Minimal API - just competition name!
        result = await client.execute_kaggle_competition("titanic")

        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Score: {result['result']['verification']['score']:.4f}")

asyncio.run(main())
```

**What Happens Automatically:**

1. **Data Download** (if data_dir not provided):
   - Downloads competition data to `~/.cache/mle_star/competitions/{competition_name}/`
   - Reuses cached data on subsequent runs
   - Automatic unzip and file extraction

2. **Kaggle API Fetch** (if credentials configured):
   - Competition title, description, and requirements
   - Auto-detects evaluation metric (accuracy, mse, auc, etc.)
   - Submission format and deadlines

3. **Submission Format Detection**:
   - Auto-detects from sample_submission file extension
   - Supports CSV, JSON, and other formats

4. **Dataset Profiling**:
   - Analyzes train.csv structure (891 rows × 12 columns for Titanic)
   - Detects data types (int64, float64, object)
   - Calculates missing values (Age: 19.9% missing)
   - Identifies target column (Survived)
   - Samples first rows for context

5. **Rich Task Description Generation**:
   ```
   Kaggle Competition: titanic
   Objective: Titanic - Machine Learning from Disaster

   Dataset Structure:
   - Rows: 891
   - Columns: 12

   Features:
     - PassengerId (int64): 891 unique, 0.0% missing
     - Survived (int64): 2 unique, 0.0% missing
     - Pclass (int64): 3 unique, 0.0% missing
     - Name (object): 891 unique, 0.0% missing
     - Sex (object): 2 unique, 0.0% missing
     - Age (float64): 88 unique, 19.9% missing
     - ...

   Evaluation Metric: accuracy
   Task: Generate code to train a model and create submission file.
   ```

**Running Examples:**

```bash
# Titanic (binary classification)
python examples/kaggle_competition.py

# MWS AI Agents 2026 (regression)
python examples/mws_ai_agents_competition.py
```

**Before** (manual setup):
```python
# Download data manually:
# $ kaggle competitions download -c mws-ai-agents-2026 -p ./data/mws_ai_agents
# $ unzip mws-ai-agents-2026.zip -d ./data/mws_ai_agents

description = """
Solve the MWS AI Agents competition...
[372 lines of manually written task description]
"""
result = await client.execute_kaggle_competition(
    "mws-ai-agents-2026",
    Path("./data/mws_ai_agents"),
    "mse",
    description=description
)
```

**After** (full automation):
```python
# Zero manual setup!
result = await client.execute_kaggle_competition("mws-ai-agents-2026")
# Framework automatically:
# - Downloads data to cache
# - Auto-detects metric (mse)
# - Auto-detects format (csv)
# - Generates rich description
```

**Advanced Options:**

```python
# Option 1: Manual data directory
result = await client.execute_kaggle_competition(
    "titanic",
    data_dir=Path("./my_data/titanic")
)

# Option 2: Manual metric override
result = await client.execute_kaggle_competition(
    "titanic",
    evaluation_metric="f1"
)

# Option 3: Disable auto-enrichment
result = await client.execute_kaggle_competition(
    "titanic",
    auto_enrich=False
)

# Option 4: Custom description (full manual control)
result = await client.execute_kaggle_competition(
    "titanic",
    data_dir=Path("./data/titanic"),
    evaluation_metric="accuracy",
    description="Custom task description...",
    auto_enrich=False
)
```

See [examples/kaggle_competition.py](examples/kaggle_competition.py) and [examples/mws_ai_agents_competition.py](examples/mws_ai_agents_competition.py) for complete examples.

### Custom ML Task

For custom machine learning tasks:

```bash
python examples/custom_ml_task.py
```

See [examples/custom_ml_task.py](examples/custom_ml_task.py)

## Configuration

Configuration can be loaded from:
1. Environment variables (`.env` file)
2. YAML configuration file
3. Python code

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_key_here

# Models
PLANNER_MODEL=meta-llama/llama-3.3-70b-instruct:free
EXECUTOR_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
VERIFIER_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct

# Resource Limits
MAX_GPU_MEMORY_GB=28
MAX_ITERATIONS=5
PARALLEL_STRATEGIES=3

# Monitoring
ENABLE_MONITORING=true
METRICS_DIR=./metrics
LOGS_DIR=./logs
```

## Technical Specifications

### Model Memory Footprint

| Model | Size | 4-bit Memory | 8-bit Memory |
|-------|------|--------------|--------------|
| Llama 3.3 70B | 70B | API (free) | API (free) |
| Qwen2.5-Coder-32B | 32B | ~10GB | ~16GB |
| Qwen2.5-Coder-14B | 14B | ~4GB | ~7GB |

**Total local VRAM: ~14GB** (leaves 18GB headroom on 32GB)

### Performance Targets

- Task initialization: <60 seconds (model loading)
- Strategy generation: <30 seconds per iteration
- Code execution: <300 seconds (configurable)
- Total workflow: <30 minutes per task

### API Rate Limits

- OpenRouter free tier: 50 requests/day
- Mitigation: Cache planning results, optimize prompts

## Security

### Multi-Layer Security

1. **Code Validation**: AST-based syntax and import checking
2. **Resource Limits**: Memory, CPU, time constraints
3. **Sandbox Isolation**: Subprocess with restricted permissions
4. **Guardrails**: Input/output validation and filtering

### Production Recommendations

- Use Docker containers for stronger isolation
- Implement network restrictions
- Monitor resource usage
- Regular security audits

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# With coverage
pytest --cov=mle_star tests/
```

### Code Quality

```bash
# Format code
black mle_star/ tests/

# Lint
flake8 mle_star/ tests/

# Type checking
mypy mle_star/
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## References

### Research
- [MLE-STAR Paper](https://arxiv.org/pdf/2506.15692) - Original architecture
- [OpenRouter](https://openrouter.ai/) - API platform
- [HuggingFace 4-bit Transformers](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

### Models
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
- [Llama 3.3 70B](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct:free)

### Benchmarks
- [HumanEval (code)](https://evalplus.github.io/leaderboard.html)
- [LiveCodeBench](https://livecodebench.github.io/leaderboard.html)
- [MMLU (reasoning)](https://artificialanalysis.ai/evaluations/mmlu-pro)

## Support

For questions, issues, or contributions:
- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/mle-star/issues)
- Documentation: [Full documentation](https://github.com/yourusername/mle-star/docs)

---

**Status**: 🚧 Active Development
**Version**: 0.1.0 (Alpha)
**Last Updated**: 2026-02-07
