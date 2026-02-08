"""
Kaggle Competition Example - Fully Automated.

This example demonstrates using MLE-STAR for Kaggle competitions
with all configuration passed directly in code (no .env file required).

Competition: https://www.kaggle.com/competitions/mws-ai-agents-2026/overview

Usage:
    1. Update the CONFIG section below with your credentials
    2. Update COMPETITION_NAME if running a different competition
    3. Run: python examples/mws_ai_agents_competition.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mle_star.api.client import MLEStarClient
from mle_star.utils.config import Config


# ============================================================================
# CONFIGURATION - Update these values with your credentials
# ============================================================================

CONFIG = {
    # Required: OpenRouter API key for LLM access
    'openrouter_api_key': 'your_openrouter_api_key_here',

    # Required: Kaggle credentials for competition data access
    'kaggle_username': 'your_kaggle_username',
    'kaggle_key': 'your_kaggle_api_key',

    # Model selection
    'planner_model': 'meta-llama/llama-3.3-70b-instruct:free',
    'executor_model': 'Qwen/Qwen2.5-Coder-32B-Instruct',
    'verifier_model': 'Qwen/Qwen2.5-Coder-14B-Instruct',

    # Workflow settings
    'max_iterations': 5,
    'parallel_strategies': 3,

    # Resource limits
    'max_gpu_memory_gb': 28.0,
    'max_execution_time': 300,
    'max_memory_mb': 4096,

    # Monitoring
    'enable_monitoring': True,
    'logs_dir': './logs',
    'metrics_dir': './metrics',

    # Security
    'enable_sandbox': True,
    'allow_network': False,
    'allow_file_io': False,

    # Development
    'debug': False,
    'log_level': 'INFO',
}

# Competition name
COMPETITION_NAME = "mws-ai-agents-2026"


# ============================================================================
# Main execution
# ============================================================================

async def main():
    """Run Kaggle competition with full automation."""
    print("=" * 70)
    print("MLE-STAR: Kaggle Competition Automation")
    print("Full automation with direct configuration!")
    print("=" * 70)
    print()

    # Validate configuration
    if CONFIG['openrouter_api_key'] == 'your_openrouter_api_key_here':
        print("❌ Error: Please update CONFIG['openrouter_api_key'] in the script")
        print("   Get your key from: https://openrouter.ai/keys")
        return 1

    # Enhanced Kaggle credential validation
    kaggle_user = CONFIG.get('kaggle_username', '')
    kaggle_key = CONFIG.get('kaggle_key', '')

    # Check for placeholder values (case-insensitive)
    placeholder_patterns = ['your_kaggle', 'placeholder', 'replace', 'todo', 'xxx', 'example']
    if any(pattern in kaggle_user.lower() for pattern in placeholder_patterns):
        print("❌ Error: Kaggle username appears to be a placeholder value")
        print(f"   Current value: {kaggle_user}")
        print("   Please replace with your actual Kaggle username")
        print()
        print("   Get credentials: https://www.kaggle.com/settings → API → Create New Token")
        return 1

    if any(pattern in kaggle_key.lower() for pattern in placeholder_patterns):
        print("❌ Error: Kaggle API key appears to be a placeholder value")
        print("   Please replace with your actual Kaggle API key")
        print()
        print("   Get credentials: https://www.kaggle.com/settings → API → Create New Token")
        return 1

    # Check for empty/missing credentials
    if not kaggle_user or not kaggle_user.strip():
        print("❌ Error: Kaggle username is empty or not set")
        print("   Please set CONFIG['kaggle_username'] to your Kaggle username")
        print()
        print("   Get credentials: https://www.kaggle.com/settings → API → Create New Token")
        return 1

    if not kaggle_key or not kaggle_key.strip():
        print("❌ Error: Kaggle API key is empty or not set")
        print("   Please set CONFIG['kaggle_key'] to your Kaggle API key")
        print()
        print("   Get credentials: https://www.kaggle.com/settings → API → Create New Token")
        return 1

    # Check credential format (Kaggle keys are 40 characters)
    if len(kaggle_key) != 40:
        print("⚠️  Warning: Kaggle API key should be 40 characters long")
        print(f"   Current length: {len(kaggle_key)} characters")
        print("   Your key might be incorrect - please verify it from:")
        print("   https://www.kaggle.com/settings → API → Create New Token")
        print()
        # Don't return - let it try anyway in case format changed

    print(f"✓ Using Kaggle credentials for user: {kaggle_user}")
    print()

    print(f"Competition: {COMPETITION_NAME}")
    print()

    # Create configuration object directly
    print("Creating configuration...")
    config = Config(
        # API Keys
        openrouter_api_key=CONFIG['openrouter_api_key'],
        kaggle_username=CONFIG['kaggle_username'],
        kaggle_key=CONFIG['kaggle_key'],

        # Model configurations
        planner_model={
            'model_name': CONFIG['planner_model'],
            'temperature': 0.8,
            'max_tokens': 2000,
        },
        executor_model={
            'model_name': CONFIG['executor_model'],
            'temperature': 0.2,
            'max_tokens': 4000,
            'load_in_4bit': True,
            'estimated_memory_gb': 10.0,
        },
        verifier_model={
            'model_name': CONFIG['verifier_model'],
            'temperature': 0.1,
            'max_tokens': 1500,
            'load_in_4bit': True,
            'estimated_memory_gb': 4.0,
        },

        # Workflow settings
        max_iterations=CONFIG['max_iterations'],
        parallel_strategies=CONFIG['parallel_strategies'],

        # Resource limits
        max_gpu_memory_gb=CONFIG['max_gpu_memory_gb'],
        max_execution_time=CONFIG['max_execution_time'],
        max_memory_mb=CONFIG['max_memory_mb'],

        # Monitoring
        enable_monitoring=CONFIG['enable_monitoring'],
        metrics_dir=Path(CONFIG['metrics_dir']),
        logs_dir=Path(CONFIG['logs_dir']),

        # Security
        enable_sandbox=CONFIG['enable_sandbox'],
        allow_network=CONFIG['allow_network'],
        allow_file_io=CONFIG['allow_file_io'],

        # Development
        debug=CONFIG['debug'],
        log_level=CONFIG['log_level'],
    )

    print("✓ Configuration created")
    print()

    # Initialize client with explicit configuration
    print("Initializing MLE-STAR client...")
    print()

    try:
        # Create client with explicit config
        client = MLEStarClient(config)

        # Initialize (loads models)
        await client.initialize()

        print("✓ Client initialized")
        print()

        # Execute with full automation!
        print("=" * 70)
        print("Executing STAR Workflow for Kaggle Competition")
        print("=" * 70)
        print()
        print("Full automation will:")
        print("  1. Download competition data to cache (~/.cache/mle_star/competitions/)")
        print("  2. Fetch competition metadata from Kaggle API")
        print("  3. Auto-detect evaluation metric")
        print("  4. Auto-detect submission format")
        print("  5. Profile the dataset (columns, types, missing values, sample data)")
        print("  6. Generate rich task description automatically")
        print("  7. Pass structured context to agents")
        print()
        print("This may take 15-30 minutes depending on:")
        print("  - Data download time (first run only)")
        print("  - Number of iterations")
        print("  - Model complexity")
        print("  - Dataset size")
        print()

        # Minimal API - just competition name!
        # Credentials are passed from config automatically
        result = await client.execute_kaggle_competition(COMPETITION_NAME)

        # Display results
        print()
        print("=" * 70)
        print("Competition Results")
        print("=" * 70)
        print()

        print(f"Status: {result['status'].upper()}")
        print(f"Iterations: {result.get('iterations', 'N/A')}")
        print()

        if result['status'] in ['success', 'partial_success']:
            best_result = result.get('result', {})
            verification = best_result.get('verification', {})

            score = verification.get('score', None)
            if score is not None:
                print(f"Validation Score: {score:.4f}")
            print()

            # Check if submission file was created
            data_cache = Path.home() / '.cache' / 'mle_star' / 'competitions' / COMPETITION_NAME
            submission_path = data_cache / f"submission_{COMPETITION_NAME}.csv"
            alt_submission_path = data_cache / "submission.csv"

            found_submission = None
            if submission_path.exists():
                found_submission = submission_path
            elif alt_submission_path.exists():
                found_submission = alt_submission_path

            if found_submission:
                print(f"✓ Submission file created: {found_submission}")
                print()

                # Show first few lines
                try:
                    with open(found_submission, 'r') as f:
                        lines = f.readlines()[:6]
                    print("Submission Preview:")
                    print("-" * 70)
                    for line in lines:
                        print(line.strip())
                    print("-" * 70)
                    print()
                except Exception as e:
                    print(f"Could not preview submission: {e}")
                    print()

            else:
                print("⚠ Submission file not found")
                print("  Expected locations:")
                print(f"    - {submission_path}")
                print(f"    - {alt_submission_path}")
                print()

            # Show generated code preview
            code = best_result.get('code', '')
            if code:
                print("Generated Code Preview (first 1500 chars):")
                print("-" * 70)
                print(code[:1500])
                if len(code) > 1500:
                    print(f"\n... ({len(code) - 1500} more characters)")
                print("-" * 70)
                print()

            # Show verification feedback
            feedback = verification.get('feedback', {})
            if feedback:
                print("Verification Feedback:")
                print()

                strengths = feedback.get('strengths', [])
                if strengths:
                    print("Strengths:")
                    for strength in strengths:
                        print(f"  ✓ {strength}")
                    print()

                weaknesses = feedback.get('weaknesses', [])
                if weaknesses:
                    print("Areas for Improvement:")
                    for weakness in weaknesses:
                        print(f"  • {weakness}")
                    print()

                suggestions = feedback.get('suggestions', [])
                if suggestions:
                    print("Suggestions for Next Iteration:")
                    for suggestion in suggestions:
                        print(f"  → {suggestion}")
                    print()

            # Next steps
            print("=" * 70)
            print("Next Steps")
            print("=" * 70)
            print()
            print("1. Review the generated code and submission file")
            print("2. Validate submission format matches sample_submission.csv")
            print("3. Check predictions are reasonable (non-negative, realistic range)")
            print("4. Submit to Kaggle:")
            print(f"   kaggle competitions submit -c {COMPETITION_NAME} \\")
            print(f"     -f {found_submission if found_submission else 'submission.csv'} \\")
            print("     -m 'MLE-STAR generated submission'")
            print("5. Check leaderboard score and compare with local validation score")
            print("6. Iterate: refine features, try ensemble, tune hyperparameters")
            print()

        else:
            print("✗ Competition task failed")
            print(f"Reason: {result.get('reason', 'unknown')}")
            print()
            print("Troubleshooting:")
            print("- Verify Kaggle credentials are correct")
            print("- Check data files exist in data directory")
            print("- Review error logs for specific issues")
            print()

        # Cleanup
        await client.cleanup()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print()
    print("Competition: https://www.kaggle.com/competitions/mws-ai-agents-2026")
    print()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
