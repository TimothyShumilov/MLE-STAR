"""
Quickstart example for MLE-STAR framework.

This example demonstrates how to use MLE-STAR to solve a simple
machine learning task: training a classifier on the Iris dataset.

Usage:
    python examples/quickstart.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mle_star.api.client import MLEStarClient
from mle_star.tasks.task import Task, TaskType


async def main():
    """Run quickstart example."""
    print("=" * 70)
    print("MLE-STAR Quickstart Example")
    print("=" * 70)
    print()

    # Define the task
    task = Task(
        description="""
        Train a classification model on the Iris dataset.

        Requirements:
        1. Load the Iris dataset (use sklearn.datasets.load_iris)
        2. Split into train/test sets (80/20 split)
        3. Train a RandomForest classifier
        4. Evaluate on test set
        5. Print accuracy and classification report

        Target: Achieve >0.95 accuracy on test set
        """,
        task_type=TaskType.CLASSIFICATION,
        success_criteria=[
            "Accuracy > 0.95 on test set",
            "Code runs without errors",
            "Uses proper train/test split"
        ],
        target_metric="accuracy",
        baseline_score=0.90
    )

    print("Task Definition:")
    print(f"  Type: {task.task_type.value}")
    print(f"  Description: {task.description.strip()[:100]}...")
    print(f"  Success Criteria: {len(task.success_criteria)} criteria")
    print()

    # Create client from environment variables
    print("Initializing MLE-STAR client...")
    print("  (This will load models - may take 30-60 seconds on first run)")
    print()

    try:
        async with MLEStarClient.from_env() as client:
            print("✓ Client initialized")
            print()

            # Show configuration
            print("Configuration:")
            print(f"  Max Iterations: {client.config.max_iterations}")
            print(f"  Parallel Strategies: {client.config.parallel_strategies}")
            print(f"  Max GPU Memory: {client.config.max_gpu_memory_gb}GB")
            print()

            # Execute task
            print("=" * 70)
            print("Executing STAR Workflow")
            print("=" * 70)
            print()

            result = await client.execute_task(task)

            # Display results
            print()
            print("=" * 70)
            print("Results")
            print("=" * 70)
            print()

            print(f"Status: {result['status'].upper()}")
            print(f"Iterations: {result.get('iterations', 'N/A')}")
            print()

            if result['status'] == 'success':
                best_result = result.get('result', {})
                verification = best_result.get('verification', {})
                execution = best_result.get('execution', {})

                print("✓ Task Completed Successfully!")
                print()
                print(f"Final Score: {verification.get('score', 0):.2f}")
                print(f"Verification Status: {verification.get('status', 'unknown')}")
                print()

                # Show code
                code = best_result.get('code', '')
                if code:
                    print("Generated Code:")
                    print("-" * 70)
                    print(code[:500])
                    if len(code) > 500:
                        print(f"\n... ({len(code) - 500} more characters)")
                    print("-" * 70)
                    print()

                # Show execution output
                output = execution.get('output', '')
                if output:
                    print("Execution Output:")
                    print("-" * 70)
                    print(output[:500])
                    if len(output) > 500:
                        print(f"\n... ({len(output) - 500} more characters)")
                    print("-" * 70)
                    print()

                # Show feedback
                feedback = verification.get('feedback', {})
                if feedback:
                    strengths = feedback.get('strengths', [])
                    if strengths:
                        print("Strengths:")
                        for strength in strengths[:3]:
                            print(f"  ✓ {strength}")
                        print()

                    suggestions = feedback.get('suggestions', [])
                    if suggestions:
                        print("Suggestions for Improvement:")
                        for suggestion in suggestions[:3]:
                            print(f"  • {suggestion}")
                        print()

            elif result['status'] == 'partial_success':
                print("⚠ Task Partially Completed")
                print(f"Reason: {result.get('reason', 'unknown')}")
                print()
                print("Best attempt was made, but success criteria not fully met.")

            else:
                print("✗ Task Failed")
                print(f"Reason: {result.get('reason', 'unknown')}")
                print()

            # Show statistics
            print("=" * 70)
            print("Statistics")
            print("=" * 70)
            print()

            stats = client.get_stats()

            if 'workflow' in stats:
                workflow_stats = stats['workflow']
                planner_stats = workflow_stats.get('planner_stats', {})
                executor_stats = workflow_stats.get('executor_stats', {})
                verifier_stats = workflow_stats.get('verifier_stats', {})

                print("Agent Activity:")
                print(f"  Planner: {planner_stats.get('messages_processed', 0)} messages")
                print(f"  Executor: {executor_stats.get('messages_processed', 0)} messages")
                print(f"  Verifier: {verifier_stats.get('messages_processed', 0)} messages")
                print()

            if 'models' in stats:
                model_stats = stats['models']
                print("GPU Memory:")
                print(f"  Used: {model_stats.get('total_memory_gb', 0):.2f}GB")
                print(f"  Max: {model_stats.get('max_memory_gb', 0):.2f}GB")
                print(f"  Loaded Models: {model_stats.get('num_loaded', 0)}")
                print()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 70)
    print("Quickstart Complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
