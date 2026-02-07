"""
Custom ML task example for MLE-STAR framework.

This example demonstrates how to define and execute a custom
machine learning task with specific requirements and constraints.

Usage:
    python examples/custom_ml_task.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mle_star.api.client import MLEStarClient
from mle_star.tasks.task import MLTask, TaskType


async def main():
    """Run custom ML task example."""
    print("=" * 70)
    print("MLE-STAR Custom ML Task Example")
    print("=" * 70)
    print()

    # Define a custom ML task with detailed specifications
    task = MLTask(
        description="""
        Build a time series forecasting model for stock price prediction.

        Objective:
        Create a model that predicts the next day's closing price
        for a stock based on historical data.

        Data Requirements:
        - Use yfinance to download stock data (AAPL ticker)
        - Date range: Last 2 years
        - Features: Open, High, Low, Close, Volume

        Feature Engineering:
        - Create technical indicators (SMA, EMA, RSI, MACD)
        - Add lagged features (previous 5 days)
        - Calculate rolling statistics

        Model Requirements:
        - Use LSTM or GRU neural network
        - Implement proper train/validation/test split (60/20/20)
        - Use time series cross-validation
        - Track training loss and validation loss

        Evaluation:
        - Use RMSE and MAE metrics
        - Create visualization comparing predictions vs actual
        - Calculate directional accuracy (up/down prediction)

        Output:
        - Save trained model
        - Generate prediction plot
        - Print evaluation metrics
        """,
        task_type=TaskType.TIME_SERIES,

        # ML-specific parameters
        dataset_info={
            'source': 'yfinance',
            'ticker': 'AAPL',
            'date_range': '2y'
        },
        feature_columns=[
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'EMA_20', 'RSI', 'MACD'
        ],
        target_column='Close',
        test_size=0.2,
        random_state=42,

        # Success criteria
        success_criteria=[
            "RMSE < 5.0 on test set",
            "MAE < 3.0 on test set",
            "Directional accuracy > 0.55",
            "Code runs without errors",
            "Model saved successfully",
            "Visualization created"
        ],

        # Constraints
        constraints={
            'max_training_time': 600,  # 10 minutes
            'required_libraries': [
                'yfinance', 'tensorflow', 'keras',
                'pandas', 'numpy', 'matplotlib', 'sklearn'
            ],
            'output_files': ['model.h5', 'predictions.png']
        },

        target_metric='RMSE',
        baseline_score=10.0  # Baseline RMSE to beat
    )

    print("Task Configuration:")
    print(f"  Type: {task.task_type.value}")
    print(f"  Target Metric: {task.target_metric}")
    print(f"  Baseline: {task.baseline_score}")
    print(f"  Success Criteria: {len(task.success_criteria)}")
    print(f"  Features: {len(task.feature_columns)}")
    print()

    print("Success Criteria:")
    for i, criterion in enumerate(task.success_criteria, 1):
        print(f"  {i}. {criterion}")
    print()

    # Initialize client
    print("Initializing MLE-STAR client...")
    print()

    try:
        async with MLEStarClient.from_env() as client:
            print("✓ Client initialized")
            print()

            # Execute task
            print("=" * 70)
            print("Executing Custom ML Task")
            print("=" * 70)
            print()
            print("This may take several minutes as the workflow:")
            print("  1. Generates multiple solution strategies")
            print("  2. Implements and tests each approach")
            print("  3. Iteratively refines based on results")
            print()

            result = await client.execute_task(task)

            # Display results
            print()
            print("=" * 70)
            print("Task Results")
            print("=" * 70)
            print()

            print(f"Status: {result['status'].upper()}")
            print(f"Iterations Used: {result.get('iterations', 'N/A')}")
            print(f"Task ID: {result.get('task_id', 'N/A')[:16]}...")
            print()

            if result['status'] in ['success', 'partial_success']:
                best_result = result.get('result', {})
                verification = best_result.get('verification', {})
                execution = best_result.get('execution', {})

                # Overall score
                score = verification.get('score', 0)
                print(f"Overall Score: {score:.2f}/1.00")
                print(f"Status: {verification.get('status', 'unknown')}")
                print()

                # Criteria met
                criteria_met = verification.get('criteria_met', {})
                if criteria_met:
                    print("Success Criteria Status:")
                    for criterion, met in criteria_met.items():
                        status = "✓" if met else "✗"
                        print(f"  {status} {criterion}")
                    print()

                # Metrics
                metrics = verification.get('metrics', {})
                if metrics:
                    print("Metrics:")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value}")
                    print()

                # Execution info
                exec_time = execution.get('execution_time', 0)
                print(f"Execution Time: {exec_time:.1f}s")
                print()

                # Issues found
                issues = verification.get('issues', [])
                if issues:
                    print("Issues Detected:")
                    for issue in issues:
                        severity = issue.get('severity', 'unknown')
                        desc = issue.get('description', 'No description')
                        print(f"  [{severity.upper()}] {desc}")
                    print()

                # Feedback
                feedback = verification.get('feedback', {})
                if feedback:
                    print("=" * 70)
                    print("Detailed Feedback")
                    print("=" * 70)
                    print()

                    strengths = feedback.get('strengths', [])
                    if strengths:
                        print("✓ Strengths:")
                        for strength in strengths:
                            print(f"    • {strength}")
                        print()

                    weaknesses = feedback.get('weaknesses', [])
                    if weaknesses:
                        print("⚠ Weaknesses:")
                        for weakness in weaknesses:
                            print(f"    • {weakness}")
                        print()

                    suggestions = feedback.get('suggestions', [])
                    if suggestions:
                        print("→ Suggestions:")
                        for suggestion in suggestions:
                            print(f"    • {suggestion}")
                        print()

                # Code preview
                code = best_result.get('code', '')
                if code:
                    print("=" * 70)
                    print("Generated Code Preview")
                    print("=" * 70)
                    print()

                    # Show first 50 lines
                    lines = code.split('\n')[:50]
                    print('\n'.join(lines))

                    if len(code.split('\n')) > 50:
                        print(f"\n... ({len(code.split('\n')) - 50} more lines)")
                    print()

            else:
                print("✗ Task Failed")
                print(f"Reason: {result.get('reason', 'unknown')}")
                print()

            # Statistics
            print("=" * 70)
            print("Execution Statistics")
            print("=" * 70)
            print()

            stats = client.get_stats()

            if 'workflow' in stats:
                wf = stats['workflow']
                print("Workflow:")
                print(f"  Max Iterations: {wf.get('max_iterations', 'N/A')}")
                print(f"  Strategies per Iteration: {wf.get('parallel_strategies', 'N/A')}")
                print()

            if 'models' in stats:
                models = stats['models']
                print("GPU Memory:")
                print(f"  Total Used: {models.get('total_memory_gb', 0):.2f}GB")
                print(f"  Available: {models.get('available_memory_gb', 0):.2f}GB")
                print(f"  Loaded Models: {', '.join(models.get('loaded_models', []))}")
                print()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 70)
    print("Custom Task Example Complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
