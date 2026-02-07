"""
Kaggle competition example for MLE-STAR framework.

This example shows how to use MLE-STAR to tackle a Kaggle competition,
specifically the Titanic survival prediction challenge.

Usage:
    python examples/kaggle_competition.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mle_star.api.client import MLEStarClient


async def main():
    """Run Kaggle competition example."""
    print("=" * 70)
    print("MLE-STAR Kaggle Competition Example")
    print("=" * 70)
    print()

    # Configuration
    competition_name = "titanic"
    data_dir = Path("./data/titanic")  # Adjust to your data directory
    evaluation_metric = "accuracy"

    print(f"Competition: {competition_name}")
    print(f"Data Directory: {data_dir}")
    print(f"Metric: {evaluation_metric}")
    print()

    # Check if data directory exists
    if not data_dir.exists():
        print(f"⚠ Warning: Data directory not found: {data_dir}")
        print()
        print("To run this example:")
        print("1. Download Kaggle competition data")
        print("2. Place train.csv, test.csv in data directory")
        print("3. Update data_dir path in this script")
        print()
        print("Continuing with example (may fail without data)...")
        print()

    # Create custom task description for better results
    description = f"""
    Solve the Kaggle {competition_name} competition.

    Task: Predict passenger survival on the Titanic.

    Requirements:
    1. Load train.csv and test.csv from {data_dir}
    2. Perform exploratory data analysis (EDA)
    3. Handle missing values appropriately
    4. Feature engineering:
       - Extract title from name
       - Create family size feature
       - Bin age into categories
       - Engineer new features as needed
    5. Train a classification model (try RandomForest, XGBoost, or ensemble)
    6. Make predictions on test set
    7. Create submission.csv in the format required by Kaggle

    Target: Achieve {evaluation_metric} > 0.80
    """

    print("Task Description:")
    print(description)
    print()

    # Initialize client
    print("Initializing MLE-STAR client...")
    print()

    try:
        async with MLEStarClient.from_env() as client:
            print("✓ Client initialized")
            print()

            # Execute Kaggle competition
            print("=" * 70)
            print("Executing STAR Workflow for Kaggle Competition")
            print("=" * 70)
            print()

            result = await client.execute_kaggle_competition(
                competition_name=competition_name,
                data_dir=data_dir,
                evaluation_metric=evaluation_metric,
                description=description
            )

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

                print(f"Score: {verification.get('score', 0):.2f}")
                print()

                # Check if submission file was created
                submission_path = data_dir / "submission.csv"
                if submission_path.exists():
                    print(f"✓ Submission file created: {submission_path}")
                    print()

                    # Show first few lines
                    try:
                        with open(submission_path, 'r') as f:
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
                    print("  The generated code may not have created submission.csv")
                    print()

                # Show generated code
                code = best_result.get('code', '')
                if code:
                    print("Generated Code (first 1000 chars):")
                    print("-" * 70)
                    print(code[:1000])
                    if len(code) > 1000:
                        print(f"\n... ({len(code) - 1000} more characters)")
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
                        print("Suggestions:")
                        for suggestion in suggestions:
                            print(f"  → {suggestion}")
                        print()

                # Next steps
                print("=" * 70)
                print("Next Steps")
                print("=" * 70)
                print()
                print("1. Review the generated code and submission file")
                print("2. Test the submission locally if possible")
                print("3. Submit to Kaggle:")
                print(f"   kaggle competitions submit -c {competition_name} -f submission.csv -m 'MLE-STAR generated'")
                print("4. Iterate based on leaderboard score")
                print()

            else:
                print("✗ Competition task failed")
                print(f"Reason: {result.get('reason', 'unknown')}")
                print()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
