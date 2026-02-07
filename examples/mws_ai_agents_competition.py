"""
MWS AI Agents 2026 Kaggle Competition Example.

This example demonstrates using MLE-STAR for the mws-ai-agents-2026 competition,
a regression task to predict rental property occupancy days.

Competition: https://www.kaggle.com/competitions/mws-ai-agents-2026/overview
Task Type: Regression
Metric: MSE (Mean Squared Error)

Usage:
    python examples/mws_ai_agents_competition.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mle_star.api.client import MLEStarClient


async def main():
    """Run MWS AI Agents 2026 competition with auto-enrichment."""
    print("=" * 70)
    print("MLE-STAR: MWS AI Agents 2026 Competition")
    print("Rental Property Occupancy Prediction (Regression)")
    print("Using automatic competition info retrieval")
    print("=" * 70)
    print()

    # Configuration
    competition_name = "mws-ai-agents-2026"
    data_dir = Path("./data/mws_ai_agents")
    metric = "mse"  # Mean Squared Error

    print(f"Competition: {competition_name}")
    print(f"Data Directory: {data_dir}")
    print(f"Metric: {metric} (lower is better)")
    print()

    # Check if data directory exists
    if not data_dir.exists():
        print(f"⚠ Data directory not found: {data_dir}")
        print()
        print("To run this example:")
        print("1. Download competition data from Kaggle:")
        print(f"   kaggle competitions download -c {competition_name} -p {data_dir}")
        print(f"2. Unzip the data:")
        print(f"   unzip {data_dir}/{competition_name}.zip -d {data_dir}")
        print("3. Ensure you have train.csv, test.csv, and sample_submission.csv")
        print()
        print("Note: Auto-enrichment will work even without Kaggle credentials,")
        print("      using dataset profiling only.")
        print()
        return 1

    # Initialize client
    print("Initializing MLE-STAR client...")
    print()

    try:
        async with MLEStarClient.from_env() as client:
            print("✓ Client initialized")
            print()

            # Execute with auto-enrichment (no manual description needed!)
            print("=" * 70)
            print("Executing STAR Workflow for Kaggle Competition")
            print("=" * 70)
            print()
            print("Auto-enrichment will:")
            print("  1. Fetch competition metadata from Kaggle API (if credentials available)")
            print("  2. Profile the dataset (columns, types, missing values, sample data)")
            print("  3. Generate rich task description automatically")
            print("  4. Pass structured context to agents")
            print()
            print("This may take 15-30 minutes depending on:")
            print("  - Number of iterations (max 5)")
            print("  - Model complexity")
            print("  - Dataset size")
            print()

            # Option 1: Full auto-enrichment (RECOMMENDED)
            result = await client.execute_kaggle_competition(
                competition_name=competition_name,
                data_dir=data_dir,
                evaluation_metric=metric
                # That's it! No hardcoded description needed.
                # Framework automatically:
                # - Fetches competition metadata from Kaggle API
                # - Profiles dataset (13 features, types, missing%, sample data)
                # - Generates enriched task description
                # - Passes structured context to agents
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

                mse_score = verification.get('score', None)
                if mse_score is not None:
                    print(f"MSE Score: {mse_score:.4f}")
                    print(f"RMSE Score: {mse_score ** 0.5:.4f}")
                print()

                # Check if submission file was created
                submission_path = data_dir / f"submission_{competition_name}.csv"
                alt_submission_path = data_dir / "submission.csv"

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
                print(f"   kaggle competitions submit -c {competition_name} \\")
                print(f"     -f {found_submission if found_submission else 'submission.csv'} \\")
                print("     -m 'MLE-STAR generated submission'")
                print("5. Check leaderboard score and compare with local MSE")
                print("6. Iterate: refine features, try ensemble, tune hyperparameters")
                print()

            else:
                print("✗ Competition task failed")
                print(f"Reason: {result.get('reason', 'unknown')}")
                print()
                print("Troubleshooting:")
                print("- Verify data files exist in data directory")
                print("- Check data format matches expected structure")
                print("- Review error logs for specific issues")
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
    print()
    print("Competition: https://www.kaggle.com/competitions/mws-ai-agents-2026")
    print()

    return 0


# Alternative usage patterns (for reference):

# Option 2: Disable auto-enrichment (use minimal description)
async def example_minimal():
    """Example with auto-enrichment disabled."""
    async with MLEStarClient.from_env() as client:
        result = await client.execute_kaggle_competition(
            "mws-ai-agents-2026",
            Path("./data/mws_ai_agents"),
            "mse",
            auto_enrich=False  # Use minimal description
        )
        return result


# Option 3: Manual description (backward compatible)
async def example_manual_description():
    """Example with custom manual description."""
    custom_description = """
    Custom task description goes here.
    You can provide specific instructions if needed.
    """

    async with MLEStarClient.from_env() as client:
        result = await client.execute_kaggle_competition(
            "mws-ai-agents-2026",
            Path("./data/mws_ai_agents"),
            "mse",
            description=custom_description  # Overrides auto-generation
        )
        return result


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
