"""Kaggle competition task adapter."""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from .task import Task, TaskType, MLTask
from mle_star.integrations.kaggle_api import KaggleAPIClient
from mle_star.utils.data_profiler import DatasetProfiler

logger = logging.getLogger(__name__)


class KaggleTask(MLTask):
    """
    Adapter for Kaggle competition tasks.

    This class extends MLTask with Kaggle-specific features:
    - Auto-detection of task type from metric
    - Standard file structure handling
    - Submission file validation
    - Competition-specific constraints

    Example:
        >>> task = KaggleTask(
        ...     competition_name="titanic",
        ...     data_dir=Path("./data"),
        ...     evaluation_metric="accuracy"
        ... )
    """

    def __init__(
        self,
        competition_name: str,
        data_dir: Path,
        evaluation_metric: str,
        description: Optional[str] = None,
        auto_enrich: bool = True,
        submission_format: str = "csv"
    ):
        """
        Initialize Kaggle task with optional auto-enrichment.

        Args:
            competition_name: Name of the Kaggle competition
            data_dir: Directory containing competition data
            evaluation_metric: Competition evaluation metric
            description: Optional task description (overrides auto-generation)
            auto_enrich: If True, automatically fetch competition metadata
                        and profile dataset. If False, use minimal description.
            submission_format: Format for submission file (default: csv)
        """
        # Store attributes early for use in helper methods
        self.competition_name = competition_name
        self.data_dir = Path(data_dir)
        self.evaluation_metric = evaluation_metric
        self.submission_format = submission_format

        # Step 1: Fetch Kaggle competition metadata (if auto_enrich=True)
        kaggle_info = {}
        if auto_enrich:
            try:
                kaggle_client = KaggleAPIClient()
                if kaggle_client.is_authenticated():
                    kaggle_info = kaggle_client.fetch_competition_metadata(competition_name)
                    if kaggle_info:
                        logger.info(f"✓ Fetched metadata for {competition_name}")
                else:
                    logger.warning(
                        "Kaggle credentials not found. "
                        "Setup ~/.kaggle/kaggle.json to enable auto-enrichment."
                    )
            except Exception as e:
                logger.warning(f"Kaggle API failed: {e}")

        # Step 2: Profile dataset (if auto_enrich=True and data exists)
        dataset_profile = {}
        if auto_enrich and self.data_dir.exists():
            try:
                profiler = DatasetProfiler()
                train_path = self.data_dir / 'train.csv'
                if train_path.exists():
                    dataset_profile = profiler.profile_dataset(
                        train_path,
                        max_rows_sample=1000,
                        quick_mode=True
                    )
                    if dataset_profile:
                        logger.info(
                            f"✓ Profiled dataset: {dataset_profile['num_rows']} rows, "
                            f"{dataset_profile['num_cols']} columns"
                        )
            except Exception as e:
                logger.warning(f"Dataset profiling failed: {e}")

        # Step 3: Generate enriched description (if not provided)
        if description is None:
            description = self._generate_enriched_description(
                competition_name,
                evaluation_metric,
                kaggle_info,
                dataset_profile
            )

        # Step 4: Auto-detect task type from metric
        task_type = self._detect_task_type(evaluation_metric)

        # Step 5: Generate success criteria
        success_criteria = [
            f"Generate valid {submission_format} submission file",
            f"Improve upon baseline {evaluation_metric}",
            "Code executes without errors",
            "All competition requirements met"
        ]

        # Step 6: Populate dataset_info for agents
        dataset_info = {
            'competition_metadata': kaggle_info,
            'data_profile': dataset_profile,
            'data_files': {
                'train': str(self.data_dir / 'train.csv'),
                'test': str(self.data_dir / 'test.csv'),
                'sample_submission': str(self.data_dir / 'sample_submission.csv')
            }
        }

        # Step 7: Initialize parent MLTask with enriched data
        super().__init__(
            description=description,
            task_type=task_type,
            dataset_info=dataset_info,  # NEW - passes enriched context to agents
            constraints={
                'competition_name': competition_name,
                'evaluation_metric': evaluation_metric,
                'submission_format': submission_format
            },
            success_criteria=success_criteria,
            data_path=str(data_dir),
            target_metric=evaluation_metric
        )

    def _detect_task_type(self, metric: str) -> TaskType:
        """
        Auto-detect task type from evaluation metric.

        Args:
            metric: Evaluation metric name

        Returns:
            Detected TaskType
        """
        metric_lower = metric.lower()

        # Classification metrics
        if any(m in metric_lower for m in ['accuracy', 'auc', 'roc', 'f1', 'precision', 'recall', 'logloss']):
            return TaskType.CLASSIFICATION

        # Regression metrics
        if any(m in metric_lower for m in ['rmse', 'mae', 'mse', 'r2', 'mape']):
            return TaskType.REGRESSION

        # NLP metrics
        if any(m in metric_lower for m in ['bleu', 'rouge', 'perplexity']):
            return TaskType.NLP

        # Default to custom
        return TaskType.CUSTOM

    def _generate_enriched_description(
        self,
        competition_name: str,
        metric: str,
        kaggle_info: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> str:
        """
        Generate rich task description from metadata and profiling.

        Args:
            competition_name: Competition identifier
            metric: Evaluation metric
            kaggle_info: Competition metadata from Kaggle API
            profile: Dataset profile from DatasetProfiler

        Returns:
            Multi-line task description with competition and dataset context
        """
        parts = [
            f"Kaggle Competition: {competition_name}",
            f"Objective: {kaggle_info.get('title', 'Optimize ' + metric)}",
            ""
        ]

        # Add competition overview if available
        if 'description' in kaggle_info and kaggle_info['description']:
            # Clean HTML/Markdown and truncate
            desc = kaggle_info['description'].replace('\n', ' ').strip()
            if len(desc) > 300:
                desc = desc[:300] + "..."
            parts.extend([
                "Competition Overview:",
                desc,
                ""
            ])

        # Add dataset structure
        if profile and 'num_rows' in profile:
            parts.extend([
                "Dataset Structure:",
                f"- Rows: {profile.get('num_rows', 'Unknown')}",
                f"- Columns: {profile.get('num_cols', 'Unknown')}",
                ""
            ])

            # Add feature details
            if 'columns' in profile and profile['columns']:
                parts.append("Features:")
                for col_name, col_info in profile['columns'].items():
                    dtype = col_info.get('dtype', 'unknown')
                    unique = col_info.get('unique_values', 0)
                    missing = col_info.get('missing_pct', 0)

                    parts.append(
                        f"  - {col_name} ({dtype}): "
                        f"{unique} unique, {missing:.1f}% missing"
                    )

                parts.append("")

        # Add evaluation info
        parts.extend([
            f"Evaluation Metric: {metric}",
        ])

        if 'evaluation' in kaggle_info and kaggle_info['evaluation']:
            parts.append(kaggle_info['evaluation'])

        parts.extend([
            "",
            "Task:",
            "1. Load and explore the dataset",
            "2. Perform feature engineering and preprocessing",
            "3. Train and validate models",
            "4. Generate predictions for test set",
            "5. Create submission file matching sample_submission.csv format"
        ])

        return "\n".join(parts)

    def get_data_files(self) -> Dict[str, Path]:
        """
        Get standard Kaggle data file paths.

        Returns:
            Dictionary mapping file types to paths

        Example:
            >>> files = task.get_data_files()
            >>> print(files['train'])
            Path('./data/train.csv')
        """
        return {
            'train': self.data_dir / 'train.csv',
            'test': self.data_dir / 'test.csv',
            'sample_submission': self.data_dir / 'sample_submission.csv'
        }

    def validate_data_files(self) -> Dict[str, bool]:
        """
        Validate that required data files exist.

        Returns:
            Dictionary of file existence status
        """
        files = self.get_data_files()
        return {
            name: path.exists()
            for name, path in files.items()
        }

    def get_submission_path(self, output_dir: Optional[Path] = None) -> Path:
        """
        Get path for submission file.

        Args:
            output_dir: Optional output directory (defaults to data_dir)

        Returns:
            Path for submission file
        """
        if output_dir is None:
            output_dir = self.data_dir

        return output_dir / f"submission.{self.submission_format}"

    def validate_submission(self, submission_path: Path) -> Dict[str, Any]:
        """
        Validate submission file format.

        Args:
            submission_path: Path to submission file

        Returns:
            Validation result dictionary
        """
        validation = {
            'valid': False,
            'exists': False,
            'correct_format': False,
            'errors': []
        }

        # Check file exists
        if not submission_path.exists():
            validation['errors'].append(f"File not found: {submission_path}")
            return validation

        validation['exists'] = True

        # Check file format
        if submission_path.suffix.lstrip('.') != self.submission_format:
            validation['errors'].append(
                f"Wrong format: expected {self.submission_format}, "
                f"got {submission_path.suffix}"
            )
            return validation

        validation['correct_format'] = True

        # Additional validation for CSV
        if self.submission_format == 'csv':
            try:
                import pandas as pd

                # Read submission
                submission = pd.read_csv(submission_path)

                # Read sample submission for comparison
                sample_path = self.data_dir / 'sample_submission.csv'
                if sample_path.exists():
                    sample = pd.read_csv(sample_path)

                    # Check columns match
                    if list(submission.columns) != list(sample.columns):
                        validation['errors'].append(
                            f"Column mismatch. Expected: {list(sample.columns)}, "
                            f"Got: {list(submission.columns)}"
                        )
                        return validation

                    # Check row count
                    if len(submission) != len(sample):
                        validation['errors'].append(
                            f"Row count mismatch. Expected: {len(sample)}, "
                            f"Got: {len(submission)}"
                        )
                        return validation

                # Check for missing values
                if submission.isnull().any().any():
                    validation['errors'].append("Submission contains missing values")
                    return validation

                validation['valid'] = True

            except Exception as e:
                validation['errors'].append(f"Error reading submission: {e}")
                return validation

        else:
            # For non-CSV formats, just check existence and format
            validation['valid'] = True

        return validation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including Kaggle-specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'competition_name': self.competition_name,
            'evaluation_metric': self.evaluation_metric,
            'submission_format': self.submission_format
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KaggleTask':
        """Create KaggleTask from dictionary."""
        return cls(
            competition_name=data['competition_name'],
            data_dir=Path(data['data_path']),
            evaluation_metric=data['evaluation_metric'],
            description=data.get('description'),
            submission_format=data.get('submission_format', 'csv')
        )

    def __str__(self) -> str:
        """String representation."""
        return (
            f"KaggleTask("
            f"competition={self.competition_name}, "
            f"metric={self.evaluation_metric})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"KaggleTask("
            f"competition_name='{self.competition_name}', "
            f"data_dir='{self.data_dir}', "
            f"evaluation_metric='{self.evaluation_metric}', "
            f"type={self.task_type}"
            f")"
        )
