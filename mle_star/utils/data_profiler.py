"""
Dataset profiling for automatic feature discovery and analysis.

This module provides lightweight dataset profiling capabilities optimized
for speed and ML task understanding. Profiles include column types, missing
values, statistics, and sample data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DatasetProfiler:
    """
    Lightweight dataset profiler for ML tasks.

    Analyzes CSV datasets to extract schema information, data quality metrics,
    and basic statistics. Optimized for speed with sampling and quick_mode.

    Performance target: < 5 seconds for typical datasets

    Examples:
        >>> profiler = DatasetProfiler()
        >>> profile = profiler.profile_dataset(Path("train.csv"))
        >>> print(f"{profile['num_rows']} rows × {profile['num_cols']} columns")
        >>> print(f"Target candidates: {profile['target_candidates']}")
    """

    def __init__(self):
        """Initialize dataset profiler."""
        pass

    def profile_dataset(
        self,
        file_path: Path,
        max_rows_sample: int = 1000,
        quick_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Profile a CSV dataset with schema and statistics.

        Loads and analyzes a CSV file to extract:
        - Dataset shape (rows × columns)
        - Column data types
        - Missing value percentages
        - Unique value counts
        - Sample values and statistics
        - Target column candidates

        Args:
            file_path: Path to CSV file
            max_rows_sample: Maximum rows to load for sampling. If dataset
                           is larger, only first N rows are loaded for speed.
                           Use -1 to load entire dataset.
            quick_mode: If True, skip expensive computations like correlation.
                       Recommended for large datasets.

        Returns:
            Dictionary with profile information:
            {
                'num_rows': int,        # Total rows (estimate if sampled)
                'num_cols': int,        # Total columns
                'columns': {
                    'column_name': {
                        'dtype': 'float64',
                        'missing_pct': 19.9,
                        'unique_values': 88,
                        'sample_values': [val1, val2, ...],
                        'stats': {'mean': X, 'std': Y, ...}  # numeric only
                    },
                    ...
                },
                'sample_data': pd.DataFrame,  # First 5 rows
                'target_candidates': ['target', 'label', ...],
                'is_sampled': bool,   # True if max_rows_sample was reached
            }

            Returns empty dict {} if profiling fails.

        Raises:
            None - all exceptions are caught and logged

        Examples:
            >>> profiler = DatasetProfiler()
            >>> profile = profiler.profile_dataset(
            ...     Path("titanic/train.csv"),
            ...     max_rows_sample=1000,
            ...     quick_mode=True
            ... )
            >>> print(profile['columns']['Age']['missing_pct'])
            19.87
        """
        try:
            logger.debug(f"Profiling dataset: {file_path}")

            # Check file exists
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return {}

            # Get total row count efficiently (without loading all data)
            total_rows = self._count_rows(file_path)
            is_sampled = total_rows > max_rows_sample > 0

            # Load data (potentially sampled)
            if max_rows_sample > 0 and total_rows > max_rows_sample:
                logger.debug(f"Sampling {max_rows_sample} of {total_rows} rows for performance")
                df = pd.read_csv(file_path, nrows=max_rows_sample)
            else:
                df = pd.read_csv(file_path)

            # Profile each column
            columns_profile = {}
            for col in df.columns:
                col_profile = self._profile_column(df[col], quick_mode=quick_mode)
                columns_profile[col] = col_profile

            # Identify potential target columns
            target_candidates = self._identify_target_candidates(df)

            # Build final profile
            profile = {
                'num_rows': total_rows,  # True count
                'num_cols': len(df.columns),
                'columns': columns_profile,
                'sample_data': df.head(5),  # First 5 rows for preview
                'target_candidates': target_candidates,
                'is_sampled': is_sampled,
            }

            logger.debug(
                f"Profiled {total_rows} rows × {len(df.columns)} columns "
                f"({'sampled' if is_sampled else 'full dataset'})"
            )

            return profile

        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse CSV file {file_path}: {e}")
            return {}

        except MemoryError:
            logger.error(
                f"Out of memory profiling {file_path}. "
                f"Try reducing max_rows_sample or using quick_mode=True"
            )
            return {}

        except Exception as e:
            logger.error(f"Failed to profile {file_path}: {e}")
            return {}

    def _count_rows(self, file_path: Path) -> int:
        """Count total rows in CSV file efficiently."""
        try:
            # Use pandas to get row count (fast with C parser)
            # nrows=1 + shape gives us count without loading all data
            return len(pd.read_csv(file_path, usecols=[0]))

        except Exception as e:
            logger.debug(f"Failed to count rows efficiently: {e}")
            # Fallback: count lines (includes header)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return sum(1 for _ in f) - 1  # Subtract header
            except Exception:
                return 0

    def _profile_column(self, series: pd.Series, quick_mode: bool = True) -> Dict[str, Any]:
        """
        Profile a single column.

        Args:
            series: Pandas Series to profile
            quick_mode: Skip expensive computations if True

        Returns:
            Column profile dictionary
        """
        profile = {
            'dtype': str(series.dtype),
            'missing_pct': (series.isna().sum() / len(series) * 100) if len(series) > 0 else 0,
            'unique_values': series.nunique(),
        }

        # Sample values (non-null)
        non_null = series.dropna()
        if len(non_null) > 0:
            sample_size = min(5, len(non_null))
            profile['sample_values'] = non_null.head(sample_size).tolist()
        else:
            profile['sample_values'] = []

        # Numeric statistics
        if pd.api.types.is_numeric_dtype(series):
            profile['stats'] = self._compute_numeric_stats(series, quick_mode)
        # Categorical statistics
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            profile['stats'] = self._compute_categorical_stats(series, quick_mode)

        return profile

    def _compute_numeric_stats(self, series: pd.Series, quick_mode: bool) -> Dict[str, Any]:
        """Compute statistics for numeric column."""
        stats = {}

        try:
            # Basic stats (fast)
            stats['mean'] = float(series.mean()) if series.notna().any() else None
            stats['std'] = float(series.std()) if series.notna().any() else None
            stats['min'] = float(series.min()) if series.notna().any() else None
            stats['max'] = float(series.max()) if series.notna().any() else None

            if not quick_mode:
                # Additional stats (slightly slower)
                stats['median'] = float(series.median()) if series.notna().any() else None
                stats['q25'] = float(series.quantile(0.25)) if series.notna().any() else None
                stats['q75'] = float(series.quantile(0.75)) if series.notna().any() else None

        except Exception as e:
            logger.debug(f"Failed to compute numeric stats: {e}")

        return stats

    def _compute_categorical_stats(self, series: pd.Series, quick_mode: bool) -> Dict[str, Any]:
        """Compute statistics for categorical column."""
        stats = {}

        try:
            # Most frequent values
            value_counts = series.value_counts()
            if len(value_counts) > 0:
                stats['most_frequent'] = value_counts.index[0]
                stats['most_frequent_count'] = int(value_counts.iloc[0])

            if not quick_mode and len(value_counts) > 1:
                # Top 3 most frequent
                top_n = min(3, len(value_counts))
                stats['top_values'] = value_counts.head(top_n).to_dict()

        except Exception as e:
            logger.debug(f"Failed to compute categorical stats: {e}")

        return stats

    def _identify_target_candidates(self, df: pd.DataFrame) -> List[str]:
        """
        Identify potential target columns based on common naming patterns.

        Args:
            df: DataFrame to analyze

        Returns:
            List of column names that are likely target variables
        """
        candidates = []

        # Common target column names (case-insensitive)
        target_names = [
            'target', 'label', 'y', 'class', 'output',
            'survived', 'sold', 'price', 'score', 'rating'
        ]

        for col in df.columns:
            col_lower = col.lower()

            # Exact match
            if col_lower in target_names:
                candidates.append(col)
                continue

            # Partial match (e.g., 'SalePrice', 'is_fraud')
            for target_name in target_names:
                if target_name in col_lower:
                    candidates.append(col)
                    break

        return candidates

    def generate_feature_description(self, profile: Dict[str, Any]) -> str:
        """
        Convert profile to human-readable feature description.

        Args:
            profile: Profile dictionary from profile_dataset()

        Returns:
            Multi-line string describing all features

        Examples:
            >>> profiler = DatasetProfiler()
            >>> profile = profiler.profile_dataset(Path("train.csv"))
            >>> desc = profiler.generate_feature_description(profile)
            >>> print(desc)
        """
        if not profile or 'columns' not in profile:
            return "No profile available"

        lines = [
            f"Dataset: {profile['num_rows']} rows × {profile['num_cols']} columns",
            ""
        ]

        if profile.get('is_sampled'):
            lines.append(f"(Analysis based on {len(profile.get('sample_data', []))} row sample)")
            lines.append("")

        lines.append("Features:")

        for col_name, col_info in profile['columns'].items():
            dtype = col_info.get('dtype', 'unknown')
            missing = col_info.get('missing_pct', 0)
            unique = col_info.get('unique_values', 0)

            line = f"  - {col_name} ({dtype}): {unique} unique, {missing:.1f}% missing"

            # Add stats hint for numeric columns
            if 'stats' in col_info and 'mean' in col_info['stats']:
                mean = col_info['stats']['mean']
                if mean is not None:
                    line += f", mean={mean:.2f}"

            lines.append(line)

        if profile.get('target_candidates'):
            lines.append("")
            lines.append(f"Potential target columns: {', '.join(profile['target_candidates'])}")

        return "\n".join(lines)

    def detect_task_type(
        self,
        profile: Dict[str, Any],
        target_col: Optional[str] = None
    ) -> str:
        """
        Infer task type (classification vs regression) from target column.

        Args:
            profile: Profile dictionary from profile_dataset()
            target_col: Name of target column. If None, uses first target_candidate

        Returns:
            'classification', 'regression', or 'unknown'

        Examples:
            >>> profiler = DatasetProfiler()
            >>> profile = profiler.profile_dataset(Path("train.csv"))
            >>> task_type = profiler.detect_task_type(profile, 'Survived')
            >>> print(task_type)  # 'classification'
        """
        if not profile or 'columns' not in profile:
            return 'unknown'

        # Determine target column
        if target_col is None:
            candidates = profile.get('target_candidates', [])
            if not candidates:
                return 'unknown'
            target_col = candidates[0]

        # Check if target column exists
        if target_col not in profile['columns']:
            return 'unknown'

        col_info = profile['columns'][target_col]
        dtype = col_info.get('dtype', '')
        unique = col_info.get('unique_values', 0)
        total_rows = profile.get('num_rows', 0)

        # Classification heuristics
        if 'int' in dtype or 'bool' in dtype or 'object' in dtype:
            # Small number of unique values relative to dataset size
            if unique < 20 or (total_rows > 0 and unique / total_rows < 0.05):
                return 'classification'

        # Regression heuristics
        if 'float' in dtype:
            return 'regression'

        # Default: if integer with many unique values, likely regression
        if 'int' in dtype and unique > 20:
            return 'regression'

        return 'unknown'
