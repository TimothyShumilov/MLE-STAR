"""
Unit tests for Kaggle API integration and data profiling.

Tests cover:
- KaggleAPIClient authentication and metadata fetching
- DatasetProfiler profiling and feature detection
- KaggleTask auto-enrichment workflow
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import tempfile
import json

from mle_star.integrations.kaggle_api import KaggleAPIClient
from mle_star.utils.data_profiler import DatasetProfiler
from mle_star.tasks.kaggle_task import KaggleTask


class TestKaggleAPIClient:
    """Test KaggleAPIClient functionality."""

    def test_initialization_without_credentials(self):
        """Test client initialization when credentials are missing."""
        with patch('mle_star.integrations.kaggle_api.KaggleApi') as mock_api:
            mock_api.return_value.authenticate.side_effect = OSError("Credentials not found")

            client = KaggleAPIClient()

            assert not client.is_authenticated()
            assert client._api is None

    def test_initialization_with_credentials(self):
        """Test client initialization with valid credentials."""
        with patch('mle_star.integrations.kaggle_api.KaggleApi') as mock_api:
            mock_instance = Mock()
            mock_api.return_value = mock_instance

            client = KaggleAPIClient()

            mock_instance.authenticate.assert_called_once()

    def test_fetch_metadata_unauthenticated(self):
        """Test metadata fetch returns empty dict when not authenticated."""
        with patch('mle_star.integrations.kaggle_api.KaggleApi') as mock_api:
            mock_api.return_value.authenticate.side_effect = OSError("No credentials")

            client = KaggleAPIClient()
            metadata = client.fetch_competition_metadata("titanic")

            assert metadata == {}

    def test_fetch_metadata_success(self):
        """Test successful metadata fetch."""
        with patch('mle_star.integrations.kaggle_api.KaggleApi') as mock_api:
            # Setup mock
            mock_instance = Mock()
            mock_api.return_value = mock_instance

            # Mock competition object
            mock_comp = Mock()
            mock_comp.title = "Titanic - Machine Learning from Disaster"
            mock_comp.description = "Predict survival on the Titanic"
            mock_comp.evaluationMetric = "Accuracy"
            mock_comp.reward = "$100"
            mock_comp.tags = ['binary classification']
            mock_comp.maxTeamSize = 5

            mock_instance.competition_view.return_value = mock_comp

            client = KaggleAPIClient()
            metadata = client.fetch_competition_metadata("titanic", use_cache=False)

            assert metadata['title'] == "Titanic - Machine Learning from Disaster"
            assert metadata['description'] == "Predict survival on the Titanic"
            assert metadata['evaluation'] == "Accuracy"
            assert 'url' in metadata

    def test_fetch_metadata_with_exception(self):
        """Test metadata fetch handles exceptions gracefully."""
        with patch('mle_star.integrations.kaggle_api.KaggleApi') as mock_api:
            mock_instance = Mock()
            mock_api.return_value = mock_instance
            mock_instance.competition_view.side_effect = Exception("API Error")

            client = KaggleAPIClient()
            metadata = client.fetch_competition_metadata("invalid-comp")

            assert metadata == {}

    def test_cache_mechanism(self, tmp_path):
        """Test that caching works correctly."""
        with patch('mle_star.integrations.kaggle_api.KaggleApi') as mock_api:
            mock_instance = Mock()
            mock_api.return_value = mock_instance

            mock_comp = Mock()
            mock_comp.title = "Test Competition"
            mock_comp.description = "Test description"
            mock_instance.competition_view.return_value = mock_comp

            client = KaggleAPIClient(cache_dir=tmp_path)

            # First call - should hit API
            metadata1 = client.fetch_competition_metadata("test-comp", use_cache=False)

            # Second call - should use cache
            metadata2 = client.fetch_competition_metadata("test-comp", use_cache=True)

            # API should only be called once
            assert mock_instance.competition_view.call_count == 1
            assert metadata1 == metadata2

    def test_clear_cache(self, tmp_path):
        """Test cache clearing."""
        client = KaggleAPIClient(cache_dir=tmp_path)

        # Create a fake cache file
        cache_file = tmp_path / "test_cache.json"
        cache_file.write_text('{"test": "data"}')

        assert cache_file.exists()

        client.clear_cache()

        # All cache files should be deleted
        assert not cache_file.exists()


class TestDatasetProfiler:
    """Test DatasetProfiler functionality."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing."""
        csv_path = tmp_path / "test_data.csv"

        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'age': [25, 30, None, 35, 40],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [85.5, 90.0, 78.5, 92.0, 88.5],
            'target': [0, 1, 0, 1, 1]
        })

        df.to_csv(csv_path, index=False)
        return csv_path

    def test_profile_dataset_basic(self, sample_csv):
        """Test basic dataset profiling."""
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(sample_csv, quick_mode=True)

        assert profile['num_rows'] == 5
        assert profile['num_cols'] == 5
        assert 'columns' in profile
        assert 'age' in profile['columns']
        assert 'sample_data' in profile

    def test_profile_missing_values(self, sample_csv):
        """Test that missing values are detected correctly."""
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(sample_csv)

        age_profile = profile['columns']['age']

        assert age_profile['missing_pct'] == 20.0  # 1 out of 5 is missing

    def test_profile_numeric_stats(self, sample_csv):
        """Test numeric column statistics."""
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(sample_csv, quick_mode=False)

        score_profile = profile['columns']['score']

        assert 'stats' in score_profile
        assert 'mean' in score_profile['stats']
        assert 'std' in score_profile['stats']
        assert score_profile['stats']['mean'] == pytest.approx(86.9, rel=0.1)

    def test_profile_categorical_stats(self, sample_csv):
        """Test categorical column statistics."""
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(sample_csv, quick_mode=False)

        name_profile = profile['columns']['name']

        assert 'stats' in name_profile
        assert 'most_frequent' in name_profile['stats']

    def test_identify_target_candidates(self, sample_csv):
        """Test target column identification."""
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(sample_csv)

        assert 'target_candidates' in profile
        assert 'target' in profile['target_candidates']

    def test_detect_task_type_classification(self, sample_csv):
        """Test task type detection for classification."""
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(sample_csv)

        task_type = profiler.detect_task_type(profile, 'target')

        assert task_type == 'classification'

    def test_detect_task_type_regression(self, tmp_path):
        """Test task type detection for regression."""
        csv_path = tmp_path / "regression_data.csv"

        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [1.5, 2.7, 3.2, 4.8, 5.1]  # Float target
        })

        df.to_csv(csv_path, index=False)

        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(csv_path)
        task_type = profiler.detect_task_type(profile, 'y')

        assert task_type == 'regression'

    def test_profile_nonexistent_file(self):
        """Test profiling handles nonexistent files gracefully."""
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(Path("/nonexistent/file.csv"))

        assert profile == {}

    def test_profile_sampling(self, tmp_path):
        """Test that sampling works for large datasets."""
        csv_path = tmp_path / "large_data.csv"

        # Create dataset with 2000 rows
        df = pd.DataFrame({
            'x': range(2000),
            'y': range(2000)
        })

        df.to_csv(csv_path, index=False)

        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(csv_path, max_rows_sample=1000)

        assert profile['num_rows'] == 2000  # True count
        assert profile['is_sampled'] == True

    def test_generate_feature_description(self, sample_csv):
        """Test feature description generation."""
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(sample_csv)

        description = profiler.generate_feature_description(profile)

        assert '5 rows × 5 columns' in description
        assert 'age' in description
        assert 'target' in description


class TestKaggleTaskEnrichment:
    """Test KaggleTask auto-enrichment functionality."""

    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create a mock data directory with CSV files."""
        data_dir = tmp_path / "kaggle_data"
        data_dir.mkdir()

        # Create mock train.csv
        train_df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature': [10, 20, 30],
            'target': [0, 1, 0]
        })
        train_df.to_csv(data_dir / "train.csv", index=False)

        # Create mock test.csv
        test_df = pd.DataFrame({
            'id': [4, 5],
            'feature': [40, 50]
        })
        test_df.to_csv(data_dir / "test.csv", index=False)

        # Create mock sample_submission.csv
        sample_df = pd.DataFrame({
            'id': [4, 5],
            'target': [0, 0]
        })
        sample_df.to_csv(data_dir / "sample_submission.csv", index=False)

        return data_dir

    def test_auto_enrich_enabled(self, mock_data_dir):
        """Test KaggleTask with auto-enrichment enabled."""
        with patch('mle_star.tasks.kaggle_task.KaggleAPIClient'):
            task = KaggleTask(
                competition_name="test-comp",
                data_dir=mock_data_dir,
                evaluation_metric="accuracy",
                auto_enrich=True
            )

            # Should have dataset_info populated
            assert task.dataset_info is not None
            assert 'data_profile' in task.dataset_info
            assert 'data_files' in task.dataset_info

    def test_auto_enrich_disabled(self, mock_data_dir):
        """Test KaggleTask with auto-enrichment disabled."""
        task = KaggleTask(
            competition_name="test-comp",
            data_dir=mock_data_dir,
            evaluation_metric="accuracy",
            auto_enrich=False
        )

        # Should have minimal description
        assert len(task.description) < 200

    def test_manual_description_override(self, mock_data_dir):
        """Test that manual description overrides auto-generation."""
        custom_desc = "Custom task description"

        task = KaggleTask(
            competition_name="test-comp",
            data_dir=mock_data_dir,
            evaluation_metric="accuracy",
            description=custom_desc,
            auto_enrich=True
        )

        assert task.description == custom_desc

    def test_enrichment_with_profiling_only(self, mock_data_dir):
        """Test enrichment when API fails but profiling succeeds."""
        with patch('mle_star.tasks.kaggle_task.KaggleAPIClient') as mock_client:
            # Mock API to return empty metadata
            mock_instance = Mock()
            mock_instance.is_authenticated.return_value = False
            mock_client.return_value = mock_instance

            task = KaggleTask(
                competition_name="test-comp",
                data_dir=mock_data_dir,
                evaluation_metric="mse",
                auto_enrich=True
            )

            # Should still have dataset profile
            assert 'data_profile' in task.dataset_info
            assert task.dataset_info['data_profile'] != {}

    def test_enrichment_graceful_failure(self, tmp_path):
        """Test that enrichment fails gracefully when data is missing."""
        non_existent_dir = tmp_path / "nonexistent"

        task = KaggleTask(
            competition_name="test-comp",
            data_dir=non_existent_dir,
            evaluation_metric="accuracy",
            auto_enrich=True
        )

        # Should still create task with minimal description
        assert task.description is not None
        assert task.competition_name == "test-comp"

    def test_task_type_detection_from_metric(self, mock_data_dir):
        """Test task type auto-detection from evaluation metric."""
        # Test classification metric
        task_clf = KaggleTask(
            competition_name="test",
            data_dir=mock_data_dir,
            evaluation_metric="accuracy",
            auto_enrich=False
        )
        assert task_clf.task_type.value == "classification"

        # Test regression metric
        task_reg = KaggleTask(
            competition_name="test",
            data_dir=mock_data_dir,
            evaluation_metric="mse",
            auto_enrich=False
        )
        assert task_reg.task_type.value == "regression"


@pytest.mark.integration
class TestEnrichmentIntegration:
    """Integration tests for full enrichment workflow."""

    def test_end_to_end_enrichment(self, tmp_path):
        """Test complete enrichment workflow."""
        # Create test data
        data_dir = tmp_path / "competition"
        data_dir.mkdir()

        train_df = pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Pclass': [3, 1, 3, 1, 3],
            'Name': ['A', 'B', 'C', 'D', 'E'],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22, 38, 26, 35, None],
            'Survived': [0, 1, 1, 1, 0]
        })
        train_df.to_csv(data_dir / "train.csv", index=False)

        test_df = train_df[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age']].copy()
        test_df.to_csv(data_dir / "test.csv", index=False)

        sample_df = pd.DataFrame({'PassengerId': [1, 2], 'Survived': [0, 0]})
        sample_df.to_csv(data_dir / "sample_submission.csv", index=False)

        # Create task with enrichment
        with patch('mle_star.tasks.kaggle_task.KaggleAPIClient'):
            task = KaggleTask(
                competition_name="titanic",
                data_dir=data_dir,
                evaluation_metric="accuracy",
                auto_enrich=True
            )

        # Verify enrichment worked
        assert "Dataset Structure" in task.description
        assert "PassengerId" in task.description
        assert task.dataset_info['data_profile']['num_rows'] == 5
        assert task.dataset_info['data_profile']['num_cols'] == 6

        # Verify profile contains expected information
        profile = task.dataset_info['data_profile']
        assert 'Age' in profile['columns']
        assert profile['columns']['Age']['missing_pct'] == 20.0
