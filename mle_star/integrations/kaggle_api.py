"""
Kaggle API integration for competition metadata retrieval.

This module provides a wrapper around the Kaggle API with error handling,
caching, and graceful degradation when credentials are unavailable.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KaggleAPIClient:
    """
    Wrapper for Kaggle API with error handling and caching.

    This client fetches competition metadata from Kaggle API and caches results
    to avoid redundant API calls. All methods gracefully handle failures by
    returning empty dicts/lists rather than raising exceptions.

    Authentication:
        Requires ~/.kaggle/kaggle.json with format:
        {
            "username": "your_username",
            "key": "your_api_key"
        }

    Examples:
        >>> client = KaggleAPIClient()
        >>> if client.is_authenticated():
        ...     metadata = client.fetch_competition_metadata("titanic")
        ...     print(metadata['title'])
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize Kaggle API client.

        Args:
            cache_dir: Directory for caching API responses. Defaults to
                      ~/.cache/mle_star/kaggle/
        """
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'mle_star' / 'kaggle'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache TTL: 24 hours (competitions don't change often)
        self.cache_ttl = timedelta(hours=24)

        # Try to import and initialize Kaggle API
        self._api = None
        self._authenticated = False
        self._initialize_api()

    def _initialize_api(self) -> None:
        """Initialize Kaggle API with authentication check."""
        try:
            # Import kaggle module (fails if not installed)
            from kaggle.api.kaggle_api_extended import KaggleApi

            # Initialize API
            api = KaggleApi()
            api.authenticate()  # Raises error if credentials not found

            self._api = api
            self._authenticated = True
            logger.info("✓ Kaggle API authenticated successfully")

        except ImportError:
            logger.warning(
                "Kaggle package not installed. "
                "Install with: pip install kaggle"
            )
            self._authenticated = False

        except OSError as e:
            # Credentials not found
            logger.warning(
                f"Kaggle credentials not found: {e}. "
                "Setup ~/.kaggle/kaggle.json to enable auto-enrichment. "
                "See: https://www.kaggle.com/docs/api#authentication"
            )
            self._authenticated = False

        except Exception as e:
            logger.warning(f"Failed to initialize Kaggle API: {e}")
            self._authenticated = False

    def is_authenticated(self) -> bool:
        """
        Check if Kaggle API is authenticated.

        Returns:
            True if credentials are configured and valid, False otherwise
        """
        return self._authenticated

    def fetch_competition_metadata(
        self,
        competition_name: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch competition metadata from Kaggle API.

        Retrieves competition information including title, description,
        evaluation metric, and other metadata. Results are cached to avoid
        redundant API calls.

        Args:
            competition_name: Kaggle competition identifier (e.g., 'titanic')
            use_cache: If True, use cached data if available

        Returns:
            Dictionary with competition metadata:
            {
                'title': str,           # Competition title
                'description': str,     # Competition overview (HTML/Markdown)
                'evaluation': str,      # Evaluation metric description
                'reward': str,          # Prize money (if any)
                'tags': List[str],      # Competition tags/categories
                'deadline': datetime,   # Submission deadline
                'url': str,             # Competition URL
                'max_team_size': int,   # Maximum team size
                'enable_private_leaderboard': bool,  # Has private LB
            }

            Returns empty dict {} if:
            - API not authenticated
            - Competition not found
            - API call fails

        Examples:
            >>> client = KaggleAPIClient()
            >>> metadata = client.fetch_competition_metadata("titanic")
            >>> print(metadata.get('title', 'Unknown'))
        """
        if not self.is_authenticated():
            logger.debug(f"Skipping metadata fetch for {competition_name} (not authenticated)")
            return {}

        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(f"comp_metadata_{competition_name}")
            if cached_data:
                logger.debug(f"Using cached metadata for {competition_name}")
                return cached_data

        # Fetch from API
        try:
            logger.debug(f"Fetching metadata for {competition_name} from Kaggle API")

            # Get competition details
            comp = self._api.competition_view(competition_name)

            # Extract relevant metadata
            metadata = {
                'title': comp.title if hasattr(comp, 'title') else competition_name,
                'description': comp.description if hasattr(comp, 'description') else '',
                'evaluation': comp.evaluationMetric if hasattr(comp, 'evaluationMetric') else '',
                'reward': comp.reward if hasattr(comp, 'reward') else 'None',
                'tags': comp.tags if hasattr(comp, 'tags') else [],
                'deadline': self._parse_deadline(comp),
                'url': f"https://www.kaggle.com/competitions/{competition_name}",
                'max_team_size': comp.maxTeamSize if hasattr(comp, 'maxTeamSize') else 1,
                'enable_private_leaderboard': getattr(comp, 'enablePrivateLeaderboard', False),
            }

            # Cache the result
            self._save_to_cache(f"comp_metadata_{competition_name}", metadata)

            logger.info(f"✓ Fetched metadata for {competition_name}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to fetch metadata for {competition_name}: {e}")
            return {}

    def fetch_data_files_info(
        self,
        competition_name: str,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get list of data files for a competition without downloading.

        Args:
            competition_name: Kaggle competition identifier
            use_cache: If True, use cached data if available

        Returns:
            List of file information dictionaries:
            [
                {
                    'name': 'train.csv',
                    'size': 12345,        # Size in bytes
                    'description': '...'  # File description (if available)
                },
                ...
            ]

            Returns empty list [] if:
            - API not authenticated
            - Competition not found
            - API call fails

        Examples:
            >>> client = KaggleAPIClient()
            >>> files = client.fetch_data_files_info("titanic")
            >>> for f in files:
            ...     print(f"{f['name']}: {f['size'] / 1024:.1f} KB")
        """
        if not self.is_authenticated():
            logger.debug(f"Skipping files info fetch for {competition_name} (not authenticated)")
            return []

        # Check cache
        if use_cache:
            cached_data = self._load_from_cache(f"comp_files_{competition_name}")
            if cached_data:
                logger.debug(f"Using cached files info for {competition_name}")
                return cached_data

        # Fetch from API
        try:
            logger.debug(f"Fetching files info for {competition_name} from Kaggle API")

            # Get competition files list
            files_list = self._api.competition_list_files(competition_name)

            # Extract file information
            files_info = []
            for file_obj in files_list:
                file_info = {
                    'name': file_obj.name if hasattr(file_obj, 'name') else 'unknown',
                    'size': file_obj.totalBytes if hasattr(file_obj, 'totalBytes') else 0,
                    'description': file_obj.description if hasattr(file_obj, 'description') else '',
                }
                files_info.append(file_info)

            # Cache the result
            self._save_to_cache(f"comp_files_{competition_name}", files_info)

            logger.debug(f"Found {len(files_info)} files for {competition_name}")
            return files_info

        except Exception as e:
            logger.error(f"Failed to fetch files info for {competition_name}: {e}")
            return []

    def _parse_deadline(self, comp: Any) -> Optional[str]:
        """Parse competition deadline from competition object."""
        try:
            if hasattr(comp, 'deadline'):
                # Convert to ISO format string for JSON serialization
                if isinstance(comp.deadline, datetime):
                    return comp.deadline.isoformat()
                return str(comp.deadline)
            return None
        except Exception:
            return None

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        # Sanitize key for filesystem
        safe_key = "".join(c for c in key if c.isalnum() or c in ('_', '-'))
        return self.cache_dir / f"{safe_key}.json"

    def _load_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Load data from cache if exists and not expired."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            # Check if cache is expired
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mtime > self.cache_ttl:
                logger.debug(f"Cache expired for {key}")
                cache_path.unlink()  # Delete expired cache
                return None

            # Load from cache
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return data

        except Exception as e:
            logger.debug(f"Failed to load cache for {key}: {e}")
            return None

    def _save_to_cache(self, key: str, data: Any) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)  # default=str handles datetime

            logger.debug(f"Cached data for {key}")

        except Exception as e:
            logger.debug(f"Failed to save cache for {key}: {e}")

    def clear_cache(self, competition_name: Optional[str] = None) -> None:
        """
        Clear cached data.

        Args:
            competition_name: If provided, clear cache only for this competition.
                             If None, clear all cached data.
        """
        try:
            if competition_name:
                # Clear specific competition cache
                for pattern in [f"comp_metadata_{competition_name}", f"comp_files_{competition_name}"]:
                    cache_path = self._get_cache_path(pattern)
                    if cache_path.exists():
                        cache_path.unlink()
                        logger.info(f"Cleared cache for {competition_name}")
            else:
                # Clear all cache
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
                logger.info("Cleared all Kaggle API cache")

        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
