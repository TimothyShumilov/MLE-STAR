"""
Integrations module for external APIs and services.

This module provides integrations with external services like Kaggle,
enabling automatic competition metadata retrieval and dataset management.
"""

from mle_star.integrations.kaggle_api import KaggleAPIClient

__all__ = ['KaggleAPIClient']
