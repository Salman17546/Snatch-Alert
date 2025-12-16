"""
SnatchAlert Scraper Integration Module

This module provides:
- Database adapter for PostgreSQL
- Celery tasks for scheduled scraping
- Configuration management
"""

from .config import DATABASE_CONFIG, REDIS_URL, SCRAPER_TARGET_INCIDENTS
from .db_adapter import SnatchAlertDBAdapter

__all__ = [
    'DATABASE_CONFIG',
    'REDIS_URL', 
    'SCRAPER_TARGET_INCIDENTS',
    'SnatchAlertDBAdapter',
]
