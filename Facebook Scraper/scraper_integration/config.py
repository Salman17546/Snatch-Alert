"""
Configuration for the scraper integration with SnatchAlert database.
Loads settings from environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Database configuration (PostgreSQL)
DATABASE_CONFIG = {
    'ENGINE': 'django.db.backends.postgresql',
    'NAME': os.getenv('DB_NAME', 'snatchalertdb'),
    'USER': os.getenv('DB_USER', 'snatch_user'),
    'PASSWORD': os.getenv('DB_PASSWORD', 'SnatchAlert123'),
    'HOST': os.getenv('DB_HOST', 'postgres'),
    'PORT': os.getenv('DB_PORT', '5432'),
}

# Redis configuration (for Celery)
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')

# Scraper settings
SCRAPER_TARGET_INCIDENTS = int(os.getenv('SCRAPER_TARGET_INCIDENTS', '50'))
SCRAPER_SCHEDULE_HOURS = int(os.getenv('SCRAPER_SCHEDULE_HOURS', '12'))
