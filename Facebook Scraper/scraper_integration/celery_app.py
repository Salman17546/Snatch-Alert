"""
Celery application configuration for the crime scraper.
"""
from celery import Celery
from celery.schedules import crontab
from .config import REDIS_URL, SCRAPER_SCHEDULE_HOURS

# Create Celery app
app = Celery(
    'scraper',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['scraper_integration.tasks']
)

# Celery configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Karachi',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,
    worker_concurrency=2,
)

# Beat schedule for periodic tasks
app.conf.beat_schedule = {
    'scrape-crime-data': {
        'task': 'scraper_integration.tasks.run_scraper_task',
        'schedule': crontab(minute=0, hour=f'*/{SCRAPER_SCHEDULE_HOURS}'),
        'args': (50,),  # Target 50 incidents per run
    },
    'cleanup-old-logs': {
        'task': 'scraper_integration.tasks.cleanup_logs_task',
        'schedule': crontab(minute=0, hour=3),  # Daily at 3 AM
    },
}

if __name__ == '__main__':
    app.start()
