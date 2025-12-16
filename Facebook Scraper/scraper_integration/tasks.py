"""
Celery tasks for the crime scraper.
"""
import logging
import os
from datetime import datetime
from pathlib import Path

from .celery_app import app
from .db_adapter import SnatchAlertDBAdapter
from .config import SCRAPER_TARGET_INCIDENTS

logger = logging.getLogger(__name__)


@app.task(bind=True, max_retries=3, default_retry_delay=300)
def run_scraper_task(self, target: int = None):
    """
    Main scraper task - scrapes crime data and saves to database.
    
    Args:
        target: Number of incidents to scrape (default from config)
    """
    if target is None:
        target = SCRAPER_TARGET_INCIDENTS
    
    task_id = self.request.id
    logger.info(f"[Task {task_id}] Starting scraper - Target: {target} incidents")
    
    try:
        # Import scraper (heavy imports inside task)
        from ultimate_crime_scraper import Configuration, ScraperOrchestrator, ExecutionMode
        
        # Initialize scraper
        config = Configuration()
        orchestrator = ScraperOrchestrator(config, mode=ExecutionMode.SCHEDULED, target=target)
        
        # Run scraping
        logger.info(f"[Task {task_id}] Scraping crime data from sources...")
        incidents = orchestrator.run()
        logger.info(f"[Task {task_id}] Scraped {len(incidents)} incidents")
        
        # Save to database
        if incidents:
            logger.info(f"[Task {task_id}] Saving to database...")
            adapter = SnatchAlertDBAdapter()
            
            if adapter.test_connection():
                incident_dicts = [inc.to_dict() for inc in incidents]
                stats = adapter.save_incidents_batch(incident_dicts)
                
                logger.info(f"[Task {task_id}] Results: {stats}")
                
                # Cleanup
                orchestrator.graceful_shutdown()
                
                return {
                    'status': 'success',
                    'task_id': task_id,
                    'scraped': len(incidents),
                    'saved': stats['saved'],
                    'duplicates': stats['duplicates'],
                    'skipped': stats['skipped'],
                    'errors': stats['errors'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise Exception("Database connection failed")
        else:
            orchestrator.graceful_shutdown()
            return {
                'status': 'success',
                'task_id': task_id,
                'scraped': 0,
                'message': 'No incidents found',
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"[Task {task_id}] Error: {e}")
        raise self.retry(exc=e)


@app.task
def cleanup_logs_task():
    """Clean up old log files to save disk space."""
    logger.info("[Cleanup] Starting log cleanup...")
    
    logs_dir = Path('logs')
    if not logs_dir.exists():
        return {'status': 'skipped', 'message': 'Logs directory not found'}
    
    deleted = 0
    for log_file in logs_dir.glob('*.log.*'):
        try:
            # Delete rotated log files older than 7 days
            if log_file.stat().st_mtime < (datetime.now().timestamp() - 7 * 24 * 3600):
                log_file.unlink()
                deleted += 1
        except Exception as e:
            logger.error(f"[Cleanup] Error deleting {log_file}: {e}")
    
    logger.info(f"[Cleanup] Deleted {deleted} old log files")
    return {'status': 'success', 'deleted': deleted}


@app.task
def test_database_task():
    """Test database connection."""
    adapter = SnatchAlertDBAdapter()
    connected = adapter.test_connection()
    count = adapter.get_recent_incidents_count(24) if connected else 0
    
    return {
        'connected': connected,
        'incidents_24h': count,
        'timestamp': datetime.now().isoformat()
    }
