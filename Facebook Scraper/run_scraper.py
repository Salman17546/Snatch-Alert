#!/usr/bin/env python3
"""
Run the crime scraper and save results to SnatchAlert database.

Usage:
    python run_scraper.py              # Scrape 50 incidents (default)
    python run_scraper.py 100          # Scrape 100 incidents
    python run_scraper.py --test       # Test database connection only
"""
import sys
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database():
    """Test database connection."""
    print("=" * 60)
    print("Testing SnatchAlert Database Connection")
    print("=" * 60)
    
    try:
        from scraper_integration.db_adapter import SnatchAlertDBAdapter
        from scraper_integration.config import DATABASE_CONFIG
        
        print(f"  Host: {DATABASE_CONFIG['HOST']}")
        print(f"  Database: {DATABASE_CONFIG['NAME']}")
        print(f"  User: {DATABASE_CONFIG['USER']}")
        
        adapter = SnatchAlertDBAdapter()
        if adapter.test_connection():
            print("✓ Connection successful!")
            count = adapter.get_recent_incidents_count(hours=24)
            print(f"✓ Incidents in last 24h: {count}")
            return True
        else:
            print("✗ Connection failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def run_scraper(target: int = 50):
    """Run the scraper and save to database."""
    print("=" * 60)
    print(f"Starting Crime Scraper - Target: {target} incidents")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        from ultimate_crime_scraper import Configuration, ScraperOrchestrator, ExecutionMode
        from scraper_integration.db_adapter import SnatchAlertDBAdapter
        
        # Initialize scraper
        config = Configuration()
        orchestrator = ScraperOrchestrator(config, mode=ExecutionMode.SCHEDULED, target=target)
        
        # Run scraping
        print("\n[1/3] Scraping crime data from sources...")
        incidents = orchestrator.run()
        print(f"      Scraped {len(incidents)} incidents")
        
        # Save to database
        if incidents:
            print("\n[2/3] Saving to SnatchAlert database...")
            adapter = SnatchAlertDBAdapter()
            
            if adapter.test_connection():
                incident_dicts = [inc.to_dict() for inc in incidents]
                stats = adapter.save_incidents_batch(incident_dicts)
                
                print(f"      Saved: {stats['saved']}")
                print(f"      Duplicates: {stats['duplicates']}")
                print(f"      Skipped: {stats['skipped']}")
                print(f"      Errors: {stats['errors']}")
            else:
                print("      ✗ Database connection failed - saving to local file only")
                orchestrator.incident_store.save_incidents(incidents, append_mode=True)
        else:
            print("\n[2/3] No incidents to save")
        
        # Cleanup
        print("\n[3/3] Cleanup...")
        orchestrator.graceful_shutdown()
        
        print("\n" + "=" * 60)
        print("✓ Scraping completed successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Run crime scraper for SnatchAlert')
    parser.add_argument('target', type=int, nargs='?', default=50,
                       help='Number of incidents to scrape (default: 50)')
    parser.add_argument('--test', action='store_true',
                       help='Test database connection only')
    
    args = parser.parse_args()
    
    if args.test:
        success = test_database()
    else:
        success = run_scraper(args.target)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
