# SnatchAlert Crime Scraper

Automated crime data scraper with Docker deployment, Celery scheduling, and PostgreSQL integration.

## Quick Start

### 1. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start Services
```bash
# Start all services (PostgreSQL, Redis, Celery Worker, Celery Beat)
docker-compose up -d

# View logs
docker-compose logs -f
```

### 3. Run Manual Scrape (Optional)
```bash
docker-compose --profile manual run scraper
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose                            │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ PostgreSQL│  │  Redis   │  │  Celery  │  │  Celery  │    │
│  │    DB    │  │  Broker  │  │  Worker  │  │   Beat   │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │           │
│       └─────────────┴─────────────┴─────────────┘           │
│                           │                                  │
│                    Scraper Task                              │
│                           │                                  │
│              ┌────────────┴────────────┐                    │
│              │                         │                    │
│         Facebook/Twitter          RSS Feeds                 │
│         (Selenium)               (feedparser)               │
└─────────────────────────────────────────────────────────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| postgres | 5432 | PostgreSQL database |
| redis | 6379 | Redis for Celery broker |
| celery_worker | - | Executes scraper tasks |
| celery_beat | - | Schedules periodic tasks |

## Scheduled Tasks

| Task | Schedule | Description |
|------|----------|-------------|
| `run_scraper_task` | Every 12 hours | Scrapes 50 incidents |
| `cleanup_logs_task` | Daily 3 AM | Removes old log files |

## Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f celery_worker

# Run one-time scrape
docker-compose --profile manual run scraper

# Test database connection
docker-compose exec celery_worker python run_scraper.py --test

# Scale workers
docker-compose up -d --scale celery_worker=3

# Rebuild after code changes
docker-compose build && docker-compose up -d
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_NAME` | snatchalertdb | PostgreSQL database name |
| `DB_USER` | snatch_user | Database user |
| `DB_PASSWORD` | SnatchAlert123 | Database password |
| `SCRAPER_TARGET_INCIDENTS` | 50 | Incidents per scrape run |
| `SCRAPER_SCHEDULE_HOURS` | 12 | Hours between scrapes |

### Required API Keys

At least one LLM API key is required for text processing:
- `GROQ_API_KEY` (recommended - free tier)
- `OPENROUTER_API_KEY`
- `GEMINI_API_KEY`

## Database Schema

| Table | Description |
|-------|-------------|
| `incident_fact` | Main incidents table |
| `location_dim` | Geographic locations |
| `incident_type_dim` | Crime categories |
| `stolen_item_dim` | Stolen devices |
| `imei_registry` | IMEI tracking |

## Files

```
Facebook Scraper/
├── docker-compose.yml      # Docker services
├── Dockerfile              # Container build
├── init-db.sql            # Database schema
├── run_scraper.py         # CLI entry point
├── ultimate_crime_scraper.py  # Core scraper
├── scraper_integration/
│   ├── config.py          # Configuration
│   ├── db_adapter.py      # PostgreSQL adapter
│   ├── celery_app.py      # Celery config
│   └── tasks.py           # Celery tasks
├── .env                   # Your config (create from .env.example)
└── requirements.txt
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Database connection failed | Check DB_HOST, ensure postgres container is running |
| Celery not processing | Check Redis connection, view worker logs |
| Chrome/Selenium errors | Ensure Chrome is installed in container |
| No incidents scraped | Check LLM API keys are configured |
