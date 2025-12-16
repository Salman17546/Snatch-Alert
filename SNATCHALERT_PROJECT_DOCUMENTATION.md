# SnatchAlert - Crime Reporting & Stolen Device Tracking System

## Project Description (300 characters)

> **SnatchAlert** is an AI-powered crime reporting platform for Karachi that combines real-time incident reporting with automated social media scraping. Citizens report thefts, track stolen phones via IMEI, and view crime heatmaps—while our scraper continuously collects data from 7+ sources.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Use Cases](#use-cases)
5. [API Documentation](#api-documentation)
6. [Database Schema](#database-schema)
7. [Deployment Guide](#deployment-guide)

---

## Overview

SnatchAlert addresses the growing crime problem in Karachi by providing:

- **For Citizens**: A mobile-friendly platform to report incidents, check if phones are stolen, and view crime hotspots
- **For Authorities**: Aggregated crime data from multiple sources for better resource allocation
- **For Researchers**: Historical crime data with geographic and temporal analysis

### Technology Stack

| Component | Technology |
|-----------|------------|
| Backend API | Django 5.2 + Django REST Framework |
| Database | PostgreSQL 15 |
| Task Queue | Celery + Redis |
| Web Scraping | Selenium + Chrome + BeautifulSoup |
| NLP Processing | 7-tier LLM fallback (Groq, OpenRouter, Gemini, etc.) |
| Containerization | Docker + Docker Compose |

---

## Key Features

### 1. Incident Reporting
- Report crimes with location, time, and description
- Upload photos as evidence
- Anonymous reporting option
- Automatic geocoding of locations

### 2. IMEI Tracking
- Register stolen phone IMEI numbers
- Public IMEI lookup to check if a phone is stolen
- Automatic alerts when stolen devices are reported

### 3. Crime Heatmaps
- Visual representation of crime hotspots
- Filter by crime type, date range, and area

### 4. Automated Data Collection
- Scrapes crime reports from Facebook, Twitter, News Sites, RSS Feeds
- LLM-powered extraction of location, date, crime type, device details
- Runs automatically every 12 hours

### 5. User Management
- JWT-based authentication
- Email verification and password reset


---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       DOCKER COMPOSE                             │
│                                                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    │
│  │ PostgreSQL│  │   Redis   │  │  Django   │  │   Nginx   │    │
│  │  :5432    │  │   :6379   │  │   :8000   │  │   :80     │    │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘    │
│        └──────────────┴──────────────┴──────────────┘           │
│                              │                                   │
│        ┌─────────────────────┴─────────────────────┐            │
│        │                                           │            │
│  ┌─────┴─────┐                             ┌───────┴───────┐    │
│  │  Celery   │                             │    Celery     │    │
│  │  Worker   │                             │     Beat      │    │
│  │ (Scraper) │                             │  (Scheduler)  │    │
│  └───────────┘                             └───────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Use Cases

### UC-01: Report a Crime Incident
**Actor**: Citizen (Registered/Anonymous)
1. User opens app and clicks "Report Incident"
2. Fills location, date/time, crime type, description
3. Optionally adds photos and device details
4. Chooses anonymous or identified reporting
5. System geocodes location and saves incident

**API**: `POST /api/reports/incidents/create/`

---

### UC-02: Check if Phone is Stolen (IMEI Lookup)
**Actor**: Any User (Buyer, Seller, Citizen)
1. User enters 15-digit IMEI number
2. System searches IMEI registry
3. Returns: Not Found (safe) or Stolen (with incident details)

**API**: `POST /api/reports/imei/check/`

---

### UC-03: Register Stolen Phone IMEI
**Actor**: Registered User (Victim)
1. User reports phone snatching incident
2. Enters phone brand, model, IMEI number
3. IMEI added to stolen registry
4. Future checks will flag this device

**API**: `POST /api/reports/imei/register/`

---

### UC-04: View Crime Heatmap
**Actor**: Any User
1. User opens heatmap page
2. Map displays crime density by area
3. Can filter by crime type, date range, area
4. Clicking hotspot shows recent incidents

**API**: `GET /api/reports/heatmap/`

---

### UC-05: Set Area Alert
**Actor**: Registered User
1. User selects area on map
2. Sets crime types to monitor
3. Receives notifications when new incidents match

**API**: `POST /api/reports/alerts/subscribe/`

---

### UC-06: User Registration & Authentication
**Actor**: New User
1. User signs up with email and password
2. Receives verification email
3. Clicks link to activate account
4. Can now report incidents, register devices, set alerts

**API**: `POST /api/auth/register/`, `POST /api/auth/login/`

---

### UC-07: Password Reset
**Actor**: Registered User
1. User clicks "Forgot Password"
2. Enters email, receives reset link
3. Sets new password

**API**: `POST /api/auth/password-reset/`

---

### UC-08: Automated Crime Data Collection
**Actor**: System (Celery Scheduler)
1. Every 12 hours, scraper collects data from Facebook, Twitter, News sites
2. LLM extracts location, date, crime type, device details
3. Quality filter removes duplicates and low-quality data
4. Valid incidents saved to database

**Trigger**: Celery Beat (every 12 hours)

---

### UC-09: Admin Dashboard
**Actor**: System Administrator
1. Admin logs into Django admin panel (`/admin/`)
2. Can view/manage incidents, users, IMEI registry
3. Can moderate content and update incident status


---

## API Documentation

### Authentication Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register/` | POST | Register new user |
| `/api/auth/login/` | POST | Login (returns JWT) |
| `/api/auth/token/refresh/` | POST | Refresh JWT token |
| `/api/auth/password-reset/` | POST | Request password reset |
| `/api/auth/profile/` | GET/PUT | View/update profile |

### Incident Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/reports/incidents/` | GET | List all incidents |
| `/api/reports/incidents/create/` | POST | Report new incident |
| `/api/reports/incidents/{id}/` | GET | Get incident details |
| `/api/reports/heatmap/` | GET | Crime heatmap data |

### IMEI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/reports/imei/check/` | POST | Check if IMEI is stolen |
| `/api/reports/imei/register/` | POST | Register stolen IMEI |

### Alert Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/reports/alerts/subscribe/` | POST | Subscribe to area alerts |
| `/api/reports/alerts/{id}/` | DELETE | Unsubscribe |

---

## Database Schema

### Core Tables

| Table | Description |
|-------|-------------|
| `incident_fact` | Main incidents table (occurred_at, location_id, description, status) |
| `location_dim` | Geographic locations (city, district, neighborhood, lat/long) |
| `incident_type_dim` | Crime categories (Phone Snatching, Robbery, Theft, etc.) |
| `stolen_item_dim` | Stolen devices (phone_brand, phone_model, imei) |
| `imei_registry` | Stolen IMEI tracking (imei, status, incident_id) |
| `custom_user` | User accounts (email, phone_number, is_verified) |

---

## Deployment Guide

### Prerequisites
- Docker & Docker Compose
- At least 4GB RAM, 20GB disk

### Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 2. Start all services
docker-compose up -d

# 3. Run migrations
docker-compose exec snatchalert_api python manage.py migrate

# 4. Create admin user
docker-compose exec snatchalert_api python manage.py createsuperuser

# 5. Verify services
docker-compose ps
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| postgres | 5432 | PostgreSQL database |
| redis | 6379 | Celery message broker |
| snatchalert_api | 8000 | Django REST API |
| scraper_worker | - | Celery worker (scraper) |
| scraper_beat | - | Scheduler (every 12h) |
| nginx | 80 | Reverse proxy (production) |

### Commands

```bash
# View logs
docker-compose logs -f snatchalert_api

# Manual scrape
docker-compose exec scraper_worker python run_scraper.py 50

# Test database
docker-compose exec scraper_worker python run_scraper.py --test

# Restart services
docker-compose restart
```

---

## Project Structure

```
Project Root/
├── docker-compose.yml          # All services
├── .env                        # Configuration
├── nginx.conf                  # Reverse proxy
│
├── SnatchAlert/SnatchAlert/    # Django API
│   ├── Dockerfile
│   ├── manage.py
│   ├── accounts/              # User management
│   ├── reports/               # Incident reporting
│   ├── phones/                # IMEI tracking
│   └── vehicles/              # Vehicle tracking
│
└── Facebook Scraper/           # Crime Scraper
    ├── Dockerfile
    ├── init-db.sql
    ├── run_scraper.py
    ├── ultimate_crime_scraper.py
    └── scraper_integration/
        ├── celery_app.py
        ├── tasks.py
        └── db_adapter.py
```

---

## License

Developed for educational and research purposes.
