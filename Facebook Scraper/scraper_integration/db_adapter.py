"""
Database Adapter for SnatchAlert Integration.

This module provides direct database access to SnatchAlert's PostgreSQL database,
mapping scraped crime incidents to the SnatchAlert schema.

SnatchAlert Schema (Snowflake):
- incident_fact: Central fact table for incidents
- location_dim: Geographic dimension
- incident_type_dim: Crime type dimension
- stolen_item_dim: Stolen items with IMEI
- imei_registry: Stolen phone IMEI tracking
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
from decimal import Decimal
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

from .config import DATABASE_CONFIG

logger = logging.getLogger(__name__)


class SnatchAlertDBAdapter:
    """Database adapter for saving scraped incidents to SnatchAlert's PostgreSQL database."""
    
    INCIDENT_TYPE_MAPPING = {
        'snatching': 'Phone Snatching',
        'theft': 'Theft',
        'robbery': 'Robbery',
        'burglary': 'Burglary',
        'car theft': 'Vehicle Theft',
        'bike theft': 'Vehicle Theft',
        'mugging': 'Robbery',
        'pickpocket': 'Theft',
        'unknown': 'Other Crime',
    }
    
    def __init__(self):
        """Initialize database connection parameters."""
        self.db_config = DATABASE_CONFIG
        logger.info(f"[DBAdapter] Initialized with host: {self.db_config['HOST']}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(
                dbname=self.db_config['NAME'],
                user=self.db_config['USER'],
                password=self.db_config['PASSWORD'],
                host=self.db_config['HOST'],
                port=self.db_config['PORT']
            )
            yield conn
        except psycopg2.Error as e:
            logger.error(f"[DBAdapter] Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    logger.info("[DBAdapter] Database connection successful")
                    return result[0] == 1
        except Exception as e:
            logger.error(f"[DBAdapter] Connection test failed: {e}")
            return False
    
    def get_or_create_location(self, conn, incident: Dict) -> int:
        """Get or create a LocationDim record."""
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            city = incident.get('city', 'Karachi')
            area = incident.get('area', '') or 'Unknown'
            sub_area = incident.get('sub_area', '')
            latitude = incident.get('latitude')
            longitude = incident.get('longitude')
            
            district = area
            neighborhood = sub_area if sub_area else ''
            
            cur.execute("""
                SELECT id FROM location_dim 
                WHERE city = %s AND district = %s AND neighborhood = %s
                LIMIT 1
            """, (city, district, neighborhood))
            
            result = cur.fetchone()
            if result:
                return result['id']
            
            cur.execute("""
                INSERT INTO location_dim (province, city, district, neighborhood, street_address, latitude, longitude, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                'Sindh',
                city,
                district,
                neighborhood,
                incident.get('location', ''),
                Decimal(str(latitude)) if latitude else None,
                Decimal(str(longitude)) if longitude else None,
                datetime.now()
            ))
            
            conn.commit()
            result = cur.fetchone()
            logger.debug(f"[DBAdapter] Created new location: {district}, {neighborhood}")
            return result['id']
    
    def get_or_create_incident_type(self, conn, incident_type: str) -> int:
        """Get or create an IncidentTypeDim record."""
        category = self.INCIDENT_TYPE_MAPPING.get(incident_type.lower(), 'Other Crime')
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id FROM incident_type_dim 
                WHERE category = %s
                LIMIT 1
            """, (category,))
            
            result = cur.fetchone()
            if result:
                return result['id']
            
            cur.execute("""
                INSERT INTO incident_type_dim (category, description, created_at)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (
                category,
                f"Auto-created from scraper: {incident_type}",
                datetime.now()
            ))
            
            conn.commit()
            result = cur.fetchone()
            logger.debug(f"[DBAdapter] Created new incident type: {category}")
            return result['id']
    
    def create_stolen_item(self, conn, incident: Dict) -> Optional[int]:
        """Create a StolenItemDim record if device info is available."""
        device_model = incident.get('device_model', '')
        imei = incident.get('imei_number', '')
        
        if not device_model and not imei:
            return None
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            item_type = 'phone'
            phone_brand = ''
            phone_model = device_model
            
            brands = ['iPhone', 'Samsung', 'Xiaomi', 'Oppo', 'Vivo', 'Realme', 'OnePlus', 'Huawei', 'Nokia', 'Motorola']
            for brand in brands:
                if brand.lower() in device_model.lower():
                    phone_brand = brand
                    phone_model = device_model.replace(brand, '').strip()
                    break
            
            cur.execute("""
                INSERT INTO stolen_item_dim (item_type, description, imei, phone_brand, phone_model, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                item_type,
                f"Scraped device: {device_model}",
                imei if imei else None,
                phone_brand,
                phone_model,
                datetime.now()
            ))
            
            conn.commit()
            result = cur.fetchone()
            logger.debug(f"[DBAdapter] Created stolen item: {device_model}")
            return result['id']

    def register_imei(self, conn, incident: Dict, incident_id: int) -> Optional[int]:
        """Register IMEI in IMEIRegistry if available."""
        imei = incident.get('imei_number', '')
        if not imei or len(imei) < 15:
            return None
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM imei_registry WHERE imei = %s", (imei,))
            if cur.fetchone():
                logger.debug(f"[DBAdapter] IMEI already registered: {imei}")
                return None
            
            device_model = incident.get('device_model', '')
            phone_brand = ''
            phone_model = device_model
            
            brands = ['iPhone', 'Samsung', 'Xiaomi', 'Oppo', 'Vivo', 'Realme', 'OnePlus', 'Huawei']
            for brand in brands:
                if brand.lower() in device_model.lower():
                    phone_brand = brand
                    phone_model = device_model.replace(brand, '').strip()
                    break
            
            cur.execute("""
                INSERT INTO imei_registry (imei, phone_brand, phone_model, owner_name, owner_contact, 
                                          incident_id, status, reported_at, updated_at, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                imei, phone_brand, phone_model, '',
                incident.get('victim_phone', ''),
                incident_id, 'stolen',
                datetime.now(), datetime.now(),
                f"Auto-registered from scraper. Source: {incident.get('source', 'Unknown')}"
            ))
            
            conn.commit()
            result = cur.fetchone()
            logger.info(f"[DBAdapter] Registered IMEI: {imei[:4]}...{imei[-4:]}")
            return result['id']
    
    def incident_exists(self, conn, incident_id: str) -> bool:
        """Check if an incident with this ID already exists."""
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM incident_fact 
                WHERE description LIKE %s
            """, (f"%[ID:{incident_id[:8]}]%",))
            result = cur.fetchone()
            return result[0] > 0
    
    def _normalize_incident(self, incident: Dict) -> Optional[Dict]:
        """Normalize incident data with proper defaults. Returns None if should be skipped."""
        if incident.get('is_statistical', False):
            logger.debug("[DBAdapter] Skipping statistical entry")
            return None
        
        quality_score = float(incident.get('quality_score', 0.5))
        if quality_score < 0.3:
            logger.debug(f"[DBAdapter] Skipping low quality entry (score: {quality_score})")
            return None
        
        normalized = {
            'incident_id': incident.get('incident_id', ''),
            'source': incident.get('source', 'Unknown'),
            'source_url': incident.get('source_url', ''),
            'city': incident.get('city', 'Karachi'),
            'area': incident.get('area', '') or 'Unknown Area',
            'sub_area': incident.get('sub_area', ''),
            'location': incident.get('location', ''),
            'latitude': incident.get('latitude'),
            'longitude': incident.get('longitude'),
            'incident_type': incident.get('incident_type', '') or 'unknown',
            'description': incident.get('description', '') or incident.get('raw_text', '')[:500],
            'device_model': incident.get('device_model', ''),
            'imei_number': incident.get('imei_number', ''),
            'victim_phone': incident.get('victim_phone', ''),
            'confidence_score': float(incident.get('confidence_score', 0.5)),
            'quality_score': quality_score,
        }
        
        incident_date = incident.get('incident_date', '')
        if not incident_date or incident_date in ['Unknown', 'N/A', '']:
            incident_date = datetime.now().strftime('%Y-%m-%d')
        normalized['incident_date'] = incident_date
        
        incident_time = incident.get('incident_time', '')
        if not incident_time or incident_time in ['Unknown', 'N/A', '']:
            incident_time = '12:00 PM'
        normalized['incident_time'] = incident_time
        
        return normalized
    
    def _parse_datetime(self, date_str: str, time_str: str) -> datetime:
        """Parse date and time strings into datetime object."""
        date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y']
        parsed_date = None
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        
        if not parsed_date:
            parsed_date = datetime.now()
        
        time_formats = ['%I:%M %p', '%H:%M', '%I:%M%p', '%H:%M:%S']
        for fmt in time_formats:
            try:
                parsed_time = datetime.strptime(time_str.strip(), fmt)
                return parsed_date.replace(hour=parsed_time.hour, minute=parsed_time.minute)
            except ValueError:
                continue
        
        return parsed_date.replace(hour=12, minute=0)

    def save_incident(self, incident: Dict) -> Optional[int]:
        """Save a single scraped incident to SnatchAlert database."""
        try:
            normalized = self._normalize_incident(incident)
            if normalized is None:
                return None
            
            with self.get_connection() as conn:
                incident_id = normalized.get('incident_id', '')
                if incident_id and self.incident_exists(conn, incident_id):
                    logger.debug(f"[DBAdapter] Skipping duplicate incident: {incident_id[:8]}")
                    return None
                
                location_id = self.get_or_create_location(conn, normalized)
                incident_type_id = self.get_or_create_incident_type(conn, normalized.get('incident_type', 'unknown'))
                stolen_item_id = self.create_stolen_item(conn, normalized)
                
                occurred_at = self._parse_datetime(normalized['incident_date'], normalized['incident_time'])
                
                description = normalized.get('description', 'No description available')
                source = normalized.get('source', 'Unknown')
                source_url = normalized.get('source_url', '')
                
                full_description = f"{description}\n\n[Scraped from: {source}]\n[ID:{incident_id[:8]}]"
                if source_url:
                    full_description += f"\n[Source URL: {source_url}]"
                
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        INSERT INTO incident_fact (
                            occurred_at, location_id, victim_id, incident_type_id, stolen_item_id,
                            value_estimate, fir_filed, description, is_anonymous, status,
                            created_at, updated_at, reported_by_id
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        occurred_at, location_id, None, incident_type_id, stolen_item_id,
                        None, False, full_description, True, 'reported',
                        datetime.now(), datetime.now(), None
                    ))
                    
                    conn.commit()
                    result = cur.fetchone()
                    fact_id = result['id']
                
                self.register_imei(conn, normalized, fact_id)
                
                logger.info(f"[DBAdapter] Saved incident {fact_id}: {normalized.get('incident_type')} at {normalized.get('area')}")
                return fact_id
                
        except Exception as e:
            logger.error(f"[DBAdapter] Error saving incident: {e}")
            return None
    
    def save_incidents_batch(self, incidents: List[Dict]) -> Dict[str, int]:
        """Save multiple incidents in batch."""
        stats = {'saved': 0, 'duplicates': 0, 'skipped': 0, 'errors': 0, 'total': len(incidents)}
        
        real_incidents = []
        for incident in incidents:
            if incident.get('is_statistical', False):
                stats['skipped'] += 1
            elif float(incident.get('quality_score', 0.5)) < 0.3:
                stats['skipped'] += 1
            else:
                real_incidents.append(incident)
        
        logger.info(f"[DBAdapter] Starting batch save: {len(real_incidents)} real incidents (skipped {stats['skipped']})")
        
        for incident in real_incidents:
            try:
                result = self.save_incident(incident)
                if result:
                    stats['saved'] += 1
                else:
                    stats['duplicates'] += 1
            except Exception as e:
                logger.error(f"[DBAdapter] Batch save error: {e}")
                stats['errors'] += 1
        
        logger.info(f"[DBAdapter] Batch complete: {stats['saved']} saved, {stats['duplicates']} duplicates, {stats['skipped']} skipped, {stats['errors']} errors")
        return stats
    
    def get_recent_incidents_count(self, hours: int = 24) -> int:
        """Get count of incidents saved in the last N hours."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*) FROM incident_fact 
                        WHERE created_at >= NOW() - INTERVAL '%s hours'
                    """, (hours,))
                    result = cur.fetchone()
                    return result[0]
        except Exception as e:
            logger.error(f"[DBAdapter] Error getting recent count: {e}")
            return 0
