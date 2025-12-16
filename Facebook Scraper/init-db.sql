-- SnatchAlert Database Schema
-- This creates the tables needed for the crime scraper

-- Location dimension table
CREATE TABLE IF NOT EXISTS location_dim (
    id SERIAL PRIMARY KEY,
    province VARCHAR(100) DEFAULT 'Sindh',
    city VARCHAR(100) NOT NULL DEFAULT 'Karachi',
    district VARCHAR(200),
    neighborhood VARCHAR(200),
    street_address TEXT,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Incident type dimension table
CREATE TABLE IF NOT EXISTS incident_type_dim (
    id SERIAL PRIMARY KEY,
    category VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Victim dimension table
CREATE TABLE IF NOT EXISTS victim_dim (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200),
    phone VARCHAR(20),
    email VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stolen item dimension table
CREATE TABLE IF NOT EXISTS stolen_item_dim (
    id SERIAL PRIMARY KEY,
    item_type VARCHAR(50) NOT NULL DEFAULT 'phone',
    description TEXT,
    imei VARCHAR(20),
    phone_brand VARCHAR(100),
    phone_model VARCHAR(200),
    license_plate VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Main incident fact table
CREATE TABLE IF NOT EXISTS incident_fact (
    id SERIAL PRIMARY KEY,
    occurred_at TIMESTAMP NOT NULL,
    location_id INTEGER REFERENCES location_dim(id),
    victim_id INTEGER REFERENCES victim_dim(id),
    incident_type_id INTEGER REFERENCES incident_type_dim(id),
    stolen_item_id INTEGER REFERENCES stolen_item_dim(id),
    value_estimate DECIMAL(12, 2),
    fir_filed BOOLEAN DEFAULT FALSE,
    description TEXT,
    is_anonymous BOOLEAN DEFAULT TRUE,
    status VARCHAR(50) DEFAULT 'reported',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reported_by_id INTEGER
);

-- IMEI registry table
CREATE TABLE IF NOT EXISTS imei_registry (
    id SERIAL PRIMARY KEY,
    imei VARCHAR(20) UNIQUE NOT NULL,
    phone_brand VARCHAR(100),
    phone_model VARCHAR(200),
    owner_name VARCHAR(200),
    owner_contact VARCHAR(100),
    incident_id INTEGER REFERENCES incident_fact(id),
    status VARCHAR(50) DEFAULT 'stolen',
    reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_incident_fact_occurred_at ON incident_fact(occurred_at);
CREATE INDEX IF NOT EXISTS idx_incident_fact_status ON incident_fact(status);
CREATE INDEX IF NOT EXISTS idx_incident_fact_created_at ON incident_fact(created_at);
CREATE INDEX IF NOT EXISTS idx_location_dim_city ON location_dim(city);
CREATE INDEX IF NOT EXISTS idx_location_dim_district ON location_dim(district);
CREATE INDEX IF NOT EXISTS idx_imei_registry_imei ON imei_registry(imei);
CREATE INDEX IF NOT EXISTS idx_imei_registry_status ON imei_registry(status);

-- Insert default incident types
INSERT INTO incident_type_dim (category, description) VALUES
    ('Phone Snatching', 'Mobile phone snatching incidents'),
    ('Robbery', 'Armed or unarmed robbery'),
    ('Theft', 'General theft incidents'),
    ('Vehicle Theft', 'Car or motorcycle theft'),
    ('Burglary', 'Breaking and entering'),
    ('Other Crime', 'Other crime types')
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO snatch_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO snatch_user;
