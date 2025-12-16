# SnatchAlert API Documentation

Complete API reference for the SnatchAlert crime reporting and tracking system with email-based authentication and IMEI alert system.

## Base URL

```
http://localhost:8000/api
```

## Authentication

SnatchAlert uses JWT (JSON Web Token) authentication with **email-based login** (no username required).

Include the access token in the Authorization header:
```
Authorization: Bearer <your_access_token>
```

## Response Format

### Success Response
```json
{
  "data": { ... },
  "message": "Success message"
}
```

### Error Response
```json
{
  "error": "Error message",
  "details": { ... }
}
```

---

## 🔐 Authentication Endpoints

### Register User
Create a new user account with email.

**Endpoint:** `POST /auth/register/`  
**Permission:** Public

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123",
  "password2": "SecurePass123",
  "first_name": "John",
  "last_name": "Doe",
  "phone": "+923001234567"
}
```

**Response:** `201 Created`
```json
{
  "user": {
    "id": 1,
    "email": "user@example.com",
    "first_name": "John",
    "last_name": "Doe",
    "phone": "+923001234567"
  },
  "tokens": {
    "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "access": "eyJ0eXAiOiJKV1QiLCJhbGc..."
  },
  "message": "Registration successful"
}
```

### Login
Authenticate with email and password.

**Endpoint:** `POST /auth/login/`  
**Permission:** Public

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123"
}
```

**Response:** `200 OK`
```json
{
  "user": {
    "id": 1,
    "email": "user@example.com",
    "first_name": "John",
    "last_name": "Doe",
    "role": "user"
  },
  "tokens": {
    "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "access": "eyJ0eXAiOiJKV1QiLCJhbGc..."
  },
  "message": "Login successful"
}
```

### Refresh Token
Get a new access token using refresh token.

**Endpoint:** `POST /auth/token/refresh/`  
**Permission:** Public

**Request:**
```json
{
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

**Response:** `200 OK`
```json
{
  "access": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

### Get User Profile
Retrieve authenticated user's profile.

**Endpoint:** `GET /auth/profile/`  
**Permission:** Authenticated

**Response:** `200 OK`
```json
{
  "id": 1,
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "phone": "+923001234567",
  "role": "user",
  "is_verified": false,
  "date_joined": "2024-12-05T10:30:00Z"
}
```

### Update Email
Update user's email address.

**Endpoint:** `POST /auth/profile/update-email/`  
**Permission:** Authenticated

**Request:**
```json
{
  "new_email": "newemail@example.com",
  "password": "CurrentPassword123"
}
```

**Response:** `200 OK`
```json
{
  "message": "Email updated successfully",
  "user": {
    "id": 1,
    "email": "newemail@example.com",
    "first_name": "John",
    "last_name": "Doe"
  }
}
```

### Update Password
Change user's password.

**Endpoint:** `POST /auth/profile/update-password/`  
**Permission:** Authenticated

**Request:**
```json
{
  "old_password": "CurrentPassword123",
  "new_password": "NewSecurePass456",
  "new_password2": "NewSecurePass456"
}
```

**Response:** `200 OK`
```json
{
  "message": "Password updated successfully"
}
```

---

## 🔑 Password Reset Endpoints

### Request Password Reset
Generate reset token and send email.

**Endpoint:** `POST /auth/password-reset/request/`  
**Permission:** Public

**Request:**
```json
{
  "email": "user@example.com"
}
```

**Response:** `200 OK`
```json
{
  "message": "Password reset link sent to your email",
  "token": "abc123xyz..."
}
```

### Verify Reset Token
Verify if reset token is valid.

**Endpoint:** `POST /auth/password-reset/verify/`  
**Permission:** Public

**Request:**
```json
{
  "token": "abc123xyz..."
}
```

**Response:** `200 OK`
```json
{
  "valid": true,
  "message": "Token is valid",
  "email": "user@example.com"
}
```

### Confirm Password Reset
Reset password with token.

**Endpoint:** `POST /auth/password-reset/confirm/`  
**Permission:** Public

**Request:**
```json
{
  "token": "abc123xyz...",
  "new_password": "NewSecurePass456",
  "new_password2": "NewSecurePass456"
}
```

**Response:** `200 OK`
```json
{
  "message": "Password reset successful. You can now login with your new password."
}
```

---

## 📱 IMEI Tracking Endpoints

### Register Stolen IMEI
Register a stolen phone IMEI.

**Endpoint:** `POST /reports/imei/register/`  
**Permission:** Authenticated

**Request:**
```json
{
  "imei": "123456789012345",
  "phone_brand": "Samsung",
  "phone_model": "Galaxy S21",
  "owner_name": "John Doe",
  "owner_contact": "+923001234567",
  "status": "stolen",
  "notes": "Stolen from car"
}
```

**Response:** `201 Created`
```json
{
  "id": 1,
  "imei": "123456789012345",
  "phone_brand": "Samsung",
  "phone_model": "Galaxy S21",
  "owner_name": "John Doe",
  "status": "stolen",
  "reported_at": "2024-12-05T15:00:00Z"
}
```

### Check IMEI Status
Check if an IMEI is stolen (triggers alert if found).

**Endpoint:** `POST /reports/imei/check/`  
**Permission:** Public

**Request:**
```json
{
  "imei": "123456789012345"
}
```

**Response (Stolen):** `200 OK`
```json
{
  "found": true,
  "status": "stolen",
  "phone_brand": "Samsung",
  "phone_model": "Galaxy S21",
  "reported_at": "2024-12-05T15:00:00Z",
  "message": "⚠️ WARNING: This IMEI is registered as stolen",
  "warning": "This device has been reported stolen. Do not purchase!",
  "advice": "Contact local authorities if you have information about this device."
}
```

**Response (Safe):** `200 OK`
```json
{
  "found": false,
  "message": "This IMEI is not in our stolen registry",
  "status": "safe"
}
```

### List All IMEIs
Get list of all registered IMEIs (Admin only).

**Endpoint:** `GET /reports/imei/list/`  
**Permission:** Admin/Authority

**Query Parameters:**
- `status` - Filter by status (stolen, recovered, flagged)
- `search` - Search IMEI, brand, model, owner

### Update IMEI Status
Update IMEI status (Admin only).

**Endpoint:** `PATCH /reports/imei/{id}/update/`  
**Permission:** Admin/Authority

**Request:**
```json
{
  "status": "recovered",
  "notes": "Phone recovered by police"
}
```

---

## 🔔 IMEI Alert Endpoints

### Get My Device Alerts
Get all device alerts for authenticated user.

**Endpoint:** `GET /reports/imei/alerts/`  
**Permission:** Authenticated

**Response:** `200 OK`
```json
{
  "unread_count": 2,
  "total_count": 5,
  "alerts": [
    {
      "id": 1,
      "imei": "123456789012345",
      "phone_brand": "Samsung",
      "phone_model": "Galaxy S21",
      "alert_type": "check_detected",
      "message": "🚨 ALERT: Your stolen device has been detected!",
      "is_read": false,
      "created_at": "2024-12-05T14:30:00Z",
      "check_info": {
        "ip_address": "192.168.1.100",
        "checked_at": "2024-12-05T14:30:00Z"
      }
    }
  ]
}
```

### Mark Alert as Read
Mark a specific alert as read.

**Endpoint:** `POST /reports/imei/alerts/{alert_id}/read/`  
**Permission:** Authenticated

**Response:** `200 OK`
```json
{
  "message": "Alert marked as read"
}
```

### Mark All Alerts as Read
Mark all alerts as read.

**Endpoint:** `POST /reports/imei/alerts/read-all/`  
**Permission:** Authenticated

**Response:** `200 OK`
```json
{
  "message": "3 alerts marked as read"
}
```

### View IMEI Check History
View check history for user's registered IMEIs.

**Endpoint:** `GET /reports/imei/check-history/`  
**Permission:** Authenticated

**Response:** `200 OK`
```json
{
  "total_checks": 15,
  "checks": [
    {
      "id": 1,
      "imei": "123456789012345",
      "phone_brand": "Samsung",
      "phone_model": "Galaxy S21",
      "checked_at": "2024-12-05T14:30:00Z",
      "ip_address": "192.168.1.100",
      "alert_sent": true
    }
  ]
}
```

---

## 🚨 Incident Management Endpoints

### Create Incident
Report a new crime incident.

**Endpoint:** `POST /reports/incidents/create/`  
**Permission:** Public (allows anonymous reporting)

**Request:**
```json
{
  "occurred_at": "2024-12-05T14:30:00Z",
  "incident_type_name": "Mobile Snatching",
  "location_data": {
    "province": "Punjab",
    "city": "Lahore",
    "district": "Gulberg",
    "neighborhood": "MM Alam Road",
    "street_address": "Main Boulevard",
    "latitude": 31.5204,
    "longitude": 74.3587
  },
  "victim_data": {
    "name": "John Doe",
    "age": 28,
    "gender": "male",
    "phone_number": "+923001234567"
  },
  "stolen_item_data": {
    "item_type": "phone",
    "imei": "123456789012345",
    "phone_brand": "Samsung",
    "phone_model": "Galaxy S21",
    "value_estimate": 75000
  },
  "value_estimate": 75000,
  "fir_filed": true,
  "description": "Phone snatched at gunpoint",
  "is_anonymous": false
}
```

### List Incidents
Get a list of all incidents with filtering.

**Endpoint:** `GET /reports/incidents/`  
**Permission:** Public

**Query Parameters:**
- `city` - Filter by city
- `district` - Filter by district
- `incident_type__category` - Filter by incident type
- `status` - Filter by status (reported, investigating, resolved, closed)
- `date_from` - Filter from date (YYYY-MM-DD)
- `date_to` - Filter to date (YYYY-MM-DD)
- `fir_filed` - Filter by FIR status (true/false)
- `search` - Search in description and location
- `page` - Page number
- `page_size` - Items per page

### Get My Incidents
Get incidents reported by authenticated user.

**Endpoint:** `GET /reports/incidents/my/`  
**Permission:** Authenticated

### Update Incident
Update an existing incident.

**Endpoint:** `PATCH /reports/incidents/{id}/update/`  
**Permission:** Authenticated (owner only)

### Delete Incident
Delete an incident report.

**Endpoint:** `DELETE /reports/incidents/{id}/delete/`  
**Permission:** Authenticated (owner only)

---

## 📊 Crime Analytics Endpoints

### Crime Heatmap
Get crime hotspot data for map visualization.

**Endpoint:** `GET /reports/heatmap/`  
**Permission:** Public

**Query Parameters:**
- `days` - Number of days to include (default: 30)
- `city` - Filter by city

**Response:** `200 OK`
```json
[
  {
    "latitude": "31.520400",
    "longitude": "74.358700",
    "incident_count": 15,
    "city": "Lahore",
    "district": "Gulberg"
  }
]
```

### Area Safety Score
Calculate safety scores for different areas.

**Endpoint:** `GET /reports/safety-score/`  
**Permission:** Public

**Query Parameters:**
- `city` - Filter by city
- `days` - Number of days to analyze (default: 90)

**Response:** `200 OK`
```json
[
  {
    "location_id": 1,
    "city": "Lahore",
    "district": "Gulberg",
    "neighborhood": "MM Alam Road",
    "incident_count": 15,
    "safety_score": 65.5,
    "risk_level": "Medium"
  }
]
```

**Risk Levels:**
- `Low` - Safety score >= 80
- `Medium` - Safety score 60-79
- `High` - Safety score 40-59
- `Critical` - Safety score < 40

### Crime Statistics
Get overall crime statistics.

**Endpoint:** `GET /reports/statistics/`  
**Permission:** Public

**Response:** `200 OK`
```json
{
  "total_incidents": 150,
  "period_days": 30,
  "by_incident_type": [
    {
      "incident_type__category": "Mobile Snatching",
      "count": 75
    }
  ],
  "top_cities": [
    {
      "location__city": "Lahore",
      "count": 80
    }
  ],
  "fir_filed_percentage": 65.5
}
```

---

## ⚠️ Area Alert Endpoints

### List Active Alerts
Get active location-based alerts.

**Endpoint:** `GET /reports/alerts/`  
**Permission:** Public

**Query Parameters:**
- `alert_type` - Filter by type (high_crime, recent_incident, warning)
- `severity` - Filter by severity (low, medium, high, critical)
- `location__city` - Filter by city

### Create Alert
Create a new area alert (Admin/Authority only).

**Endpoint:** `POST /reports/alerts/create/`  
**Permission:** Admin/Authority

---

## 💡 Community Endpoints

### List Safety Tips
Get community safety tips.

**Endpoint:** `GET /core/safety-tips/`  
**Permission:** Public

### Submit Feedback
Submit user feedback or suggestions.

**Endpoint:** `POST /core/feedback/`  
**Permission:** Public

**Request:**
```json
{
  "subject": "App Suggestion",
  "message": "It would be great to have push notifications",
  "contact_email": "user@example.com"
}
```

### List Incident Types
Get all available incident types.

**Endpoint:** `GET /core/incident-types/`  
**Permission:** Public

---

## 📝 Quick Examples

### Authentication Flow
```bash
# Register
curl -X POST http://localhost:8000/api/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"Pass123","password2":"Pass123"}'

# Login
curl -X POST http://localhost:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"Pass123"}'
```

### Password Reset Flow
```bash
# Request reset
curl -X POST http://localhost:8000/api/auth/password-reset/request/ \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com"}'

# Confirm reset
curl -X POST http://localhost:8000/api/auth/password-reset/confirm/ \
  -H "Content-Type: application/json" \
  -d '{"token":"abc123","new_password":"NewPass123","new_password2":"NewPass123"}'
```

### IMEI Alert System
```bash
# Check IMEI (triggers alert if stolen)
curl -X POST http://localhost:8000/api/reports/imei/check/ \
  -H "Content-Type: application/json" \
  -d '{"imei":"123456789012345"}'

# Get alerts
curl -X GET http://localhost:8000/api/reports/imei/alerts/ \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## 📊 Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 204 | No Content |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 500 | Internal Server Error |

---

## 🔧 Technical Details

### Pagination
List endpoints support pagination:
- `page` - Page number (default: 1)
- `page_size` - Items per page (default: 20, max: 100)

### File Uploads
Use `multipart/form-data` for file uploads (FIR documents, item images).

### Rate Limiting
Currently no rate limiting implemented. Consider adding for production.

---

## 📞 Support

- **Server:** http://127.0.0.1:8000/
- **Interactive Docs:** http://127.0.0.1:8000/api/docs/
- **Admin Panel:** http://127.0.0.1:8000/admin/
- **Email:** support@snatchalert.com