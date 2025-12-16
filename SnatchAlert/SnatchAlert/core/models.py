from django.db import models
from django.utils import timezone

class LocationDim(models.Model):
    """Dimension table for location information"""
    province = models.CharField(max_length=100, blank=True)
    city = models.CharField(max_length=100)
    district = models.CharField(max_length=100, blank=True)
    neighborhood = models.CharField(max_length=100, blank=True)
    street_address = models.CharField(max_length=255, blank=True)
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    class Meta:
        db_table = 'location_dim'
        indexes = [
            models.Index(fields=['city', 'district']),
            models.Index(fields=['latitude', 'longitude']),
        ]

    def __str__(self):
        return f"{self.neighborhood or self.district or self.city}"


class VictimDim(models.Model):
    """Dimension table for victim information"""
    user = models.ForeignKey("accounts.CustomUser", on_delete=models.SET_NULL, null=True, blank=True)
    name = models.CharField(max_length=150, null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=20, choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')], null=True, blank=True)
    phone_number = models.CharField(max_length=20, null=True, blank=True)
    email = models.EmailField(null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    class Meta:
        db_table = 'victim_dim'

    def __str__(self):
        return self.name or "Anonymous Victim"


class IncidentTypeDim(models.Model):
    """Dimension table for incident types/categories"""
    category = models.CharField(max_length=100, unique=True)
    description = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    class Meta:
        db_table = 'incident_type_dim'

    def __str__(self):
        return self.category


class StolenItemDim(models.Model):
    """Dimension table for stolen items with phone and vehicle specific fields"""
    ITEM_TYPES = [
        ('phone', 'Phone'),
        ('car', 'Car'),
        ('bike', 'Bike'),
        ('bag', 'Bag'),
        ('other', 'Other'),
    ]
    
    item_type = models.CharField(max_length=20, choices=ITEM_TYPES)
    description = models.TextField(null=True, blank=True)
    value_estimate = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    # Phone-specific fields
    imei = models.CharField(max_length=20, null=True, blank=True, db_index=True)
    phone_brand = models.CharField(max_length=100, null=True, blank=True)
    phone_model = models.CharField(max_length=100, null=True, blank=True)
    
    # Vehicle-specific fields
    license_plate = models.CharField(max_length=50, null=True, blank=True, db_index=True)
    chassis_number = models.CharField(max_length=50, null=True, blank=True)
    vehicle_make = models.CharField(max_length=100, null=True, blank=True)
    vehicle_model = models.CharField(max_length=100, null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    class Meta:
        db_table = 'stolen_item_dim'
        indexes = [
            models.Index(fields=['item_type']),
            models.Index(fields=['imei']),
            models.Index(fields=['license_plate']),
        ]

    def __str__(self):
        if self.item_type == 'phone' and self.imei:
            return f"Phone - {self.phone_brand} {self.phone_model} ({self.imei})"
        elif self.item_type in ['car', 'bike'] and self.license_plate:
            return f"{self.item_type.title()} - {self.license_plate}"
        return f"{self.item_type.title()}"


class SafetyTip(models.Model):
    """Community safety tips"""
    title = models.CharField(max_length=200)
    content = models.TextField()
    category = models.CharField(max_length=100, blank=True)
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey("accounts.CustomUser", on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)

    class Meta:
        db_table = 'safety_tips'
        ordering = ['-created_at']

    def __str__(self):
        return self.title


class UserFeedback(models.Model):
    """User feedback and suggestions"""
    user = models.ForeignKey("accounts.CustomUser", on_delete=models.SET_NULL, null=True, blank=True)
    subject = models.CharField(max_length=200)
    message = models.TextField()
    contact_email = models.EmailField(null=True, blank=True)
    is_resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    class Meta:
        db_table = 'user_feedback'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.subject} - {self.created_at.date()}"

