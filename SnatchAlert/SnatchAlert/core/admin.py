from django.contrib import admin
from .models import LocationDim, VictimDim, IncidentTypeDim, StolenItemDim, SafetyTip, UserFeedback


@admin.register(LocationDim)
class LocationDimAdmin(admin.ModelAdmin):
    list_display = ['id', 'city', 'district', 'neighborhood', 'latitude', 'longitude']
    list_filter = ['city', 'district']
    search_fields = ['city', 'district', 'neighborhood', 'street_address']


@admin.register(VictimDim)
class VictimDimAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'age', 'gender', 'phone_number', 'user']
    list_filter = ['gender']
    search_fields = ['name', 'phone_number', 'email']


@admin.register(IncidentTypeDim)
class IncidentTypeDimAdmin(admin.ModelAdmin):
    list_display = ['id', 'category', 'description']
    search_fields = ['category', 'description']


@admin.register(StolenItemDim)
class StolenItemDimAdmin(admin.ModelAdmin):
    list_display = ['id', 'item_type', 'imei', 'phone_brand', 'license_plate', 'vehicle_make']
    list_filter = ['item_type']
    search_fields = ['imei', 'phone_brand', 'phone_model', 'license_plate', 'vehicle_make']


@admin.register(SafetyTip)
class SafetyTipAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'category', 'is_active', 'created_by', 'created_at']
    list_filter = ['is_active', 'category', 'created_at']
    search_fields = ['title', 'content']


@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ['id', 'subject', 'user', 'contact_email', 'is_resolved', 'created_at']
    list_filter = ['is_resolved', 'created_at']
    search_fields = ['subject', 'message', 'contact_email']
