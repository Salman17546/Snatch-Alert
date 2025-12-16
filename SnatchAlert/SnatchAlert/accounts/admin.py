from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import CustomUser, PasswordResetToken


@admin.register(CustomUser)
class CustomUserAdmin(BaseUserAdmin):
    list_display = ['email', 'first_name', 'last_name', 'role', 'is_verified', 'is_staff', 'date_joined']
    list_filter = ['role', 'is_verified', 'is_staff', 'is_superuser', 'date_joined']
    search_fields = ['email', 'first_name', 'last_name', 'phone']
    ordering = ['-date_joined']
    
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal Info', {'fields': ('first_name', 'last_name', 'phone', 'address')}),
        ('Permissions', {'fields': ('role', 'is_verified', 'is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'first_name', 'last_name', 'phone', 'role', 'is_staff', 'is_superuser'),
        }),
    )


@admin.register(PasswordResetToken)
class PasswordResetTokenAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'token', 'created_at', 'expires_at', 'is_used']
    list_filter = ['is_used', 'created_at', 'expires_at']
    search_fields = ['user__email', 'token']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'
