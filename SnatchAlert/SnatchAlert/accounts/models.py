from django.db import models

# Create your models here.
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from django.utils import timezone

class CustomUserManager(BaseUserManager):
    """Custom user manager for email-based authentication"""
    
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user
    
    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('role', 'admin')
        
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        
        return self.create_user(email, password, **extra_fields)


class CustomUser(AbstractBaseUser, PermissionsMixin):
    """Custom user model with email-based authentication (no username)"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('admin', 'Admin'),
        ('authority', 'Authority'),
    ]
    
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=150, blank=True)
    last_name = models.CharField(max_length=150, blank=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')
    is_verified = models.BooleanField(default=False)
    address = models.TextField(blank=True)
    
    # Required for Django admin
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    date_joined = models.DateTimeField(default=timezone.now)
    
    objects = CustomUserManager()
    
    # Use email as the username field
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []  # No additional required fields
    
    class Meta:
        db_table = 'custom_user'

    def __str__(self):
        return self.email
    
    def get_full_name(self):
        return f"{self.first_name} {self.last_name}".strip() or self.email
    
    def get_short_name(self):
        return self.first_name or self.email
    
    def is_admin_or_authority(self):
        return self.role in ['admin', 'authority']


class PasswordResetToken(models.Model):
    """Model to store password reset tokens"""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='reset_tokens')
    token = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_used = models.BooleanField(default=False)
    
    class Meta:
        db_table = 'password_reset_tokens'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Reset token for {self.user.email}"
    
    def is_valid(self):
        """Check if token is still valid"""
        from django.utils import timezone
        return not self.is_used and self.expires_at > timezone.now()
