"""
URL configuration for SnatchAlert project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

# Swagger/OpenAPI documentation setup
schema_view = get_schema_view(
    openapi.Info(
        title="SnatchAlert API",
        default_version='v1',
        description="""
        SnatchAlert - Crime Reporting & Tracking System API
        
        Features:
        - Crime incident reporting with file uploads
        - IMEI tracking for stolen phones
        - Crime heatmaps and area safety scores
        - Community safety tips
        - Location-based alerts
        - User authentication with JWT
        """,
        terms_of_service="https://www.snatchalert.com/terms/",
        contact=openapi.Contact(email="support@snatchalert.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # API Documentation
    path('', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('api/docs/', schema_view.with_ui('swagger', cache_timeout=0), name='api-docs'),
    path('api/redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='api-redoc'),
    path('api/schema/', schema_view.without_ui(cache_timeout=0), name='api-schema'),
    
    # Authentication
    path('api/auth/', include('accounts.urls_new')),
    
    # Core features (safety tips, feedback, incident types)
    path('api/core/', include('core.urls')),
    
    # Reports (incidents, IMEI, heatmap, alerts)
    path('api/reports/', include('reports.urls_new')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
