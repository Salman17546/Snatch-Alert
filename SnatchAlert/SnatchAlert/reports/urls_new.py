from django.urls import path
from .views_new import (
    IncidentCreateView, IncidentListView, IncidentDetailView,
    IncidentUpdateView, IncidentDeleteView, MyIncidentsView,
    IMEIRegisterView, IMEICheckView, IMEIListView, IMEIUpdateView,
    CrimeHeatmapView, AreaSafetyScoreView, CrimeStatisticsView,
    AreaAlertListView, AreaAlertCreateView, AreaAlertUpdateView, AreaAlertDeleteView,
    MyDeviceAlertsView, MarkAlertReadView, MarkAllAlertsReadView, IMEICheckHistoryView
)

urlpatterns = [
    # Incident Management
    path('incidents/', IncidentListView.as_view(), name='incident-list'),
    path('incidents/create/', IncidentCreateView.as_view(), name='incident-create'),
    path('incidents/<int:pk>/', IncidentDetailView.as_view(), name='incident-detail'),
    path('incidents/<int:pk>/update/', IncidentUpdateView.as_view(), name='incident-update'),
    path('incidents/<int:pk>/delete/', IncidentDeleteView.as_view(), name='incident-delete'),
    path('incidents/my/', MyIncidentsView.as_view(), name='my-incidents'),
    
    # IMEI Tracking
    path('imei/register/', IMEIRegisterView.as_view(), name='imei-register'),
    path('imei/check/', IMEICheckView.as_view(), name='imei-check'),
    path('imei/list/', IMEIListView.as_view(), name='imei-list'),
    path('imei/<int:pk>/update/', IMEIUpdateView.as_view(), name='imei-update'),
    
    # IMEI Alerts & History
    path('imei/alerts/', MyDeviceAlertsView.as_view(), name='my-device-alerts'),
    path('imei/alerts/<int:alert_id>/read/', MarkAlertReadView.as_view(), name='mark-alert-read'),
    path('imei/alerts/read-all/', MarkAllAlertsReadView.as_view(), name='mark-all-alerts-read'),
    path('imei/check-history/', IMEICheckHistoryView.as_view(), name='imei-check-history'),
    
    # Crime Analytics & Heatmap
    path('heatmap/', CrimeHeatmapView.as_view(), name='crime-heatmap'),
    path('safety-score/', AreaSafetyScoreView.as_view(), name='area-safety-score'),
    path('statistics/', CrimeStatisticsView.as_view(), name='crime-statistics'),
    
    # Area Alerts
    path('alerts/', AreaAlertListView.as_view(), name='alert-list'),
    path('alerts/create/', AreaAlertCreateView.as_view(), name='alert-create'),
    path('alerts/<int:pk>/update/', AreaAlertUpdateView.as_view(), name='alert-update'),
    path('alerts/<int:pk>/delete/', AreaAlertDeleteView.as_view(), name='alert-delete'),
]
