from django.urls import path
from .views import (
    SafetyTipListView, SafetyTipCreateView, SafetyTipUpdateView, SafetyTipDeleteView,
    UserFeedbackCreateView, UserFeedbackListView,
    IncidentTypeListView, IncidentTypeCreateView
)

urlpatterns = [
    # Safety Tips
    path('safety-tips/', SafetyTipListView.as_view(), name='safety-tip-list'),
    path('safety-tips/create/', SafetyTipCreateView.as_view(), name='safety-tip-create'),
    path('safety-tips/<int:pk>/update/', SafetyTipUpdateView.as_view(), name='safety-tip-update'),
    path('safety-tips/<int:pk>/delete/', SafetyTipDeleteView.as_view(), name='safety-tip-delete'),
    
    # User Feedback
    path('feedback/', UserFeedbackCreateView.as_view(), name='feedback-create'),
    path('feedback/list/', UserFeedbackListView.as_view(), name='feedback-list'),
    
    # Incident Types
    path('incident-types/', IncidentTypeListView.as_view(), name='incident-type-list'),
    path('incident-types/create/', IncidentTypeCreateView.as_view(), name='incident-type-create'),
]
