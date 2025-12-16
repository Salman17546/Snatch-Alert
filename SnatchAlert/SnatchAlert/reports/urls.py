from django.urls import path
from .views import CreateIncidentView, MyReportsListView, IncidentUpdateView, IncidentDeleteView

urlpatterns = [
    path("report/", CreateIncidentView.as_view(), name="incident-report"),
    path("my-reports/", MyReportsListView.as_view(), name="my-reports"),
    path("update/<int:pk>/", IncidentUpdateView.as_view(), name="incident-update"),
    path("delete/<int:pk>/", IncidentDeleteView.as_view(), name="incident-delete"),
]
