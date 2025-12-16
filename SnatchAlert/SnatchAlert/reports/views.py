from django.shortcuts import render

# Create your views here.
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .serializers import IncidentReportSerializer, IncidentFactReadSerializer
from .models import IncidentFact
from .permissions import IsReporter

# All endpoints require authentication (Option B): enforce IsAuthenticated globally or per-view
class CreateIncidentView(generics.CreateAPIView):
    serializer_class = IncidentReportSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_context(self):
        ctx = super().get_serializer_context()
        ctx["request"] = self.request
        return ctx


class MyReportsListView(generics.ListAPIView):
    serializer_class = IncidentFactReadSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # return incidents where victim.user == request.user
        return IncidentFact.objects.filter(victim__user=self.request.user).order_by("-created_at")


class IncidentUpdateView(generics.UpdateAPIView):
    serializer_class = IncidentReportSerializer  # allow partial updates via serializer
    permission_classes = [permissions.IsAuthenticated, IsReporter]
    queryset = IncidentFact.objects.all()
    http_method_names = ["patch", "put"]

    def get_serializer_context(self):
        ctx = super().get_serializer_context()
        ctx["request"] = self.request
        return ctx


class IncidentDeleteView(generics.DestroyAPIView):
    permission_classes = [permissions.IsAuthenticated, IsReporter]
    queryset = IncidentFact.objects.all()
