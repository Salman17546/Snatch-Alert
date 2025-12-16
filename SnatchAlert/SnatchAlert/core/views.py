from rest_framework import generics, permissions, filters
from django_filters.rest_framework import DjangoFilterBackend
from .models import SafetyTip, UserFeedback, IncidentTypeDim
from .serializers import SafetyTipSerializer, UserFeedbackSerializer, IncidentTypeDimSerializer


class SafetyTipListView(generics.ListAPIView):
    """List all active safety tips"""
    serializer_class = SafetyTipSerializer
    permission_classes = [permissions.AllowAny]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter]
    filterset_fields = ['category', 'is_active']
    search_fields = ['title', 'content', 'category']
    queryset = SafetyTip.objects.filter(is_active=True)


class SafetyTipCreateView(generics.CreateAPIView):
    """Create a safety tip (admin only)"""
    serializer_class = SafetyTipSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


class SafetyTipUpdateView(generics.UpdateAPIView):
    """Update a safety tip (admin only)"""
    serializer_class = SafetyTipSerializer
    permission_classes = [permissions.IsAuthenticated]
    queryset = SafetyTip.objects.all()


class SafetyTipDeleteView(generics.DestroyAPIView):
    """Delete a safety tip (admin only)"""
    permission_classes = [permissions.IsAuthenticated]
    queryset = SafetyTip.objects.all()


class UserFeedbackCreateView(generics.CreateAPIView):
    """Submit user feedback"""
    serializer_class = UserFeedbackSerializer
    permission_classes = [permissions.AllowAny]
    
    def perform_create(self, serializer):
        if self.request.user.is_authenticated:
            serializer.save(user=self.request.user)
        else:
            serializer.save()


class UserFeedbackListView(generics.ListAPIView):
    """List all feedback (admin only)"""
    serializer_class = UserFeedbackSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter]
    filterset_fields = ['is_resolved']
    search_fields = ['subject', 'message']
    queryset = UserFeedback.objects.all()


class IncidentTypeListView(generics.ListAPIView):
    """List all incident types"""
    serializer_class = IncidentTypeDimSerializer
    permission_classes = [permissions.AllowAny]
    queryset = IncidentTypeDim.objects.all()


class IncidentTypeCreateView(generics.CreateAPIView):
    """Create incident type (admin only)"""
    serializer_class = IncidentTypeDimSerializer
    permission_classes = [permissions.IsAuthenticated]
