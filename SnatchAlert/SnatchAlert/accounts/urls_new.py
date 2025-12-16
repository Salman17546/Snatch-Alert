from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from .views_new import (
    RegisterView, EmailLoginView, UserProfileView,
    UpdateEmailView, UpdatePasswordView,
    PasswordResetRequestView, PasswordResetVerifyView, PasswordResetConfirmView
)

urlpatterns = [
    # Authentication
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', EmailLoginView.as_view(), name='email-login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    
    # User Profile
    path('profile/', UserProfileView.as_view(), name='user-profile'),
    path('profile/update-email/', UpdateEmailView.as_view(), name='update-email'),
    path('profile/update-password/', UpdatePasswordView.as_view(), name='update-password'),
    
    # Password Reset Flow
    path('password-reset/request/', PasswordResetRequestView.as_view(), name='password-reset-request'),
    path('password-reset/verify/', PasswordResetVerifyView.as_view(), name='password-reset-verify'),
    path('password-reset/confirm/', PasswordResetConfirmView.as_view(), name='password-reset-confirm'),
]
