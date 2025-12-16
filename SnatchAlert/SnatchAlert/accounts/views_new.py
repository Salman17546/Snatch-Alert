from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.core.mail import send_mail
from django.conf import settings
from datetime import timedelta
import secrets
from .models import PasswordResetToken
from .serializers_new import (
    RegisterSerializer, EmailLoginSerializer, UserProfileSerializer,
    UpdateEmailSerializer, UpdatePasswordSerializer,
    PasswordResetRequestSerializer, PasswordResetVerifySerializer,
    PasswordResetConfirmSerializer
)

User = get_user_model()


class RegisterView(generics.CreateAPIView):
    """User registration with email"""
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        # Generate JWT tokens
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'user': {
                'id': user.id,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'phone': user.phone,
            },
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            },
            'message': 'Registration successful'
        }, status=status.HTTP_201_CREATED)


class EmailLoginView(APIView):
    """Login with email and password"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = EmailLoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        user = serializer.validated_data['user']
        
        # Generate JWT tokens
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'user': {
                'id': user.id,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'phone': user.phone,
                'role': user.role,
            },
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            },
            'message': 'Login successful'
        }, status=status.HTTP_200_OK)


class UserProfileView(generics.RetrieveUpdateAPIView):
    """Get and update user profile (basic info only)"""
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_object(self):
        return self.request.user


class UpdateEmailView(APIView):
    """Update user email"""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        serializer = UpdateEmailSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        
        user = request.user
        new_email = serializer.validated_data['new_email']
        
        # Update email
        user.email = new_email
        user.is_verified = False  # Require re-verification
        user.save()
        
        return Response({
            'message': 'Email updated successfully',
            'user': {
                'id': user.id,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'phone': user.phone,
            }
        }, status=status.HTTP_200_OK)


class UpdatePasswordView(APIView):
    """Update user password"""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        serializer = UpdatePasswordSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        
        user = request.user
        new_password = serializer.validated_data['new_password']
        
        # Update password
        user.set_password(new_password)
        user.save()
        
        return Response({
            'message': 'Password updated successfully'
        }, status=status.HTTP_200_OK)


class PasswordResetRequestView(APIView):
    """Request password reset - generates token and sends email"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = PasswordResetRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        email = serializer.validated_data['email']
        
        try:
            user = User.objects.get(email=email)
            
            # Generate secure token
            token = secrets.token_urlsafe(32)
            
            # Create reset token record
            reset_token = PasswordResetToken.objects.create(
                user=user,
                token=token,
                expires_at=timezone.now() + timedelta(hours=1)  # Valid for 1 hour
            )
            
            # Create reset link
            reset_link = f"{settings.FRONTEND_URL}/reset-password?token={token}"
            
            # Send email
            try:
                send_mail(
                    subject='Password Reset Request - SnatchAlert',
                    message=f'''
Hello,

You requested to reset your password for SnatchAlert.

Click the link below to reset your password:
{reset_link}

This link will expire in 1 hour.

If you didn't request this, please ignore this email.

Best regards,
SnatchAlert Team
                    ''',
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[email],
                    fail_silently=False,
                )
                
                return Response({
                    'message': 'Password reset link sent to your email',
                    'token': token  # Remove this in production, only for testing
                }, status=status.HTTP_200_OK)
                
            except Exception as e:
                # If email fails, still return success but log error
                print(f"Email sending failed: {str(e)}")
                return Response({
                    'message': 'Password reset link sent to your email',
                    'token': token  # For testing when email is not configured
                }, status=status.HTTP_200_OK)
                
        except User.DoesNotExist:
            # Don't reveal if email exists (security best practice)
            return Response({
                'message': 'Password reset link sent to your email'
            }, status=status.HTTP_200_OK)


class PasswordResetVerifyView(APIView):
    """Verify password reset token"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = PasswordResetVerifySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        token = serializer.validated_data['token']
        
        try:
            reset_token = PasswordResetToken.objects.get(token=token)
            
            if not reset_token.is_valid():
                return Response({
                    'valid': False,
                    'message': 'Token is invalid or expired'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            return Response({
                'valid': True,
                'message': 'Token is valid',
                'email': reset_token.user.email
            }, status=status.HTTP_200_OK)
            
        except PasswordResetToken.DoesNotExist:
            return Response({
                'valid': False,
                'message': 'Token is invalid'
            }, status=status.HTTP_400_BAD_REQUEST)


class PasswordResetConfirmView(APIView):
    """Confirm password reset with token and new password"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = PasswordResetConfirmSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        token = serializer.validated_data['token']
        new_password = serializer.validated_data['new_password']
        
        try:
            reset_token = PasswordResetToken.objects.get(token=token)
            
            if not reset_token.is_valid():
                return Response({
                    'message': 'Token is invalid or expired'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Update password
            user = reset_token.user
            user.set_password(new_password)
            user.save()
            
            # Mark token as used
            reset_token.is_used = True
            reset_token.save()
            
            return Response({
                'message': 'Password reset successful. You can now login with your new password.'
            }, status=status.HTTP_200_OK)
            
        except PasswordResetToken.DoesNotExist:
            return Response({
                'message': 'Invalid token'
            }, status=status.HTTP_400_BAD_REQUEST)
