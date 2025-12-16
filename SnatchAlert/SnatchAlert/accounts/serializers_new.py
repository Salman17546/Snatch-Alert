from rest_framework import serializers
from django.contrib.auth import get_user_model
from rest_framework.validators import UniqueValidator
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth import authenticate

User = get_user_model()


class RegisterSerializer(serializers.ModelSerializer):
    """Serializer for user registration with email only (no username)"""
    email = serializers.EmailField(
        required=True,
        validators=[UniqueValidator(queryset=User.objects.all(), message="This email is already registered.")]
    )
    password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password],
        style={'input_type': 'password'}
    )
    password2 = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'},
        label="Confirm Password"
    )
    
    class Meta:
        model = User
        fields = ['email', 'password', 'password2', 'phone', 'first_name', 'last_name']
        extra_kwargs = {
            'first_name': {'required': False},
            'last_name': {'required': False},
            'phone': {'required': False}
        }
    
    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        return attrs
    
    def create(self, validated_data):
        validated_data.pop('password2')
        user = User.objects.create_user(**validated_data)
        return user


class EmailLoginSerializer(serializers.Serializer):
    """Serializer for email-based login (no username)"""
    email = serializers.EmailField(required=True)
    password = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'}
    )
    
    def validate(self, attrs):
        email = attrs.get('email')
        password = attrs.get('password')
        
        if email and password:
            # Try to get user by email
            try:
                user = User.objects.get(email=email)
                # Check password directly since we don't have username
                if not user.check_password(password):
                    raise serializers.ValidationError("Invalid email or password.")
            except User.DoesNotExist:
                raise serializers.ValidationError("Invalid email or password.")
            
            if not user.is_active:
                raise serializers.ValidationError("User account is disabled.")
            
            attrs['user'] = user
        else:
            raise serializers.ValidationError("Must include 'email' and 'password'.")
        
        return attrs


class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for viewing user profile"""
    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'phone', 'role', 'address', 'is_verified', 'date_joined']
        read_only_fields = ['id', 'role', 'is_verified', 'date_joined']


class UpdateEmailSerializer(serializers.Serializer):
    """Serializer for updating email"""
    new_email = serializers.EmailField(required=True)
    password = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'}
    )
    
    def validate_new_email(self, value):
        user = self.context['request'].user
        if User.objects.filter(email=value).exclude(id=user.id).exists():
            raise serializers.ValidationError("This email is already in use.")
        return value
    
    def validate(self, attrs):
        user = self.context['request'].user
        password = attrs.get('password')
        
        if not user.check_password(password):
            raise serializers.ValidationError({"password": "Current password is incorrect."})
        
        return attrs


class UpdatePasswordSerializer(serializers.Serializer):
    """Serializer for updating password"""
    old_password = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'}
    )
    new_password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password],
        style={'input_type': 'password'}
    )
    new_password2 = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'},
        label="Confirm New Password"
    )
    
    def validate_old_password(self, value):
        user = self.context['request'].user
        if not user.check_password(value):
            raise serializers.ValidationError("Old password is incorrect.")
        return value
    
    def validate(self, attrs):
        if attrs['new_password'] != attrs['new_password2']:
            raise serializers.ValidationError({"new_password": "New password fields didn't match."})
        return attrs


class PasswordResetRequestSerializer(serializers.Serializer):
    """Serializer for password reset request"""
    email = serializers.EmailField(required=True)


class PasswordResetVerifySerializer(serializers.Serializer):
    """Serializer for verifying password reset token"""
    token = serializers.CharField(required=True)


class PasswordResetConfirmSerializer(serializers.Serializer):
    """Serializer for password reset confirmation"""
    token = serializers.CharField(required=True)
    new_password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password],
        style={'input_type': 'password'}
    )
    new_password2 = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'},
        label="Confirm New Password"
    )
    
    def validate(self, attrs):
        if attrs['new_password'] != attrs['new_password2']:
            raise serializers.ValidationError({"new_password": "Password fields didn't match."})
        return attrs
