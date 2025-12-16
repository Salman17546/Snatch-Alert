from rest_framework import serializers
from .models import LocationDim, VictimDim, IncidentTypeDim, StolenItemDim, SafetyTip, UserFeedback


class LocationDimSerializer(serializers.ModelSerializer):
    class Meta:
        model = LocationDim
        fields = ['id', 'province', 'city', 'district', 'neighborhood', 'street_address', 'latitude', 'longitude']


class VictimDimSerializer(serializers.ModelSerializer):
    class Meta:
        model = VictimDim
        fields = ['id', 'name', 'age', 'gender', 'phone_number', 'email', 'address']
        extra_kwargs = {
            'name': {'required': False},
            'age': {'required': False},
            'gender': {'required': False},
        }


class IncidentTypeDimSerializer(serializers.ModelSerializer):
    class Meta:
        model = IncidentTypeDim
        fields = ['id', 'category', 'description']


class StolenItemDimSerializer(serializers.ModelSerializer):
    class Meta:
        model = StolenItemDim
        fields = [
            'id', 'item_type', 'description', 'value_estimate',
            'imei', 'phone_brand', 'phone_model',
            'license_plate', 'chassis_number', 'vehicle_make', 'vehicle_model'
        ]
    
    def validate(self, attrs):
        item_type = attrs.get('item_type')
        
        if item_type == 'phone':
            if not attrs.get('imei'):
                raise serializers.ValidationError({"imei": "IMEI is required for phone items"})
        elif item_type in ['car', 'bike']:
            if not attrs.get('license_plate'):
                raise serializers.ValidationError({"license_plate": "License plate is required for vehicles"})
        
        return attrs


class SafetyTipSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = SafetyTip
        fields = ['id', 'title', 'content', 'category', 'is_active', 'created_by', 'created_by_username', 'created_at', 'updated_at']
        read_only_fields = ['created_by', 'created_at', 'updated_at']


class UserFeedbackSerializer(serializers.ModelSerializer):
    user_username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = UserFeedback
        fields = ['id', 'user', 'user_username', 'subject', 'message', 'contact_email', 'is_resolved', 'created_at']
        read_only_fields = ['user', 'created_at']
