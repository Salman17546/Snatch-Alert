from rest_framework import serializers
from django.utils import timezone
from django.db import transaction
from datetime import datetime

from core.models import LocationDim, TimeDim, VictimDim
from phones.models import PhoneDim
from vehicles.models import VehicleDim
from reports.models import IncidentFact

# -------------------------
# Small helper serializers
# -------------------------
class LocationInputSerializer(serializers.Serializer):
    province = serializers.CharField(max_length=100, required=False)
    city = serializers.CharField(max_length=100)
    district = serializers.CharField(max_length=100, required=False, allow_blank=True)
    neighborhood = serializers.CharField(max_length=100, required=False, allow_blank=True)
    street_address = serializers.CharField(max_length=255)
    latitude = serializers.DecimalField(max_digits=9, decimal_places=6)
    longitude = serializers.DecimalField(max_digits=9, decimal_places=6)


class VictimInputSerializer(serializers.Serializer):
    full_name = serializers.CharField(max_length=150, required=False, allow_blank=True)
    phone_number = serializers.CharField(max_length=20, required=False, allow_blank=True)
    # If you want email/cnic add here


class PhoneInputSerializer(serializers.Serializer):
    imei = serializers.CharField(max_length=20)
    brand = serializers.CharField(max_length=100, required=False, allow_blank=True)
    model = serializers.CharField(max_length=100, required=False, allow_blank=True)
    color = serializers.CharField(max_length=50, required=False, allow_blank=True)


class VehicleInputSerializer(serializers.Serializer):
    registration_number = serializers.CharField(max_length=50)
    vehicle_type = serializers.CharField(max_length=50, required=False, allow_blank=True)
    company = serializers.CharField(max_length=100, required=False, allow_blank=True)
    model = serializers.CharField(max_length=100, required=False, allow_blank=True)
    color = serializers.CharField(max_length=50, required=False, allow_blank=True)


# -------------------------
# Main Incident serializer (input)
# -------------------------
class IncidentReportSerializer(serializers.Serializer):
    """
    Serializer for creating an Incident (one stolen item per incident).
    """
    occurred_at = serializers.DateTimeField()
    incident_type = serializers.ChoiceField(choices=[c[0] for c in IncidentFact._meta.get_field('incident_type').choices])
    location = LocationInputSerializer()
    victim = VictimInputSerializer(required=False)
    # stolen_item will contain either phone or vehicle based on item_type
    item_type = serializers.ChoiceField(choices=("phone", "vehicle", "other"))
    phone = PhoneInputSerializer(required=False)
    vehicle = VehicleInputSerializer(required=False)
    description = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    is_anonymous = serializers.BooleanField(default=False)
    fir_filed = serializers.BooleanField(default=False)

    def validate(self, attrs):
        item_type = attrs.get("item_type")
        if item_type == "phone":
            if not attrs.get("phone"):
                raise serializers.ValidationError({"phone": "Phone data required when item_type is 'phone'."})
        elif item_type == "vehicle":
            if not attrs.get("vehicle"):
                raise serializers.ValidationError({"vehicle": "Vehicle data required when item_type is 'vehicle'."})
        return attrs

    @transaction.atomic
    def create(self, validated_data):
        request = self.context.get("request")
        user = getattr(request, "user", None)

        # 1) Deduplicate/create LocationDim (L3: city+district+neighborhood+street+lat+lng)
        loc = validated_data.pop("location")
        location_obj, _ = LocationDim.objects.get_or_create(
            city=loc.get("city"),
            district=loc.get("district") or "",
            neighborhood=loc.get("neighborhood") or "",
            street_address=loc.get("street_address"),
            latitude=loc.get("latitude"),
            longitude=loc.get("longitude"),
            defaults={
                "province": loc.get("province") if "province" in loc else "",
            }
        )

        # 2) TimeDim from occurred_at
        occurred_at = validated_data.get("occurred_at")
        # split to date and time
        date = occurred_at.date()
        time_of_day = occurred_at.time()
        day_of_week = date.strftime("%A")
        time_obj, _ = TimeDim.objects.get_or_create(
            date=date,
            time_of_day=time_of_day,
            defaults={"day_of_week": day_of_week}
        )

        # 3) Victim: link to existing VictimDim for this user if exists, otherwise create
        victim_data = validated_data.pop("victim", None) or {}
        if user and user.is_authenticated:
            victim_qs = VictimDim.objects.filter(user=user)
            if victim_qs.exists():
                victim = victim_qs.first()
                # optionally update victim fields if provided
                if victim_data.get("full_name"):
                    victim.full_name = victim_data.get("full_name")
                if victim_data.get("phone_number"):
                    victim.phone_number = victim_data.get("phone_number")
                victim.save()
            else:
                victim = VictimDim.objects.create(user=user,
                                                  full_name=victim_data.get("full_name") or "",
                                                  phone_number=victim_data.get("phone_number") or "")
        else:
            # anonymous victim
            victim = VictimDim.objects.create(user=None,
                                              full_name=victim_data.get("full_name") or "",
                                              phone_number=victim_data.get("phone_number") or "")

        # 4) Create or reuse phone/vehicle dimension depending on item_type
        item_type = validated_data.pop("item_type")
        phone_obj = None
        vehicle_obj = None

        if item_type == "phone":
            phone_data = validated_data.pop("phone")
            imei = phone_data.get("imei")
            # reuse by imei if exists, otherwise create
            phone_obj, _ = PhoneDim.objects.get_or_create(
                imei=imei,
                defaults={
                    "brand": phone_data.get("brand") or "",
                    "model": phone_data.get("model") or "",
                    "color": phone_data.get("color") or "",
                }
            )
        elif item_type == "vehicle":
            vehicle_data = validated_data.pop("vehicle")
            reg = vehicle_data.get("registration_number")
            vehicle_obj, _ = VehicleDim.objects.get_or_create(
                registration_number=reg,
                defaults={
                    "vehicle_type": vehicle_data.get("vehicle_type") or "",
                    "company": vehicle_data.get("company") or "",
                    "model": vehicle_data.get("model") or "",
                    "color": vehicle_data.get("color") or "",
                }
            )

        # 5) Finally create IncidentFact (fact) — use phone or vehicle FK fields accordingly
        incident = IncidentFact.objects.create(
            incident_type=validated_data.get("incident_type"),
            time=time_obj,
            location=location_obj,
            victim=victim,
            phone=phone_obj,
            vehicle=vehicle_obj,
            description=validated_data.get("description") or "",
            is_anonymous=validated_data.get("is_anonymous", False),
            fir_filed=validated_data.get("fir_filed", False),
            created_at=validated_data.get("occurred_at", timezone.now())  # adjust if you store occurred_at separately
        )

        return incident


# -------------------------
# Serializer for responses (reading)
# -------------------------
class IncidentFactReadSerializer(serializers.ModelSerializer):
    location = serializers.SerializerMethodField()
    victim = serializers.SerializerMethodField()
    phone = serializers.SerializerMethodField()
    vehicle = serializers.SerializerMethodField()
    time = serializers.SerializerMethodField()

    class Meta:
        model = IncidentFact
        fields = [
            "id",
            "incident_type",
            "time",
            "occurred_at",
            "location",
            "victim",
            "phone",
            "vehicle",
            "description",
            "is_anonymous",
            "fir_filed",
            "created_at",
        ]

    def get_location(self, obj):
        loc = obj.location
        return {
            "id": loc.id,
            "province": getattr(loc, "province", ""),
            "city": loc.city,
            "district": getattr(loc, "district", ""),
            "neighborhood": getattr(loc, "neighborhood", ""),
            "street_address": loc.street_address,
            "latitude": str(loc.latitude) if loc.latitude is not None else None,
            "longitude": str(loc.longitude) if loc.longitude is not None else None,
        }

    def get_victim(self, obj):
        v = obj.victim
        return {
            "id": v.id,
            "full_name": v.full_name,
            "phone_number": v.phone_number,
        }

    def get_phone(self, obj):
        if not obj.phone:
            return None
        p = obj.phone
        return {"id": p.id, "imei": p.imei, "brand": p.brand, "model": p.model, "color": p.color}

    def get_vehicle(self, obj):
        if not obj.vehicle:
            return None
        v = obj.vehicle
        return {"id": v.id, "registration_number": v.registration_number, "company": v.company, "model": v.model, "color": v.color}

    def get_time(self, obj):
        t = obj.time
        return {"id": t.id, "date": str(t.date), "time_of_day": str(t.time_of_day), "day_of_week": t.day_of_week}
