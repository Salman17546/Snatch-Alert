from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import timedelta
from core.models import LocationDim, VictimDim, IncidentTypeDim, StolenItemDim, SafetyTip
from reports.models import IncidentFact, IMEIRegistry, AreaAlert

User = get_user_model()


class Command(BaseCommand):
    help = 'Seed database with sample data for SnatchAlert'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS('🌱 Starting seed data generation...'))

        # Create users
        self.stdout.write('\n👤 Creating users...')
        admin_user, created = User.objects.get_or_create(
            email='admin@snatchalert.com',
            defaults={
                'role': 'admin',
                'is_staff': True,
                'is_superuser': True,
                'is_verified': True,
                'first_name': 'Admin',
                'last_name': 'User'
            }
        )
        if created:
            admin_user.set_password('admin123')
            admin_user.save()
            self.stdout.write(self.style.SUCCESS('✓ Admin user created'))

        authority_user, created = User.objects.get_or_create(
            email='officer@police.gov',
            defaults={
                'role': 'authority',
                'is_verified': True,
                'first_name': 'Police',
                'last_name': 'Officer'
            }
        )
        if created:
            authority_user.set_password('police123')
            authority_user.save()
            self.stdout.write(self.style.SUCCESS('✓ Authority user created'))

        regular_user, created = User.objects.get_or_create(
            email='john@example.com',
            defaults={
                'role': 'user',
                'phone': '+923001234567',
                'first_name': 'John',
                'last_name': 'Doe'
            }
        )
        if created:
            regular_user.set_password('user123')
            regular_user.save()
            self.stdout.write(self.style.SUCCESS('✓ Regular user created'))

        # Create incident types
        self.stdout.write('\n📋 Creating incident types...')
        incident_types_data = [
            {'category': 'Mobile Snatching', 'description': 'Theft of mobile phones'},
            {'category': 'Vehicle Theft', 'description': 'Theft of cars, bikes, or other vehicles'},
            {'category': 'Robbery', 'description': 'Armed robbery or mugging'},
            {'category': 'Burglary', 'description': 'Breaking and entering'},
            {'category': 'Pickpocketing', 'description': 'Theft from pockets or bags'},
        ]

        for data in incident_types_data:
            IncidentTypeDim.objects.get_or_create(category=data['category'], defaults=data)
        self.stdout.write(self.style.SUCCESS(f'✓ Created {len(incident_types_data)} incident types'))

        # Create locations
        self.stdout.write('\n📍 Creating locations...')
        locations_data = [
            {'province': 'Punjab', 'city': 'Lahore', 'district': 'Gulberg', 'neighborhood': 'MM Alam Road', 'street_address': 'Main Boulevard', 'latitude': 31.5204, 'longitude': 74.3587},
            {'province': 'Punjab', 'city': 'Lahore', 'district': 'Model Town', 'neighborhood': 'Block A', 'street_address': 'Link Road', 'latitude': 31.4824, 'longitude': 74.3045},
            {'province': 'Sindh', 'city': 'Karachi', 'district': 'Clifton', 'neighborhood': 'Block 5', 'street_address': 'Khayaban-e-Ittehad', 'latitude': 24.8138, 'longitude': 67.0299},
            {'province': 'Sindh', 'city': 'Karachi', 'district': 'Saddar', 'neighborhood': 'Empress Market', 'street_address': 'MA Jinnah Road', 'latitude': 24.8607, 'longitude': 67.0011},
            {'province': 'KPK', 'city': 'Peshawar', 'district': 'Hayatabad', 'neighborhood': 'Phase 1', 'street_address': 'Main Road', 'latitude': 33.9911, 'longitude': 71.4969},
            {'province': 'Punjab', 'city': 'Islamabad', 'district': 'F-7', 'neighborhood': 'Markaz', 'street_address': 'Jinnah Avenue', 'latitude': 33.7215, 'longitude': 73.0433},
        ]

        locations = []
        for data in locations_data:
            loc, _ = LocationDim.objects.get_or_create(
                city=data['city'],
                district=data['district'],
                neighborhood=data['neighborhood'],
                defaults=data
            )
            locations.append(loc)
        self.stdout.write(self.style.SUCCESS(f'✓ Created {len(locations)} locations'))

        # Create victims
        self.stdout.write('\n👥 Creating victims...')
        victims_data = [
            {'user': regular_user, 'name': 'John Doe', 'age': 28, 'gender': 'male', 'phone_number': '+923001234567'},
            {'name': 'Anonymous Victim 1', 'age': 35, 'gender': 'female'},
            {'name': 'Ahmad Khan', 'age': 42, 'gender': 'male', 'phone_number': '+923217654321'},
        ]

        victims = []
        for data in victims_data:
            victim = VictimDim.objects.create(**data)
            victims.append(victim)
        self.stdout.write(self.style.SUCCESS(f'✓ Created {len(victims)} victims'))

        # Create stolen items
        self.stdout.write('\n📱 Creating stolen items...')
        stolen_items_data = [
            {'item_type': 'phone', 'imei': '123456789012345', 'phone_brand': 'Samsung', 'phone_model': 'Galaxy S21', 'value_estimate': 75000},
            {'item_type': 'phone', 'imei': '987654321098765', 'phone_brand': 'iPhone', 'phone_model': '13 Pro', 'value_estimate': 150000},
            {'item_type': 'car', 'license_plate': 'ABC-123', 'vehicle_make': 'Honda', 'vehicle_model': 'Civic 2020', 'value_estimate': 3500000},
            {'item_type': 'bike', 'license_plate': 'XYZ-789', 'vehicle_make': 'Honda', 'vehicle_model': 'CD 70', 'value_estimate': 85000},
        ]

        stolen_items = []
        for data in stolen_items_data:
            item = StolenItemDim.objects.create(**data)
            stolen_items.append(item)
        self.stdout.write(self.style.SUCCESS(f'✓ Created {len(stolen_items)} stolen items'))

        # Create incidents
        self.stdout.write('\n🚨 Creating incidents...')
        incidents_data = [
            {
                'occurred_at': timezone.now() - timedelta(days=2),
                'location': locations[0],
                'victim': victims[0],
                'incident_type': IncidentTypeDim.objects.get(category='Mobile Snatching'),
                'stolen_item': stolen_items[0],
                'value_estimate': 75000,
                'fir_filed': True,
                'description': 'Phone snatched at gunpoint near MM Alam Road',
                'status': 'reported',
                'reported_by': regular_user
            },
            {
                'occurred_at': timezone.now() - timedelta(days=5),
                'location': locations[2],
                'victim': victims[1],
                'incident_type': IncidentTypeDim.objects.get(category='Mobile Snatching'),
                'stolen_item': stolen_items[1],
                'value_estimate': 150000,
                'fir_filed': False,
                'description': 'iPhone stolen from car in Clifton',
                'is_anonymous': True,
                'status': 'reported'
            },
            {
                'occurred_at': timezone.now() - timedelta(days=10),
                'location': locations[1],
                'victim': victims[2],
                'incident_type': IncidentTypeDim.objects.get(category='Vehicle Theft'),
                'stolen_item': stolen_items[2],
                'value_estimate': 3500000,
                'fir_filed': True,
                'description': 'Car stolen from parking lot in Model Town',
                'status': 'investigating'
            },
        ]

        for data in incidents_data:
            IncidentFact.objects.create(**data)
        self.stdout.write(self.style.SUCCESS(f'✓ Created {len(incidents_data)} incidents'))

        # Create IMEI registry entries
        self.stdout.write('\n📲 Creating IMEI registry...')
        imei_data = [
            {
                'imei': '123456789012345',
                'phone_brand': 'Samsung',
                'phone_model': 'Galaxy S21',
                'owner_name': 'John Doe',
                'owner_contact': '+923001234567',
                'status': 'stolen',
                'reported_by': regular_user
            },
            {
                'imei': '987654321098765',
                'phone_brand': 'iPhone',
                'phone_model': '13 Pro',
                'status': 'stolen'
            },
        ]

        for data in imei_data:
            IMEIRegistry.objects.get_or_create(imei=data['imei'], defaults=data)
        self.stdout.write(self.style.SUCCESS(f'✓ Created {len(imei_data)} IMEI records'))

        # Create area alerts
        self.stdout.write('\n⚠️ Creating area alerts...')
        alerts_data = [
            {
                'location': locations[0],
                'alert_type': 'high_crime',
                'message': 'High crime rate reported in Gulberg area. Stay vigilant.',
                'severity': 'high',
                'is_active': True,
                'valid_from': timezone.now(),
                'created_by': authority_user
            },
            {
                'location': locations[2],
                'alert_type': 'recent_incident',
                'message': 'Multiple phone snatching incidents reported in Clifton.',
                'severity': 'medium',
                'is_active': True,
                'valid_from': timezone.now(),
                'created_by': authority_user
            },
        ]

        for data in alerts_data:
            AreaAlert.objects.create(**data)
        self.stdout.write(self.style.SUCCESS(f'✓ Created {len(alerts_data)} area alerts'))

        # Create safety tips
        self.stdout.write('\n💡 Creating safety tips...')
        tips_data = [
            {
                'title': 'Keep Your Phone Secure',
                'content': 'Always keep your phone in your front pocket or bag. Avoid using it while walking on busy streets.',
                'category': 'Mobile Safety',
                'is_active': True,
                'created_by': admin_user
            },
            {
                'title': 'Vehicle Security Tips',
                'content': 'Always lock your vehicle and park in well-lit areas. Install a GPS tracker for added security.',
                'category': 'Vehicle Safety',
                'is_active': True,
                'created_by': admin_user
            },
            {
                'title': 'Stay Alert in Crowded Places',
                'content': 'Be aware of your surroundings in crowded areas. Keep valuables close and avoid displaying expensive items.',
                'category': 'General Safety',
                'is_active': True,
                'created_by': admin_user
            },
        ]

        for data in tips_data:
            SafetyTip.objects.create(**data)
        self.stdout.write(self.style.SUCCESS(f'✓ Created {len(tips_data)} safety tips'))

        self.stdout.write(self.style.SUCCESS('\n✅ Seed data generation completed!'))
        self.stdout.write(self.style.WARNING('\n📝 Login credentials:'))
        self.stdout.write('   Admin: username=admin, password=admin123')
        self.stdout.write('   Authority: username=police_officer, password=police123')
        self.stdout.write('   User: username=john_doe, password=user123')
