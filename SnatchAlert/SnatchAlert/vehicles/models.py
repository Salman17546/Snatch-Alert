from django.db import models

# from django.db import models
# Create your models here.

class VehicleDim(models.Model):
    vehicle_type = models.CharField(max_length=50)   # car/bike/rickshaw
    company = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    registration_number = models.CharField(max_length=50, unique=True)
    color = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return f"{self.vehicle_type} - {self.registration_number}"
