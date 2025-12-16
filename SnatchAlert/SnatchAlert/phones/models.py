from django.db import models

# from django.db import models
# Create your models here.

class PhoneDim(models.Model):
    brand = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    imei = models.CharField(max_length=20, unique=True)
    color = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return f"{self.brand} {self.model} ({self.imei})"
