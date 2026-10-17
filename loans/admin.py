from django.contrib import admin

from .models import LoanApplication


@admin.register(LoanApplication)
class LoanApplicationAdmin(admin.ModelAdmin):
    list_display = ["id", "created_at", "applicant_income", "loan_amount",
                    "credit_history", "approved", "probability"]
    list_filter = ["approved", "credit_history", "property_area"]
