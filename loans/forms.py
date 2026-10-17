from django import forms

from .models import (APPLICANT_INCOME_RANGE, COAPPLICANT_INCOME_RANGE, LOAN_AMOUNT_RANGE,
                     LoanApplication)

RANGES = {
    "applicant_income": APPLICANT_INCOME_RANGE,
    "coapplicant_income": COAPPLICANT_INCOME_RANGE,
    "loan_amount": LOAN_AMOUNT_RANGE,
}

REQUIRED_MESSAGE = "Це поле обов'язкове."


class LoanApplicationForm(forms.ModelForm):
    class Meta:
        model = LoanApplication
        fields = [
            "gender", "married", "dependents", "education", "self_employed",
            "applicant_income", "coapplicant_income", "loan_amount", "loan_amount_term",
            "credit_history", "property_area",
        ]
        help_texts = {
            "applicant_income": "Від {} до {}.".format(*APPLICANT_INCOME_RANGE),
            "coapplicant_income": ("Від {} до {}. Якщо співзаявника немає, вкажіть 0."
                                   .format(*COAPPLICANT_INCOME_RANGE)),
            "loan_amount": "Від {} до {}.".format(*LOAN_AMOUNT_RANGE),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.error_messages["required"] = REQUIRED_MESSAGE
            # Перший порожній пункт у списках Django підписує англійською, замінюю його.
            if isinstance(field, forms.ChoiceField):
                field.choices = [("", "Оберіть варіант")] + list(field.choices)[1:]
        # Межі в самому полі вводу: стрілки браузера не вийдуть за діапазон даних.
        for name, (low, high) in RANGES.items():
            self.fields[name].widget.attrs.update(min=low, max=high)
