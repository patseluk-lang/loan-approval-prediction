import pandas as pd
from django.conf import settings
from django.test import TestCase
from django.urls import reverse
from django.utils.html import escape

from .forms import REQUIRED_MESSAGE, LoanApplicationForm
from .models import (APPLICANT_INCOME_RANGE, COAPPLICANT_INCOME_RANGE, LOAN_AMOUNT_RANGE,
                     LoanApplication)
from .predictor import predict

VALID_DATA = {
    "gender": "Male",
    "married": "Yes",
    "dependents": "0",
    "education": "Graduate",
    "self_employed": "No",
    "applicant_income": 4000,
    "coapplicant_income": 1500,
    "loan_amount": 120000,
    "loan_amount_term": 360,
    "credit_history": 1,
    "property_area": "Semiurban",
}


class LoanApplicationFormTests(TestCase):
    def test_valid_data(self):
        self.assertTrue(LoanApplicationForm(data=VALID_DATA).is_valid())

    def test_empty_form_marks_every_field_as_required(self):
        form = LoanApplicationForm(data={})
        self.assertFalse(form.is_valid())
        self.assertEqual(set(form.errors), set(VALID_DATA))
        for errors in form.errors.values():
            self.assertEqual(errors, [REQUIRED_MESSAGE])

    def test_negative_income_is_rejected(self):
        form = LoanApplicationForm(data={**VALID_DATA, "coapplicant_income": -100})
        self.assertFalse(form.is_valid())
        self.assertIn("coapplicant_income", form.errors)

    def test_values_outside_training_data_are_rejected(self):
        ranges = {
            "applicant_income": APPLICANT_INCOME_RANGE,
            "coapplicant_income": COAPPLICANT_INCOME_RANGE,
            "loan_amount": LOAN_AMOUNT_RANGE,
        }
        for field, (low, high) in ranges.items():
            for value in (low - 1, high + 1):
                with self.subTest(field=field, value=value):
                    form = LoanApplicationForm(data={**VALID_DATA, field: value})
                    self.assertFalse(form.is_valid())
                    self.assertIn(field, form.errors)

    def test_boundary_values_are_allowed(self):
        for index in (0, 1):
            form = LoanApplicationForm(data={
                **VALID_DATA,
                "applicant_income": APPLICANT_INCOME_RANGE[index],
                "coapplicant_income": COAPPLICANT_INCOME_RANGE[index],
                "loan_amount": LOAN_AMOUNT_RANGE[index],
            })
            self.assertTrue(form.is_valid(), form.errors)

    def test_ranges_match_training_data(self):
        df = pd.read_csv(settings.BASE_DIR / "loan_data.csv")
        self.assertEqual(APPLICANT_INCOME_RANGE,
                         (df["ApplicantIncome"].min(), df["ApplicantIncome"].max()))
        self.assertEqual(COAPPLICANT_INCOME_RANGE,
                         (df["CoapplicantIncome"].min(), df["CoapplicantIncome"].max()))
        # У даних сума в тисячах, у формі - повністю.
        self.assertEqual(LOAN_AMOUNT_RANGE,
                         (df["LoanAmount"].min() * 1000, df["LoanAmount"].max() * 1000))

    def test_zero_coapplicant_income_is_allowed(self):
        form = LoanApplicationForm(data={**VALID_DATA, "coapplicant_income": 0})
        self.assertTrue(form.is_valid())

    def test_text_instead_of_number_is_rejected(self):
        form = LoanApplicationForm(data={**VALID_DATA, "loan_amount": "сто"})
        self.assertFalse(form.is_valid())
        self.assertIn("loan_amount", form.errors)

    def test_unknown_choice_is_rejected(self):
        form = LoanApplicationForm(data={**VALID_DATA, "loan_amount_term": 999})
        self.assertFalse(form.is_valid())
        self.assertIn("loan_amount_term", form.errors)


class PredictorTests(TestCase):
    def test_returns_decision_and_probability(self):
        approved, probability = predict(LoanApplication(**VALID_DATA))
        self.assertIsInstance(approved, bool)
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)

    def test_bad_credit_history_lowers_probability(self):
        # Найсильніша ознака в даних: без кредитної історії схвалюють лише 5% заявок.
        _, good = predict(LoanApplication(**VALID_DATA))
        _, bad = predict(LoanApplication(**{**VALID_DATA, "credit_history": 0}))
        self.assertLess(bad, good)


class ApplicationViewTests(TestCase):
    def test_form_page_opens(self):
        response = self.client.get(reverse("loans:form"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "обов'язкові для заповнення")

    def test_valid_application_is_saved_and_redirects_to_result(self):
        response = self.client.post(reverse("loans:form"), VALID_DATA)
        application = LoanApplication.objects.get()
        self.assertRedirects(response, reverse("loans:result", args=[application.pk]))

    def test_empty_post_shows_required_messages_and_saves_nothing(self):
        response = self.client.post(reverse("loans:form"), {})
        self.assertEqual(response.status_code, 200)
        # Апостроф у тексті помилки шаблон екранує, тому шукаю екрановану версію.
        self.assertContains(response, escape(REQUIRED_MESSAGE), count=len(VALID_DATA))
        self.assertEqual(LoanApplication.objects.count(), 0)

    def test_result_page_shows_decision(self):
        self.client.post(reverse("loans:form"), VALID_DATA)
        application = LoanApplication.objects.get()
        response = self.client.get(reverse("loans:result", args=[application.pk]))
        expected = "Займ буде схвалено" if application.approved else "У займі буде відмовлено"
        self.assertContains(response, expected)

    def test_missing_result_returns_404(self):
        response = self.client.get(reverse("loans:result", args=[999]))
        self.assertEqual(response.status_code, 404)

    def test_history_lists_applications(self):
        self.client.post(reverse("loans:form"), VALID_DATA)
        self.client.post(reverse("loans:form"), {**VALID_DATA, "credit_history": 0})
        response = self.client.get(reverse("loans:history"))
        self.assertEqual(len(response.context["applications"]), 2)

    def test_empty_history(self):
        response = self.client.get(reverse("loans:history"))
        self.assertContains(response, "Заявок ще немає")

    def test_about_page_shows_metrics(self):
        response = self.client.get(reverse("loans:about"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Матриця помилок")
        self.assertContains(response, "charts/confusion_matrix.png")
        self.assertContains(response, "charts/feature_importance.png")
