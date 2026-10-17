"""Звернення до навченої моделі.

Модель завантажується один раз при першому прогнозі і далі береться з пам'яті:
читати файл з диска на кожен запит було б повільно.
"""
import json
from functools import cache

import joblib
import pandas as pd
from django.conf import settings


@cache
def load_model():
    return joblib.load(settings.ML_MODEL_PATH)


def load_metrics():
    return json.loads(settings.ML_METRICS_PATH.read_text(encoding="utf-8"))


def predict(application):
    """Повертає (схвалено, ймовірність схвалення) для заявки LoanApplication.

    Назви колонок мають збігатися з тими, на яких навчалась модель:
    за ними ColumnTransformer знаходить потрібні ознаки.
    Суму займу у формі вводять повністю, а модель навчена на тисячах,
    тому ділю її на 1000.
    """
    row = pd.DataFrame([{
        "ApplicantIncome": application.applicant_income,
        "CoapplicantIncome": application.coapplicant_income,
        "LoanAmount": application.loan_amount / 1000,
        "Gender": application.gender,
        "Married": application.married,
        "Dependents": application.dependents,
        "Education": application.education,
        "Self_Employed": application.self_employed,
        "Property_Area": application.property_area,
        "Credit_History": application.credit_history,
        "Loan_Amount_Term": application.loan_amount_term,
    }])
    probability = load_model().predict_proba(row)[0][1]
    return bool(probability >= 0.5), round(float(probability), 3)
