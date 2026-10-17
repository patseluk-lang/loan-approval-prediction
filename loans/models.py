from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

# Діапазони значень у loan_data.csv. Поза ними модель даних не бачила: ліс дерев
# сприйме і 500000, і 150000 однаково - як "більше за найбільший відомий поріг",
# тож прогноз для таких значень нічого не означав би.
APPLICANT_INCOME_RANGE = (150, 9703)
COAPPLICANT_INCOME_RANGE = (0, 33837)
# У даних сума займу записана в тисячах (так сказано в описі набору даних),
# а у формі її вводять повністю: 9 тис. = 9000.
LOAN_AMOUNT_RANGE = (9000, 150000)


def in_range(bounds, what):
    low, high = bounds
    message = f"Модель навчена на {what} від {low} до {high}; інше значення вона оцінити не може."
    return [MinValueValidator(low, message), MaxValueValidator(high, message)]


class LoanApplication(models.Model):
    """Заявка на займ: введені дані і рішення моделі.

    Значення полів-виборів збігаються з тими, що в loan_data.csv,
    тому їх можна напряму передати в модель.
    """

    GENDER_CHOICES = [("Male", "Чоловік"), ("Female", "Жінка")]
    YES_NO_CHOICES = [("Yes", "Так"), ("No", "Ні")]
    DEPENDENTS_CHOICES = [("0", "0"), ("1", "1"), ("2", "2"), ("3+", "3 і більше")]
    EDUCATION_CHOICES = [("Graduate", "Вища освіта"), ("Not Graduate", "Без вищої освіти")]
    CREDIT_HISTORY_CHOICES = [(1, "Відповідає вимогам"), (0, "Не відповідає вимогам")]
    PROPERTY_AREA_CHOICES = [("Urban", "Місто"), ("Semiurban", "Передмістя"), ("Rural", "Село")]
    # Строки, які трапляються в даних. Іншого строку модель не бачила.
    LOAN_TERM_CHOICES = [(term, f"{term} міс.")
                         for term in (12, 36, 60, 84, 120, 180, 240, 300, 360, 480)]

    gender = models.CharField("Стать", max_length=6, choices=GENDER_CHOICES)
    married = models.CharField("У шлюбі", max_length=3, choices=YES_NO_CHOICES)
    dependents = models.CharField("Кількість утриманців", max_length=2, choices=DEPENDENTS_CHOICES)
    education = models.CharField("Освіта", max_length=12, choices=EDUCATION_CHOICES)
    self_employed = models.CharField("Самозайнятий", max_length=3, choices=YES_NO_CHOICES)
    # В описі набору даних одиниці доходів не вказані, тому підписую їх без одиниць.
    applicant_income = models.PositiveIntegerField(
        "Дохід заявника", validators=in_range(APPLICANT_INCOME_RANGE, "доходах"))
    coapplicant_income = models.PositiveIntegerField(
        "Дохід співзаявника", validators=in_range(COAPPLICANT_INCOME_RANGE, "доходах"))
    loan_amount = models.PositiveIntegerField(
        "Сума займу", validators=in_range(LOAN_AMOUNT_RANGE, "сумах займу"))
    loan_amount_term = models.PositiveSmallIntegerField("Строк займу", choices=LOAN_TERM_CHOICES)
    credit_history = models.PositiveSmallIntegerField("Кредитна історія",
                                                      choices=CREDIT_HISTORY_CHOICES)
    property_area = models.CharField("Район нерухомості", max_length=9,
                                     choices=PROPERTY_AREA_CHOICES)

    approved = models.BooleanField("Схвалено")
    probability = models.FloatField("Ймовірність схвалення")
    created_at = models.DateTimeField("Дата заявки", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Заявка на займ"
        verbose_name_plural = "Заявки на займ"

    def __str__(self):
        result = "схвалено" if self.approved else "відмова"
        return f"Заявка №{self.pk}: {result}"
