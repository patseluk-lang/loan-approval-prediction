from django.shortcuts import get_object_or_404, redirect, render

from .forms import LoanApplicationForm
from .models import LoanApplication
from .predictor import load_metrics, predict


def application_form(request):
    if request.method == "POST":
        form = LoanApplicationForm(request.POST)
        if form.is_valid():
            application = form.save(commit=False)
            application.approved, application.probability = predict(application)
            application.save()
            # Після POST перенаправляю на сторінку результату, щоб оновлення сторінки
            # у браузері не відправило ту саму заявку вдруге.
            return redirect("loans:result", pk=application.pk)
    else:
        form = LoanApplicationForm()
    return render(request, "loans/form.html", {"form": form})


def application_result(request, pk):
    application = get_object_or_404(LoanApplication, pk=pk)
    return render(request, "loans/result.html", {"application": application})


def application_history(request):
    applications = LoanApplication.objects.all()
    return render(request, "loans/history.html", {"applications": applications})


def about_model(request):
    return render(request, "loans/about.html", {"metrics": load_metrics()})
