from django.urls import path

from . import views

app_name = "loans"

urlpatterns = [
    path("", views.application_form, name="form"),
    path("applications/<int:pk>/", views.application_result, name="result"),
    path("history/", views.application_history, name="history"),
    path("about/", views.about_model, name="about"),
]
