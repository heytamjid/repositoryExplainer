from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),  # default route
    path("ask/", views.ask_question, name="ask_question"),
    path("api/ask/", views.api_ask, name="api_ask"),
]
