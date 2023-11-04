from django.urls import path
from . import views

urlpatterns = [
    path('', views.show_resume, name='show_resume'),
    path('predict/', views.show_digit, name='show_digit'),
]
