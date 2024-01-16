from django.urls import path
from . import views
urlpatterns=[
    path('word-prediction', views.Welcome),
    path('predict/', views.predict, name='predict')




]