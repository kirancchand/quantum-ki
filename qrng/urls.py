from django.urls import path
from qrng import views

urlpatterns = [
    path('',  views.index),
    path('home',  views.home),
    path('requesthandler/<id>/',  views.requesthandler),
    path('randomnumbergenerator',  views.randomnumbergenerator),
    path('smokerprediction',  views.smokerprediction),
    path('random', views.random),
    path('factor/', views.factor)
    ]