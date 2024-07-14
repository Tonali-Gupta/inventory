from django.urls import path
from RecommendationSystemML import views

urlpatterns = [
    path('',views.home),
    path('predict/',views.predict)
]
