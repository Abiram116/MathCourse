from django.urls import path
from .views import index, solve_lp

urlpatterns = [
    path('', index, name='index'),
    path('solve/', solve_lp, name='solve_lp'),
]