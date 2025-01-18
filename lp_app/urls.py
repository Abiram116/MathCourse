# lp_app/urls.py

from django.urls import path
from .views import (
    home,
    graphical_solver,
    solve_lp
)

# URLs for Linear Programming app
urlpatterns = [
    path('', home, name='home'),
    path('graphical/', graphical_solver, name='graphical'),
    path('graphical/solve/', solve_lp, name='solve_lp'),
]