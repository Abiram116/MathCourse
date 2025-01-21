# lp_app/urls.py

from django.urls import path
from .views import (
    home,
    graphical_solver,
    solve_lp,
    simplex_solver_view,
    solve_simplex_lp
)

# URLs for Linear Programming app
urlpatterns = [
    path('', home, name='home'),
    # Graphical method paths
    path('graphical/', graphical_solver, name='graphical'),
    path('graphical/solve/', solve_lp, name='solve_lp'),
    # Simplex method paths
    path('simplex/', simplex_solver_view, name='simplex'),
    path('simplex/solve/', solve_simplex_lp, name='solve_simplex_lp'),
]