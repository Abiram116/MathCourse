from django.urls import path
from .views import (
    home,
    graphical_solver,
    solve_lp,
    simplex_solver_view,
    solve_simplex_lp,
    transportation_solver_view,
    solve_transportation,
    fractional_solver_view,
    solve_fractional
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
    # Transportation problem paths
    path('transportation/', transportation_solver_view, name='transportation'),
    path('transportation/solve/', solve_transportation, name='solve_transportation'),
    # Fractional programming paths
    path('fractional/', fractional_solver_view, name='fractional'),
    path('fractional/solve/', solve_fractional, name='solve_fractional'),
]