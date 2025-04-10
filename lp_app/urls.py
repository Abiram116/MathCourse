from django.urls import path
from .views import (
    home,
    graphical_solver,
    solve_lp,
    simplex_solver_view,
    solve_simplex_lp,
    transportation_solver_view,
    solve_transportation,
    integer_solver_view,
    solve_integer,
    fractional_solver_view,
    solve_fractional,
    nonlinear_solver_view,
    solve_nonlinear
)
from .views.base_views import dijkstra_view, shortest_path

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
    # Integer programming paths
    path('integer/', integer_solver_view, name='integer'),
    path('integer/solve/', solve_integer, name='solve_integer'),
    # Fractional programming paths
    path('fractional/', fractional_solver_view, name='fractional'),
    path('fractional/solve/', solve_fractional, name='solve_fractional'),
    # Dijkstra's algorithm paths
    path('dijkstra/', dijkstra_view, name='dijkstra'),
    path('api/shortest_path/', shortest_path, name='shortest_path'),
    # Nonlinear programming paths
    path('nonlinear/', nonlinear_solver_view, name='nonlinear'),
    path('nonlinear/solve/', solve_nonlinear, name='solve_nonlinear'),
]
