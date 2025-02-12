# lp_app/views/__init__.py
from lp_app.views.base_views import (
    home
)

from lp_app.views.graphical_views import (
    graphical_solver,
    solve_lp
)

from lp_app.views.simplex_views import (
    simplex_solver_view,
    solve_simplex_lp
)

from lp_app.views.transportation_views import (
    transportation_solver_view,
    solve_transportation
)

__all__ = [
    'home',
    'graphical_solver',
    'solve_lp',
    'simplex_solver_view',
    'solve_simplex_lp',
    'transportation_solver_view',
    'solve_transportation'
]