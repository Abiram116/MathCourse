# lp_app/views/__init__.py

from .base_views import (
    home
)


from .graphical_views import (
    graphical_solver,
    solve_lp
)

# This makes the views available when importing from lp_app.views
__all__ = [
    'home',
    'graphical_solver',
    'solve_lp'
]