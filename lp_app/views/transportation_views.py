from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
import numpy as np
import json
from scipy.optimize import linprog
from typing import Dict, Union, List, Optional
from numpy.typing import NDArray

@ensure_csrf_cookie
def transportation_solver_view(request) -> render:
    """
    Renders the transportation solver page.
    
    Args:
        request: Django HTTP request object
        
    Returns:
        Rendered transportation.html template
    """
    return render(request, 'transportation.html')

@ensure_csrf_cookie
def solve_transportation(request) -> JsonResponse:
    """
    Handles POST requests to solve transportation problems.
    
    Args:
        request: Django HTTP request object containing JSON data with:
            - costMatrix: 2D array of transportation costs
            - supply: Array of supply quantities
            - demand: Array of demand quantities
            
    Returns:
        JsonResponse with either:
            - solution: 2D array of optimal transportation quantities
            - total_cost: Total minimum cost
            - status: Solution status message
        OR
            - error: Error message if solution fails
    """
    if request.method != 'POST':
        return JsonResponse(
            {'error': 'Invalid request method'}, 
            status=405
        )
    
    try:
        # Parse request data
        data = json.loads(request.body)
        
        # Convert inputs to numpy arrays
        cost_matrix = np.array(data['costMatrix'], dtype=float)
        supply = np.array(data['supply'], dtype=float)
        demand = np.array(data['demand'], dtype=float)
        
        # Validate input dimensions
        m, n = cost_matrix.shape
        if len(supply) != m or len(demand) != n:
            raise ValueError("Supply/demand dimensions don't match cost matrix")
            
        # Check if problem is balanced
        if not np.isclose(sum(supply), sum(demand)):
            raise ValueError("Total supply must equal total demand")

        # Solve the transportation problem
        result = solve_transportation_problem(cost_matrix, supply, demand)
        
        if result["solution"] is None:
            return JsonResponse({
                'error': f'No feasible solution found: {result["status"]}'
            }, status=400)

        # Convert numpy arrays to lists for JSON serialization
        return JsonResponse({
            'solution': result["solution"].tolist(),
            'total_cost': float(result["total_cost"]),
            'status': result["status"]
        })
        
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        return JsonResponse({
            'error': f'An error occurred: {str(e)}'
        }, status=400)

def solve_transportation_problem(
    cost_matrix: NDArray,
    supply: NDArray,
    demand: NDArray
) -> Dict[str, Optional[Union[NDArray, float, str]]]:
    """
    Solves the transportation problem using linear programming.

    The problem minimizes the total transportation cost while satisfying
    supply and demand constraints.

    Args:
        cost_matrix: 2D array (m x n) where m is number of sources and 
                    n is number of destinations. Each element represents 
                    the cost of transporting one unit.
        supply: 1D array of length m containing supply capacity for each source
        demand: 1D array of length n containing demand requirement for each destination

    Returns:
        Dictionary containing:
            - solution: 2D array of optimal transportation quantities (None if no solution)
            - total_cost: Minimum total transportation cost (None if no solution)
            - status: Solution status message
    """
    # Get problem dimensions
    m, n = cost_matrix.shape

    # Flatten cost matrix for linprog
    c = cost_matrix.flatten()

    # Initialize constraint matrices
    A_eq = []
    b_eq = []

    # Supply constraints (row-wise)
    for i in range(m):
        constraint = np.zeros(m * n)
        constraint[i * n:(i + 1) * n] = 1
        A_eq.append(constraint)
        b_eq.append(supply[i])

    # Demand constraints (column-wise)
    for j in range(n):
        constraint = np.zeros(m * n)
        constraint[j::n] = 1
        A_eq.append(constraint)
        b_eq.append(demand[j])

    # Convert constraints to numpy arrays
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # Solve using linprog with HiGHS solver
    result = linprog(
        c=c,
        A_eq=A_eq, 
        b_eq=b_eq,
        bounds=(0, None),
        method='highs'
    )

    # Return formatted results
    if result.success:
        return {
            "solution": result.x.reshape(m, n),
            "total_cost": result.fun,
            "status": result.message
        }
    
    return {
        "solution": None,
        "total_cost": None,
        "status": result.message
    }