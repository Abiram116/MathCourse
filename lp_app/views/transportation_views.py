from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods
import numpy as np
import json
from scipy.optimize import linprog

@ensure_csrf_cookie
def transportation_solver_view(request):
    """
    Render the transportation solver page
    """
    return render(request, 'transportation.html')

@ensure_csrf_cookie
@require_http_methods(["POST"])
def solve_transportation(request):
    """
    API endpoint to solve the transportation problem
    """
    try:
        # Parse JSON data from request
        data = json.loads(request.body)
        
        # Extract and validate input data
        cost_matrix = np.array(data['costMatrix'], dtype=float)
        supply = np.array(data['supply'], dtype=float)
        demand = np.array(data['demand'], dtype=float)
        
        # Validate dimensions
        m, n = cost_matrix.shape
        if len(supply) != m or len(demand) != n:
            return JsonResponse({
                'error': "Supply/demand dimensions don't match cost matrix dimensions"
            }, status=400)
            
        # Validate balanced problem
        if not np.isclose(sum(supply), sum(demand)):
            return JsonResponse({
                'error': "Total supply must equal total demand"
            }, status=400)

        # Prepare linear programming constraints
        c = cost_matrix.flatten()  # Objective function coefficients
        A_eq = []  # Equality constraint matrix
        b_eq = []  # Equality constraint bounds

        # Supply constraints (row sums)
        for i in range(m):
            row = np.zeros(m * n)
            row[i * n:(i + 1) * n] = 1
            A_eq.append(row)
            b_eq.append(supply[i])

        # Demand constraints (column sums)
        for j in range(n):
            col = np.zeros(m * n)
            col[j::n] = 1
            A_eq.append(col)
            b_eq.append(demand[j])

        # Convert to numpy arrays
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

        # Solve linear programming problem
        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0, None),
            method='highs'  # Using the HiGHS solver
        )

        if not result.success:
            return JsonResponse({
                'error': f'Failed to find optimal solution: {result.message}'
            }, status=400)

        # Format solution
        allocation_matrix = result.x.reshape(m, n).tolist()
        optimal_cost = float(result.fun)
        
        # Return solution
        return JsonResponse({
            'solution': {
                'allocation': allocation_matrix,
                'total_cost': optimal_cost,
                'iterations': result.nit,
                'status': 'Optimal solution found'
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON data'
        }, status=400)
    except ValueError as e:
        return JsonResponse({
            'error': f'Invalid input data: {str(e)}'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': f'An unexpected error occurred: {str(e)}'
        }, status=500)