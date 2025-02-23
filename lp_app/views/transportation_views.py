from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
import numpy as np
import json
from scipy.optimize import linprog

def home(request):
    return render(request, 'home.html')

@ensure_csrf_cookie
def transportation_solver_view(request):
    return render(request, 'transportation.html')

@ensure_csrf_cookie
def solve_transportation(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    
    try:
        data = json.loads(request.body)
        
        cost_matrix = np.array(data['costMatrix'], dtype=float)
        supply = np.array(data['supply'], dtype=float)
        demand = np.array(data['demand'], dtype=float)
        
        m, n = cost_matrix.shape
        if len(supply) != m or len(demand) != n:
            return JsonResponse({'error': "Dimensions don't match"}, status=400)
            
        if not np.isclose(sum(supply), sum(demand)):
            return JsonResponse({'error': "Supply must equal demand"}, status=400)

        # Solve using linear programming
        c = cost_matrix.flatten()
        A_eq = []
        b_eq = []

        # Supply constraints
        for i in range(m):
            row = np.zeros(m * n)
            row[i * n:(i + 1) * n] = 1
            A_eq.append(row)
            b_eq.append(supply[i])

        # Demand constraints
        for j in range(n):
            col = np.zeros(m * n)
            col[j::n] = 1
            A_eq.append(col)
            b_eq.append(demand[j])

        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0, None),
            method='highs'
        )

        if not result.success:
            return JsonResponse({
                'error': f'No solution found: {result.message}'
            }, status=400)

        solution = result.x.reshape(m, n).tolist()
        total_cost = float(result.fun)
        
        return JsonResponse({
            'solution': solution,
            'total_cost': total_cost,
            'status': 'Success'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)