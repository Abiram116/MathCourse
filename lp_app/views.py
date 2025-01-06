from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import io
import base64
import json
import re
import matplotlib
matplotlib.use('Agg')

def index(request):
    return render(request, 'index.html')

def parse_expression(expr):
    # Remove spaces
    expr = expr.replace(' ', '')
    
    # Split by + or -
    terms = re.findall(r'[+-]?\d*\.?\d*[xy]|[+-]?\d+\.?\d*', expr)
    
    coefficients = [0, 0]  # [x_coeff, y_coeff]
    
    for term in terms:
        if 'x' in term:
            coeff = term.replace('x', '')
            coeff = 1 if coeff in ['+', ''] else -1 if coeff == '-' else float(coeff)
            coefficients[0] = coeff
        elif 'y' in term:
            coeff = term.replace('y', '')
            coeff = 1 if coeff in ['+', ''] else -1 if coeff == '-' else float(coeff)
            coefficients[1] = coeff
        
    return coefficients

def parse_constraint(expr):
    # Split by <= or >=
    parts = re.split(r'<=|>=|=', expr)
    left_side = parts[0]
    right_side = float(parts[1])
    
    coefficients = parse_expression(left_side)
    return coefficients + [right_side]

def solve_lp(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Parse objective function
            objective = parse_expression(data['objective'])
            
            # Parse constraints
            constraints = [parse_constraint(c) for c in data['constraints'].split(';')]
            constraints_matrix = np.array(constraints)
            
            # Set up the problem
            c = np.array(objective)
            if data['optimization_type'] == 'maximize':
                c = -c  # Negate for maximization
                
            A = constraints_matrix[:, :2]  # First two columns (x and y coefficients)
            b = constraints_matrix[:, 2]   # Last column (right hand side)
            
            # Add non-negativity constraints
            bounds = [(0, None), (0, None)]
            
            # Solve linear programming problem
            res = linprog(
                c=c,
                A_ub=A,
                b_ub=b,
                bounds=bounds,
                method='highs'
            )
            
            if not res.success:
                return JsonResponse({'error': 'Optimization failed'}, status=400)
            
            # Generate visualization
            plt.figure(figsize=(10, 8))
            
            # Plot feasible region
            x = np.linspace(0, max(20, np.max(b)/2), 200)
            
            # Plot constraints
            for i in range(len(b)):
                if A[i, 1] != 0:  # Avoid division by zero
                    y = (b[i] - A[i, 0] * x) / A[i, 1]
                    plt.plot(x, y, label=f'{A[i,0]}x + {A[i,1]}y ≤ {b[i]}')
            
            # Plot optimal point
            if res.success:
                plt.plot(res.x[0], res.x[1], 'r*', markersize=15, label='Optimal Point')
            
            # Customize plot
            plt.grid(True, alpha=0.3)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Linear Programming Solution')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set reasonable axis limits
            max_val = max(np.max(b), np.max(res.x)) * 1.2
            plt.xlim(0, max_val)
            plt.ylim(0, max_val)
            
            # Convert plot to base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            optimal_value = float(-res.fun if data['optimization_type'] == 'maximize' else res.fun)
            
            return JsonResponse({
                'solution': res.x.tolist(),
                'optimal_value': optimal_value,
                'image': image_base64
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)