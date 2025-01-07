from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import io
import base64
import json
import re
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for matplotlib

def index(request):
    return render(request, 'index.html')

def parse_expression(expr):
    expr = expr.replace(' ', '')
    terms = re.findall(r'[+-]?\d*\.?\d*[xy]|[+-]?\d+\.?\d*', expr)
    coefficients = [0, 0]
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
    parts = re.split(r'<=|>=|=', expr)
    if len(parts) != 2:
        raise ValueError("Invalid constraint format")
    left_side = parts[0]
    right_side = float(parts[1])
    coefficients = parse_expression(left_side)
    return coefficients + [right_side]

def solve_lp(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            objective = parse_expression(data['objective'])
            constraints = [parse_constraint(c) for c in data['constraints'].split(';') if c.strip()]
            constraints_matrix = np.array(constraints)

            c = np.array(objective)
            if data['optimization_type'] == 'maximize':
                c = -c

            A = constraints_matrix[:, :2]
            b = constraints_matrix[:, 2]
            bounds = [(0, None), (0, None)]

            res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if not res.success:
                return JsonResponse({'error': 'Optimization failed: ' + res.message}, status=400)

            plt.figure(figsize=(10, 8))

            # Define a range for x
            x = np.linspace(0, max(20, np.max(b) / 2), 400)

            # Plot constraints and feasible region
            for i in range(len(b)):
                if A[i, 1] != 0:
                    y = (b[i] - A[i, 0] * x) / A[i, 1]
                    plt.plot(x, y, label=f'{A[i,0]}x + {A[i,1]}y ≤ {b[i]}', linewidth=2.0)
                else:
                    plt.axvline(x=b[i] / A[i, 0], color='blue', linestyle='--', label=f'x ≤ {b[i] / A[i,0]}')

            # Generate vertices for the feasible region
            vertices = [[0, 0]]
            for i in range(len(b)):
                if A[i, 1] != 0:
                    y1 = (b[i] - A[i, 0] * 0) / A[i, 1]
                    y2 = (b[i] - A[i, 0] * (np.max(b))) / A[i, 1]
                    vertices.append([0, y1])
                    vertices.append([np.max(b), y2])

            # Filter only valid points within bounds
            vertices = np.array([v for v in vertices if v[0] >= 0 and v[1] >= 0])

            # Check if vertices form a valid 2D region
            if len(vertices) > 2 and not np.allclose(vertices[:, 0], vertices[0, 0]):
                hull = ConvexHull(vertices)
                feasible_polygon = Polygon(vertices[hull.vertices], alpha=0.3, color='green', label='Feasible Region')
                plt.gca().add_patch(feasible_polygon)
            else:
                plt.fill(vertices[:, 0], vertices[:, 1], color='green', alpha=0.3, label='Feasible Region (Degenerate)')

            # Plot the optimal point
            plt.plot(res.x[0], res.x[1], 'ro', markersize=10, label='Optimal Point')
            plt.text(res.x[0], res.x[1], f'({res.x[0]:.2f}, {res.x[1]:.2f})', fontsize=12, color='red', weight='bold')

            # Style the graph
            plt.grid(True, alpha=0.3)
            plt.xlabel('x', fontsize=14)
            plt.ylabel('y', fontsize=14)
            plt.title('Linear Programming Solution', fontsize=16, weight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.tight_layout()

            # Dynamically adjust axis limits
            max_val = max(np.max(b), np.max(res.x)) * 1.2
            plt.xlim(0, max_val)
            plt.ylim(0, max_val)

            # Save the graph to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
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
