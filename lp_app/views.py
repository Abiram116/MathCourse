from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import io
import base64
import json
import re
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
    # Identify the inequality sign and swap it
    if '>=' in expr:
        expr = expr.replace('>=', '<=')
        sign_multiplier = -1  # We will multiply the coefficients by -1
    
    elif '<=' in expr:
        # expr = expr.replace('<=', '>=')
        sign_multiplier = 1  # We will multiply the coefficients by 1
        
    else:
        raise ValueError("Invalid constraint format")

    parts = re.split(r'<=|>=|=', expr)
    if len(parts) != 2:
        raise ValueError("Invalid constraint format")

    left_side = parts[0]
    right_side = float(parts[1])

    # Parse the expression for the left side
    coefficients = parse_expression(left_side)
    
    # Apply the sign multiplier to both coefficients and the right-hand side value
    coefficients = [sign_multiplier * coeff for coeff in coefficients]
    right_side *= sign_multiplier
    
    return coefficients + [right_side]


def get_vertices(A, b):
    vertices = []
    num_constraints = len(b)
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            A_sub = np.array([A[i], A[j]])
            b_sub = np.array([b[i], b[j]])
            if np.linalg.det(A_sub) != 0:  # Ensure not parallel
                vertex = np.linalg.solve(A_sub, b_sub)
                if all(np.dot(A, vertex) <= b) and all(vertex >= 0):  # Feasibility check
                    vertices.append(vertex)
    return np.array(vertices)

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

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(16, 12))
            
            colors = ['#10B981', '#3B82F6', '#8B5CF6', '#F59E0B', '#EC4899']
            max_x = max(20, np.max(b) / 2)
            x = np.linspace(0, max_x, 400)

            def get_y_limits(a_i, b_i, x):
                if a_i[1] != 0:
                    return (b_i - a_i[0] * x) / a_i[1]
                return b_i / a_i[0]

            y_limits = []
            for a_i, b_i in zip(A, b):
                y_limits.append(get_y_limits(a_i, b_i, x))

            y_lower_bound = np.maximum(0, np.max(y_limits, axis=0))
            y_upper_bound = np.minimum(15, np.min(y_limits, axis=0))

            plt.fill_between(x, y_lower_bound, y_upper_bound, where=y_upper_bound >= y_lower_bound, color='pink', alpha=0.5, label='Feasible Region')

            for i, (a_i, b_i) in enumerate(zip(A, b)):
                if a_i[1] != 0:
                    y = (b_i - a_i[0] * x) / a_i[1]
                    plt.plot(x, y, color=colors[i % len(colors)], label=f'{a_i[0]}x + {a_i[1]}y ≤ {b_i}', linewidth=2.5)
                else:
                    plt.axvline(x=b_i / a_i[0], color=colors[i % len(colors)], label=f'x ≤ {b_i / a_i[0]}', linewidth=2.5)

            vertices = get_vertices(A, b)
            if len(vertices) > 2:
                hull = ConvexHull(vertices)
                feasible_polygon = Polygon(vertices[hull.vertices], alpha=0.3, color='#94A3B8', label='Feasible Region')
                ax.add_patch(feasible_polygon)

            plt.plot(res.x[0], res.x[1], 'o', color='#F472B6', markersize=15, label='Optimal Point')
            plt.annotate(f'({res.x[0]:.2f}, {res.x[1]:.2f})', (res.x[0], res.x[1]), 
                        xytext=(10, 10), textcoords='offset points', 
                        color='white', fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='#1F2937', edgecolor='#4B5563', alpha=0.8))

            ax.grid(True, linestyle='--', alpha=0.2, color='#4B5563')
            ax.set_facecolor('#111827')
            fig.patch.set_facecolor('#111827')
            
            plt.xlabel('x', color='white', fontsize=14, fontweight='medium')
            plt.ylabel('y', color='white', fontsize=14, fontweight='medium')
            plt.title('Linear Programming Solution', color='white', fontsize=16, pad=20, fontweight='bold')
            
            legend = plt.legend(loc='upper left', frameon=True, 
                                facecolor='#1F2937', edgecolor='#4B5563',
                                fontsize=12)
            plt.setp(legend.get_texts(), color='white')

            for spine in ax.spines.values():
                spine.set_color('#4B5563')

            ax.tick_params(colors='white', labelsize=12)
            
            max_val = max(np.max(b), np.max(res.x)) * 1.2
            plt.xlim(0, max_val)
            plt.ylim(0, max_val)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, 
                       facecolor='#111827', edgecolor='none')
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
