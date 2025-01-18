from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
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

def home(request):
    return render(request, 'home.html')

@ensure_csrf_cookie
def graphical_solver(request):
    return render(request, 'graphical.html')

def parse_expression(expr):
    # Remove whitespace
    expr = expr.replace(' ', '')
    
    # Find all terms that match coefficient and variable patterns
    terms = re.findall(r'[+-]?[\d.]*[xy]|[+-]?[\d.]+', expr)
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
    # Handle different inequality signs
    if '>=' in expr:
        expr = expr.replace('>=', '<=')
        sign_multiplier = -1
    elif '<=' in expr:
        sign_multiplier = 1
    else:
        raise ValueError("Invalid constraint format. Must use <= or >=")

    # Split at inequality sign
    parts = re.split(r'<=|>=|=', expr)
    if len(parts) != 2:
        raise ValueError("Invalid constraint format. Must be in the form: ax + by <= c")

    left_side = parts[0].strip()
    try:
        right_side = float(parts[1].strip())
    except ValueError:
        raise ValueError("Right-hand side must be a number")

    coefficients = parse_expression(left_side)
    coefficients = [sign_multiplier * coeff for coeff in coefficients]
    right_side *= sign_multiplier

    return coefficients + [right_side]

def get_vertices(A, b):
    vertices = []
    num_constraints = len(b)
    
    # Add origin if it's feasible
    if all(bi >= 0 for bi in b):
        vertices.append([0, 0])
    
    # Add axis intersections if they're feasible
    for i in range(len(A[0])):
        for j, (a_row, b_val) in enumerate(zip(A, b)):
            if a_row[i] != 0:
                point = [0, 0]
                point[i] = b_val / a_row[i]
                if point[i] >= 0 and all(np.dot(A[k], point) <= b[k] for k in range(len(A))):
                    vertices.append(point)

    # Find intersection points of constraint lines
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            A_sub = np.array([A[i], A[j]])
            b_sub = np.array([b[i], b[j]])
            try:
                if np.linalg.det(A_sub) != 0:
                    vertex = np.linalg.solve(A_sub, b_sub)
                    if all(np.dot(A, vertex) <= b + 1e-10) and all(vertex >= 0):  # Added small tolerance
                        vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue
                
    return np.array(vertices)

@ensure_csrf_cookie
def solve_lp(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from request body
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid JSON data'}, status=400)

            # Validate required fields
            if not all(key in data for key in ['optimization_type', 'objective', 'constraints']):
                return JsonResponse({'error': 'Missing required fields'}, status=400)

            # Parse objective function and constraints
            try:
                objective = parse_expression(data['objective'])
                constraints = [parse_constraint(c) for c in data['constraints'].split(';') if c.strip()]
                if not constraints:
                    raise ValueError("No valid constraints provided")
            except ValueError as e:
                return JsonResponse({'error': str(e)}, status=400)

            constraints_matrix = np.array(constraints)
            
            # Set up optimization problem
            c = np.array(objective)
            if data['optimization_type'] == 'maximize':
                c = -c

            A = constraints_matrix[:, :2]
            b = constraints_matrix[:, 2]
            bounds = [(0, None), (0, None)]

            # Solve LP problem
            try:
                res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
                if not res.success:
                    return JsonResponse({'error': 'Optimization failed: ' + res.message}, status=400)
            except Exception as e:
                return JsonResponse({'error': f'Optimization error: {str(e)}'}, status=400)

            # Create visualization
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(16, 12))

            colors = ['#10B981', '#3B82F6', '#8B5CF6', '#F59E0B', '#EC4899']
            max_x = max(20, np.max(b) / 2)
            x = np.linspace(0, max_x, 400)

            # Plot constraints
            for i, (a_i, b_i) in enumerate(zip(A, b)):
                if a_i[1] != 0:
                    y = (b_i - a_i[0] * x) / a_i[1]
                    plt.plot(x, y, color=colors[i % len(colors)], 
                            label=f'{a_i[0]}x + {a_i[1]}y ≤ {b_i}', linewidth=2.5)
                else:
                    plt.axvline(x=b_i / a_i[0], color=colors[i % len(colors)], 
                               label=f'x ≤ {b_i / a_i[0]}', linewidth=2.5)

            # Plot feasible region
            vertices = get_vertices(A, b)
            if len(vertices) > 2:
                hull = ConvexHull(vertices)
                feasible_polygon = Polygon(vertices[hull.vertices], alpha=0.3, 
                                         color='#94A3B8', label='Feasible Region')
                ax.add_patch(feasible_polygon)

            # Plot optimal point
            plt.plot(res.x[0], res.x[1], 'o', color='#F472B6', 
                    markersize=15, label='Optimal Point')
            plt.annotate(f'({res.x[0]:.2f}, {res.x[1]:.2f})', 
                        (res.x[0], res.x[1]),
                        xytext=(10, 10), 
                        textcoords='offset points',
                        color='white', 
                        fontsize=12, 
                        fontweight='bold',
                        bbox=dict(facecolor='#1F2937', 
                                edgecolor='#4B5563', 
                                alpha=0.8))

            # Customize plot appearance
            ax.grid(True, linestyle='--', alpha=0.2, color='#4B5563')
            ax.set_facecolor('#111827')
            fig.patch.set_facecolor('#111827')

            plt.xlabel('x', color='white', fontsize=14, fontweight='medium')
            plt.ylabel('y', color='white', fontsize=14, fontweight='medium')
            plt.title('Linear Programming Solution', color='white', 
                     fontsize=16, pad=20, fontweight='bold')

            legend = plt.legend(loc='upper left', 
                              frameon=True,
                              facecolor='#1F2937', 
                              edgecolor='#4B5563',
                              fontsize=12)
            plt.setp(legend.get_texts(), color='white')

            for spine in ax.spines.values():
                spine.set_color('#4B5563')

            ax.tick_params(colors='white', labelsize=12)

            # Set plot limits
            max_val = max(np.max(b), np.max(res.x)) * 1.2
            plt.xlim(0, max_val)
            plt.ylim(0, max_val)
            plt.tight_layout()

            # Convert plot to base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', 
                       bbox_inches='tight', 
                       dpi=300,
                       facecolor='#111827', 
                       edgecolor='none')
            plt.close()
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Calculate optimal value
            optimal_value = float(-res.fun if data['optimization_type'] == 'maximize' else res.fun)

            # Return solution
            return JsonResponse({
                'solution': res.x.tolist(),
                'optimal_value': optimal_value,
                'image': image_base64
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)