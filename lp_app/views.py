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
    """
    Find vertices of the feasible region using a more robust approach
    that works for all types of linear programming problems.
    """
    vertices = []
    num_constraints = len(b)
    
    # Calculate bounds based on constraint coefficients and constants
    coeff_max = max(abs(A.max()), abs(A.min()))
    b_max = max(abs(b.max()), abs(b.min()))
    max_bound = max(b_max / coeff_max if coeff_max != 0 else b_max, 100) * 1.5
    
    # Define bounding box vertices
    box_points = np.array([
        [0, 0],
        [max_bound, 0],
        [0, max_bound],
        [max_bound, max_bound]
    ])
    
    # Check each bounding box point
    for point in box_points:
        if all(np.dot(A[k], point) <= b[k] + 1e-10 for k in range(num_constraints)):
            vertices.append(point)
    
    # Find intersections of constraint lines
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            try:
                # Get the two lines we're intersecting
                A_sub = np.array([A[i], A[j]])
                b_sub = np.array([b[i], b[j]])
                
                # Check if lines are not parallel
                if abs(np.linalg.det(A_sub)) > 1e-10:
                    # Solve the intersection
                    intersection = np.linalg.solve(A_sub, b_sub)
                    
                    # Check if intersection point satisfies all constraints
                    if (intersection[0] >= -1e-10 and intersection[1] >= -1e-10 and
                        all(np.dot(A[k], intersection) <= b[k] + 1e-10 for k in range(num_constraints))):
                        vertices.append(intersection)
            except np.linalg.LinAlgError:
                continue
        
        # Find intersections with axes
        for axis in range(2):  # 0 for x-axis, 1 for y-axis
            if abs(A[i][axis]) > 1e-10:  # Avoid division by zero
                # Point where constraint line intersects with axis
                point = np.zeros(2)
                point[axis] = b[i] / A[i][axis]
                
                # Check if point satisfies all constraints
                if (point[axis] >= 0 and 
                    all(np.dot(A[k], point) <= b[k] + 1e-10 for k in range(num_constraints))):
                    vertices.append(point)

    # Convert to numpy array and clean up
    if len(vertices) > 0:
        vertices = np.array(vertices)
        # Remove duplicates with tolerance
        vertices = np.unique(np.round(vertices, decimals=10), axis=0)
        # Remove points with negative coordinates (allowing small numerical errors)
        vertices = vertices[np.all(vertices >= -1e-10, axis=1)]
        
        if len(vertices) >= 3:
            # Sort vertices counterclockwise around centroid for proper polygon
            centroid = vertices.mean(axis=0)
            angles = np.arctan2(vertices[:, 1] - centroid[1], 
                              vertices[:, 0] - centroid[0])
            vertices = vertices[np.argsort(angles)]
    
    return vertices if len(vertices) > 0 else np.array([[0, 0]])

def update_plot_bounds(ax, vertices, A, b):
    """
    Set appropriate plot bounds that show the full feasible region
    """
    if len(vertices) > 0:
        # Get the range of vertices
        min_x, min_y = vertices.min(axis=0)
        max_x, max_y = vertices.max(axis=0)
        
        # Calculate ranges
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        # Add padding (20% of range or minimum of 2 units)
        padding_x = max(x_range * 0.2, 2)
        padding_y = max(y_range * 0.2, 2)
        
        # Set limits with padding
        plt.xlim(max(0, min_x - padding_x), max_x + padding_x)
        plt.ylim(max(0, min_y - padding_y), max_y + padding_y)
    else:
        # Fallback: use constraint coefficients to set bounds
        coeff_max = max(abs(A.max()), abs(A.min()))
        b_max = max(abs(b.max()), abs(b.min()))
        max_val = max(b_max / coeff_max if coeff_max != 0 else b_max, 10)
        plt.xlim(0, max_val * 1.2)
        plt.ylim(0, max_val * 1.2)

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
                feasible_polygon = Polygon(vertices[hull.vertices], closed=True, alpha=0.3, 
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