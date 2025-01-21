from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
import numpy as np
from scipy.optimize import linprog
import json
import re
import string

@ensure_csrf_cookie
def simplex_solver_view(request):
    return render(request, 'simplex.html')

def get_variable_mapping(num_vars):
    """Create a mapping of alphabetical variables to indices."""
    variables = list(string.ascii_lowercase[:num_vars])
    return {var: idx for idx, var in enumerate(variables)}

def parse_expression(expr, num_vars):
    # Remove whitespace
    expr = expr.replace(' ', '')
    
    # Get variable mapping
    var_mapping = get_variable_mapping(num_vars)
    
    # Initialize coefficients array with zeros
    coefficients = [0] * num_vars
    
    # Find all terms that match coefficient and variable patterns
    # Modified pattern to match single alphabetical variables
    terms = re.findall(r'[+-]?[\d.]*[a-z]|[+-]?[\d.]+', expr)
    
    for term in terms:
        if any(c.isalpha() for c in term):
            # Extract variable (last character)
            var = term[-1]
            if var not in var_mapping:
                raise ValueError(f"Variable {var} exceeds the specified number of variables ({num_vars})")
            
            # Extract coefficient
            coeff_str = term[:-1]
            coeff = 1 if coeff_str in ['+', ''] else -1 if coeff_str == '-' else float(coeff_str)
            coefficients[var_mapping[var]] = coeff
            
    return coefficients

def parse_constraint(expr, num_vars):
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
        raise ValueError("Invalid constraint format. Must be in the form: aa + bb + ... <= c")

    left_side = parts[0].strip()
    try:
        right_side = float(parts[1].strip())
    except ValueError:
        raise ValueError("Right-hand side must be a number")

    coefficients = parse_expression(left_side, num_vars)
    coefficients = [sign_multiplier * coeff for coeff in coefficients]
    right_side *= sign_multiplier

    return coefficients + [right_side]

@ensure_csrf_cookie
def solve_simplex_lp(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from request body
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid JSON data'}, status=400)

            # Validate required fields
            required_fields = ['optimization_type', 'variables', 'objective', 'constraints']
            if not all(key in data for key in required_fields):
                return JsonResponse({'error': 'Missing required fields'}, status=400)

            try:
                num_vars = int(data['variables'])
                if num_vars < 1:
                    raise ValueError("Number of variables must be positive")
                if num_vars > 26:
                    raise ValueError("Maximum 26 variables (a-z) supported")
            except ValueError as e:
                return JsonResponse({'error': str(e)}, status=400)

            # Parse objective function and constraints
            try:
                objective = parse_expression(data['objective'], num_vars)
                constraints = [parse_constraint(c, num_vars) 
                             for c in data['constraints'].split(';') 
                             if c.strip()]
                if not constraints:
                    raise ValueError("No valid constraints provided")
            except ValueError as e:
                return JsonResponse({'error': str(e)}, status=400)

            constraints_matrix = np.array(constraints)
            
            # Set up optimization problem
            c = np.array(objective)
            if data['optimization_type'] == 'maximize':
                c = -c

            A = constraints_matrix[:, :-1]  # All columns except last
            b = constraints_matrix[:, -1]   # Last column
            bounds = [(0, None)] * num_vars  # Non-negative constraints for all variables

            # Solve LP problem using the simplex method
            try:
                res = linprog(
                    c=c, 
                    A_ub=A, 
                    b_ub=b, 
                    bounds=bounds, 
                    method='highs-ds',  # Dual simplex method
                    options={'disp': True}  # Enable iteration details
                )
                
                if not res.success:
                    return JsonResponse({'error': 'Optimization failed: ' + res.message}, status=400)
                
            except Exception as e:
                return JsonResponse({'error': f'Optimization error: {str(e)}'}, status=400)

            # Calculate optimal value
            optimal_value = float(-res.fun if data['optimization_type'] == 'maximize' else res.fun)

            # Create solution with variable names
            var_names = list(string.ascii_lowercase[:num_vars])
            solution = {var: float(val) for var, val in zip(var_names, res.x)}

            # Return solution
            return JsonResponse({
                'solution': solution,
                'optimal_value': optimal_value,
                'iterations': res.nit  # Number of iterations
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)