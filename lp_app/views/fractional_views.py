from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
import numpy as np
from scipy.optimize import linprog
import json
import re
import string

@ensure_csrf_cookie
def fractional_solver_view(request):
    """Render the fractional programming solver page."""
    return render(request, 'fractional.html')

def get_variable_mapping(num_vars):
    """Create a mapping of alphabetical variables to indices.
    Always include x and y variables regardless of num_vars value.
    """
    # Make sure x and y are always available (most common variables)
    variables = ['x', 'y']
    
    # Add more variables if needed
    if num_vars > 2:
        # Skip x and y which are already in our list
        additional_vars = [v for v in string.ascii_lowercase[2:num_vars] 
                          if v not in variables]
        variables.extend(additional_vars)
    
    # Only keep the first num_vars elements
    variables = variables[:num_vars]
    
    print(f"Variable mapping for {num_vars} variables: {variables}")
    return {var: idx for idx, var in enumerate(variables)}

def parse_expression(expr, num_vars):
    """Parse a mathematical expression into coefficients for variables."""
    print(f"Parsing expression: '{expr}'")
    
    # Remove whitespace and handle LaTeX escapes if present
    expr = expr.replace(' ', '')
    expr = expr.replace('\\', '')  # Remove any remaining backslashes from LaTeX
    print(f"After removing whitespace and LaTeX escapes: '{expr}'")
    
    # Get variable mapping
    var_mapping = get_variable_mapping(num_vars)
    print(f"Variable mapping: {var_mapping}")
    
    # Initialize coefficients array with zeros
    coefficients = [0] * num_vars
    
    # Find all terms that match coefficient and variable patterns
    # Modified pattern to match single alphabetical variables
    terms = re.findall(r'[+-]?[\d.]*[a-z]|[+-]?[\d.]+', expr)
    print(f"Found terms: {terms}")
    
    # Constant term (initialized to 0)
    constant = 0
    
    for term in terms:
        print(f"Processing term: '{term}'")
        if any(c.isalpha() for c in term):
            # Extract variable (last character)
            var = term[-1]
            print(f"Found variable: {var}")
            if var not in var_mapping:
                # Try to handle the case when variables are swapped
                if var in ['x', 'y'] and num_vars >= 2:
                    print(f"Variable {var} is standard (x,y) but not in mapping. Using fixed mapping x=0, y=1")
                    var_idx = 0 if var == 'x' else 1
                else:
                    raise ValueError(f"Variable {var} exceeds the specified number of variables ({num_vars})")
            else:
                var_idx = var_mapping[var]
            
            # Extract coefficient
            coeff_str = term[:-1]
            coeff = 1 if coeff_str in ['+', ''] else -1 if coeff_str == '-' else float(coeff_str)
            print(f"Coefficient string: '{coeff_str}', coefficient: {coeff}")
            coefficients[var_idx] = coeff
        else:
            # This is a constant term
            constant = float(term)
            print(f"Constant term: {constant}")
            
    print(f"Final coefficients: {coefficients}, constant: {constant}")
    return coefficients, constant

def parse_constraint(expr, num_vars):
    """Parse a constraint into coefficients and right-hand side."""
    print(f"Parsing constraint: '{expr}'")
    
    # Remove any remaining LaTeX escapes
    expr = expr.replace('\\', '')
    
    # Handle different inequality signs
    sign_multiplier = None
    parts = None
    
    if '>=' in expr:
        print("Found >= operator")
        expr = expr.replace('>=', '<=')
        sign_multiplier = -1
        parts = expr.split('<=')
    elif '<=' in expr:
        print("Found <= operator")
        sign_multiplier = 1
        parts = expr.split('<=')
    else:
        # Try with Unicode symbols
        if '≥' in expr:
            print("Found ≥ operator")
            expr = expr.replace('≥', '<=')
            sign_multiplier = -1
            parts = expr.split('<=')
        elif '≤' in expr:
            print("Found ≤ operator")
            sign_multiplier = 1
            parts = expr.split('≤')
        elif 'le' in expr:  # Handle plain text 'le' that might come from \le
            print("Found 'le' text (from LaTeX)")
            expr = expr.replace('le', '<=')
            sign_multiplier = 1
            parts = expr.split('<=')
        elif 'ge' in expr:  # Handle plain text 'ge' that might come from \ge
            print("Found 'ge' text (from LaTeX)")
            expr = expr.replace('ge', '<=')
            sign_multiplier = -1
            parts = expr.split('<=')
        else:
            # Last resort - check for = sign and default to <=
            if '=' in expr:
                print("Found = operator, treating as <=")
                sign_multiplier = 1
                parts = expr.split('=')
            else:
                raise ValueError(f"Invalid constraint format. Must use <= or >=. Got: {expr}")
    
    print(f"Parts after split: {parts}")
    
    if not parts or len(parts) != 2:
        raise ValueError(f"Invalid constraint format. Must be in the form: aa + bb + ... <= c. Got: {expr}")
    
    left_side = parts[0].strip()
    print(f"Left side: '{left_side}'")
    
    try:
        right_side = float(parts[1].strip())
        print(f"Right side: {right_side}")
    except ValueError:
        raise ValueError(f"Right-hand side must be a number. Got: {parts[1].strip()}")
    
    coefficients, constant = parse_expression(left_side, num_vars)
    print(f"Parsed coefficients: {coefficients}, constant: {constant}")
    
    coefficients = [sign_multiplier * coeff for coeff in coefficients]
    right_side = sign_multiplier * (right_side - constant * sign_multiplier)
    
    print(f"Final coefficients: {coefficients}, right_side: {right_side}")
    
    return coefficients + [right_side]

@ensure_csrf_cookie
def solve_fractional(request):
    """Handle the POST request to solve a fractional programming problem."""
    if request.method == 'POST':
        try:
            # Parse JSON data from request body
            try:
                data = json.loads(request.body)
                print("Received data:", data)  # Debugging
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid JSON data'}, status=400)
            
            # Validate required fields
            required_fields = ['optimization_type', 'numerator', 'denominator', 'constraints', 'variables']
            if not all(key in data for key in required_fields):
                missing = [field for field in required_fields if field not in data]
                return JsonResponse({'error': f'Missing required fields: {missing}'}, status=400)
            
            try:
                num_vars = int(data['variables'])
                if num_vars < 1:
                    raise ValueError("Number of variables must be positive")
                if num_vars > 26:
                    raise ValueError("Maximum 26 variables (a-z) supported")
            except ValueError as e:
                return JsonResponse({'error': str(e)}, status=400)
            
            # Parse numerator, denominator, and constraints
            try:
                numerator_coeffs, numerator_const = parse_expression(data['numerator'], num_vars)
                denominator_coeffs, denominator_const = parse_expression(data['denominator'], num_vars)
                
                # Check that denominator is not zero or negative
                if all(coeff == 0 for coeff in denominator_coeffs) and denominator_const <= 0:
                    raise ValueError("Denominator cannot be zero or negative in the feasible region")
                
                # Debug constraint parsing
                constraint_str = data['constraints']
                print("Constraints received:", constraint_str)
                constraint_list = constraint_str.split(';')
                print("Constraint list after split:", constraint_list)
                constraint_list = [c.strip() for c in constraint_list if c.strip()]
                print("Constraint list after cleaning:", constraint_list)
                
                # Process each constraint
                constraints = []
                for c in constraint_list:
                    print(f"Processing constraint: '{c}'")
                    parsed = parse_constraint(c, num_vars)
                    print(f"Parsed result: {parsed}")
                    constraints.append(parsed)
                
                if not constraints:
                    raise ValueError("No valid constraints provided")
                
            except ValueError as e:
                print(f"Error occurred: {str(e)}")
                return JsonResponse({'error': str(e)}, status=400)
            
            # Apply Charnes-Cooper transformation to convert fractional program to linear program
            # For a problem max N(x)/D(x) subject to Ax <= b, x >= 0:
            # Let y = x/D(x) and t = 1/D(x)
            # This becomes max N(y) subject to D(y) = 1, Ay <= bt, t >= 0, y >= 0
            
            maximize = data['optimization_type'] == 'maximize'
            
            # Original constraints Ax <= b
            constraints_matrix = np.array(constraints)
            A = constraints_matrix[:, :-1]  # All columns except last
            b = constraints_matrix[:, -1]   # Last column
            
            # New variable count: original variables + t
            new_n = num_vars + 1
            
            # New objective (for t, y variables)
            c = np.zeros(new_n)
            for i in range(num_vars):
                c[i] = numerator_coeffs[i]
            c[new_n-1] = numerator_const
            
            if not maximize:
                c = -c
            
            # New constraints
            # 1. Original constraints: A*y - b*t <= 0
            A_new_1 = np.zeros((len(constraints), new_n))
            for i in range(len(constraints)):
                for j in range(num_vars):
                    A_new_1[i, j] = A[i, j]
                A_new_1[i, new_n-1] = -b[i]
            
            b_new_1 = np.zeros(len(constraints))
            
            # 2. Denominator constraint: denominator_coeffs * y + denominator_const * t = 1
            A_new_2 = np.zeros((1, new_n))
            for i in range(num_vars):
                A_new_2[0, i] = denominator_coeffs[i]
            A_new_2[0, new_n-1] = denominator_const
            
            b_new_2 = np.array([1.0])
            
            # 3. t > 0 constraint (handled by bounds)
            
            # Combine all constraints
            A_eq = A_new_2
            b_eq = b_new_2
            
            A_ub = A_new_1
            b_ub = b_new_1
            
            # Non-negative constraints for all variables
            bounds = [(0, None)] * new_n
            
            # Solve LP problem using the simplex method
            try:
                res = linprog(
                    c=-c if maximize else c,  # Negate if maximizing
                    A_ub=A_ub, 
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=bounds, 
                    method='highs-ds',  # Dual simplex method
                    options={'disp': True}  # Enable iteration details
                )
                
                if not res.success:
                    return JsonResponse({'error': 'Optimization failed: ' + res.message}, status=400)
                
            except Exception as e:
                return JsonResponse({'error': f'Optimization error: {str(e)}'}, status=400)
            
            # Extract solution
            y = res.x[:-1]  # All elements except the last
            t = res.x[-1]   # Last element (t value)
            
            # Convert back to original variables: x = y/t
            x = y / t if t > 0 else y
            
            # Calculate optimal value
            if t > 0:
                # Original objective value: N(x)/D(x)
                numerator_value = sum(numerator_coeffs[i] * x[i] for i in range(num_vars)) + numerator_const
                denominator_value = sum(denominator_coeffs[i] * x[i] for i in range(num_vars)) + denominator_const
                optimal_value = numerator_value / denominator_value
            else:
                optimal_value = float('inf') if maximize else float('-inf')
            
            # Create solution with variable names
            var_names = list(string.ascii_lowercase[:num_vars])
            solution = {var: float(val) for var, val in zip(var_names, x)}
            
            # Return solution
            return JsonResponse({
                'solution': solution,
                'optimal_value': optimal_value,
                'iterations': res.nit  # Number of iterations
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)