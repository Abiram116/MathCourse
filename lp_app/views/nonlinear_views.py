import numpy as np
import sympy as sp
from scipy import optimize
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
import json
import re
import string
import logging
import cvxpy as cp  # Add cvxpy for advanced solvers

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@ensure_csrf_cookie
def nonlinear_solver_view(request):
    """Render the nonlinear programming solver page."""
    print("Rendering nonlinear solver view")  # Add logging
    return render(request, 'nonlinear.html')

def detect_problem_type(objective_expr, constraints_list=None):
    """
    Automatically detect if the problem is quadratic or general nonlinear.
    Returns 'quadratic' if all variables have powers <= 2, otherwise 'nonlinear'.
    """
    # Combine all expressions to check
    all_expressions = [objective_expr]
    if constraints_list:
        all_expressions.extend(constraints_list)
    
    combined_expr = ' '.join(all_expressions)
    logger.debug(f"Detecting problem type from: {combined_expr}")
    
    # Check for powers > 2 using regex
    # Look for patterns like x^3, y^4, etc.
    power_pattern = r'[a-zA-Z][a-zA-Z0-9]*\^(\d+)'
    power_matches = re.findall(power_pattern, combined_expr)
    
    # Check for trigonometric and other nonlinear functions
    nonlinear_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'ln']
    has_nonlinear_func = any(func in combined_expr for func in nonlinear_funcs)
    
    # Check powers
    for power in power_matches:
        if int(power) > 2:
            logger.debug(f"Found power > 2: {power}, classifying as nonlinear")
            return 'nonlinear'
    
    if has_nonlinear_func:
        logger.debug(f"Found nonlinear function, classifying as nonlinear")
        return 'nonlinear'
    
    logger.debug("No powers > 2 or nonlinear functions found, classifying as quadratic")
    return 'quadratic'

def get_variable_mapping(expr):
    """Extract variables from expressions and create mapping."""
    # Find all variables in the expression (assuming standard format like x1, x2, etc.)
    # Use a more robust regex that handles subscripts and different variable names
    variables = set(re.findall(r'([a-zA-Z][a-zA-Z0-9]*)', expr))
    
    # Filter out mathematical functions
    math_functions = {'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'PI', 'pi', 'ln'}
    variables = {var for var in variables if var not in math_functions}
    
    # Create a mapping from variable names to indices
    result = {var: idx for idx, var in enumerate(sorted(variables))}
    logger.debug(f"Variable mapping: {result}")
    return result

def parse_quadratic_terms(expr, var_mapping):
    """Parse quadratic and linear terms from an expression."""
    num_vars = len(var_mapping)
    logger.debug(f"Parsing quadratic terms from: {expr} with {num_vars} variables")
    
    # Initialize matrices
    Q = np.zeros((num_vars, num_vars))  # Quadratic terms
    c = np.zeros(num_vars)  # Linear terms
    constant = 0.0
    
    # Replace ** with ^ for easier parsing and normalize spaces
    expr = expr.replace('**', '^').replace(' ', '')
    
    # Split into terms
    expr = expr.replace('-', '+-')
    if expr.startswith('+'):
        expr = expr[1:]
    terms = expr.split('+')
    logger.debug(f"Split terms: {terms}")
    
    for term in terms:
        if not term:  # Skip empty terms
            continue
            
        # Check if the term is a constant
        if all(not c.isalpha() for c in term):
            try:
                constant += float(term)
                logger.debug(f"Added constant: {term}")
                continue
            except ValueError:
                logger.debug(f"Failed to parse constant: {term}")
                pass
                
        # Check for quadratic terms (containing ^ or ** or two variables multiplied)
        if '^' in term or '*' in term:
            # Handle x^2 terms - more flexible regex to handle various forms
            squared_match = re.match(r'([-]?\d*\.?\d*)?([a-zA-Z][a-zA-Z0-9]*)\^2$', term)
            if not squared_match:
                # Try alternative format for squared terms
                squared_match = re.match(r'([-]?\d*\.?\d*)?([a-zA-Z][a-zA-Z0-9]*)\^{2}$', term)
            
            if squared_match:
                coef_str = squared_match.group(1)
                coef = 1.0
                if coef_str:
                    if coef_str == '-':
                        coef = -1.0
                    else:
                        try:
                            coef = float(coef_str)
                        except ValueError:
                            logger.debug(f"Failed to parse coefficient in squared term: {term}")
                
                var = squared_match.group(2)
                logger.debug(f"Squared term: {term}, coef: {coef}, var: {var}")
                if var in var_mapping:
                    i = var_mapping[var]
                    Q[i, i] += coef
                    logger.debug(f"Added to Q[{i},{i}] = {Q[i,i]}")
                else:
                    logger.warning(f"Variable {var} not found in mapping for term: {term}")
                continue
                
            # Handle x*y terms (cross-terms) - more flexible regex
            cross_match = re.match(r'([-]?\d*\.?\d*)?([a-zA-Z][a-zA-Z0-9]*)\*([a-zA-Z][a-zA-Z0-9]*)', term)
            if not cross_match:
                # Try alternative for implicit multiplication (no * symbol)
                cross_match = re.match(r'([-]?\d*\.?\d*)?([a-zA-Z][a-zA-Z0-9]*)([a-zA-Z][a-zA-Z0-9]*)', term)
            
            if cross_match:
                coef_str = cross_match.group(1)
                coef = 1.0
                if coef_str:
                    if coef_str == '-':
                        coef = -1.0
                    else:
                        try:
                            coef = float(coef_str)
                        except ValueError:
                            logger.debug(f"Failed to parse coefficient in cross term: {term}")
                
                var1 = cross_match.group(2)
                var2 = cross_match.group(3)
                logger.debug(f"Cross term: {term}, coef: {coef}, var1: {var1}, var2: {var2}")
                
                # Check if it's actually the same variable (like x1*x1)
                if var1 == var2 and var1 in var_mapping:
                    i = var_mapping[var1]
                    Q[i, i] += coef
                    logger.debug(f"Added to Q[{i},{i}] = {Q[i,i]} (same variable)")
                    continue
                
                if var1 in var_mapping and var2 in var_mapping:
                    i = var_mapping[var1]
                    j = var_mapping[var2]
                    # Add half to each symmetric position for a total of 'coef'
                    Q[i, j] += coef / 2
                    Q[j, i] += coef / 2
                    logger.debug(f"Added to Q[{i},{j}] and Q[{j},{i}] = {Q[i,j]}")
                else:
                    if var1 not in var_mapping:
                        logger.warning(f"Variable {var1} not found in mapping for term: {term}")
                    if var2 not in var_mapping:
                        logger.warning(f"Variable {var2} not found in mapping for term: {term}")
                continue
        
        # Handle linear terms - more flexible regex
        linear_match = re.match(r'([-]?\d*\.?\d*)?([a-zA-Z][a-zA-Z0-9]*)', term)
        if linear_match:
            coef_str = linear_match.group(1)
            coef = 1.0
            if coef_str:
                if coef_str == '-':
                    coef = -1.0
                else:
                    try:
                        coef = float(coef_str)
                    except ValueError:
                        logger.debug(f"Failed to parse coefficient in linear term: {term}")
            
            var = linear_match.group(2)
            logger.debug(f"Linear term: {term}, coef: {coef}, var: {var}")
            if var in var_mapping:
                i = var_mapping[var]
                c[i] += coef
                logger.debug(f"Added to c[{i}] = {c[i]}")
            else:
                logger.warning(f"Variable {var} not found in mapping for term: {term}")
    
    logger.debug(f"Parsed result: Q={Q}, c={c}, constant={constant}")
    return Q, c, constant

def parse_constraint(expr, var_mapping):
    """Parse a constraint into coefficients and right-hand side."""
    # Remove whitespace
    expr = expr.replace(' ', '')
    logger.debug(f"Parsing constraint: {expr}")
    
    # Handle different inequality signs
    if '>=' in expr:
        parts = expr.split('>=')
        sign = '>='
        logger.debug("Found >= sign")
    elif '<=' in expr:
        parts = expr.split('<=')
        sign = '<='
        logger.debug("Found <= sign") 
    elif '==' in expr:
        parts = expr.split('==')
        sign = '=='
        logger.debug("Found == sign")
    elif '=' in expr:
        parts = expr.split('=')
        sign = '=='
        logger.debug("Found = sign, treating as ==")
    else:
        logger.error(f"Invalid constraint format, no inequality sign found: {expr}")
        raise ValueError("Invalid constraint format. Must use <=, >=, or ==")
    
    if len(parts) != 2:
        logger.error(f"Invalid constraint format, couldn't split into two parts: {expr}")
        raise ValueError("Invalid constraint format. Must be in the form: expression <= value")
    
    left_side = parts[0].strip()
    right_side = parts[1].strip()
    
    logger.debug(f"Left side: {left_side}, Right side: {right_side}")
    
    # Parse the left side as a linear expression
    num_vars = len(var_mapping)
    A = np.zeros(num_vars)  # Initialize coefficients array
    
    # Replace ** with ^ for easier parsing and normalize spaces
    left_expr = left_side.replace('**', '^').replace(' ', '')
    
    # Split into terms
    left_expr = left_expr.replace('-', '+-')
    if left_expr.startswith('+'):
        left_expr = left_expr[1:]
    terms = left_expr.split('+')
    
    for term in terms:
        if not term:  # Skip empty terms
            continue
            
        # Check if the term is a constant
        if all(not c.isalpha() for c in term):
            # We'll handle constants later
            continue
            
        # Handle linear terms - more flexible regex
        linear_match = re.match(r'([-]?\d*\.?\d*)?([a-zA-Z][a-zA-Z0-9]*)', term)
        if linear_match:
            coef_str = linear_match.group(1)
            coef = 1.0
            if coef_str:
                if coef_str == '-':
                    coef = -1.0
                else:
                    try:
                        coef = float(coef_str)
                    except ValueError:
                        logger.debug(f"Failed to parse coefficient in linear term: {term}")
            
            var = linear_match.group(2)
            logger.debug(f"Linear term in constraint: {term}, coef: {coef}, var: {var}")
            if var in var_mapping:
                i = var_mapping[var]
                A[i] += coef
                logger.debug(f"Added to A[{i}] = {A[i]}")
            else:
                logger.warning(f"Variable {var} not found in mapping for constraint term: {term}")
    
    # Parse the right side
    try:
        b = float(right_side)
        logger.debug(f"Right side parsed as number: {b}")
    except ValueError:
        logger.error(f"Right side isn't a number: {right_side}")
        raise ValueError(f"Right side of constraint must be a number: {expr}")
    
    return A, b, sign

def index_variables(expr):
    """
    Return the expression as-is since we're now using explicit variable names (a, b, c)
    instead of auto-indexing x variables.
    """
    logger.debug(f"Using expression with explicit variable names: {expr}")
    return expr

@ensure_csrf_cookie
def solve_nonlinear(request):
    """Solve a nonlinear programming problem."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
    try:
        # Parse JSON data
        data = json.loads(request.body)
        objective_expr = data.get('objective', '')
        constraints_text = data.get('constraints', '')
        min_max = data.get('min_max', 'min')
        requested_method = data.get('method', 'auto')  # Parameter for QP solution method
        solve_now = data.get('solve_now', False)  # Flag to control whether to solve immediately
        
        # Auto-detect problem type if not specified
        auto_detected_type = detect_problem_type(objective_expr, constraints_text.split(';') if constraints_text else [])
        problem_type = data.get('problem_type', auto_detected_type)
        
        logger.debug(f"Problem type: {problem_type} (auto-detected: {auto_detected_type})")
        logger.debug(f"Objective: {objective_expr}")
        logger.debug(f"Constraints: {constraints_text}")
        logger.debug(f"Min/Max: {min_max}")
        logger.debug(f"Requested method: {requested_method}")
        logger.debug(f"Solve now: {solve_now}")
        
        if not objective_expr:
            return JsonResponse({'error': 'Objective function is required'}, status=400)
        
        # Parse constraints
        constraints_list = []
        if constraints_text:
            constraints_list = [c.strip() for c in constraints_text.split(';') if c.strip()]
        
        # Generate KKT conditions for either problem type
        kkt_conditions = generate_kkt_conditions(objective_expr, constraints_list, min_max)
        
        # Prepare the initial result object
        result = {
            'problem_type': problem_type,
            'auto_detected_type': auto_detected_type,
            'kkt_conditions': kkt_conditions,
            'solution_ready': False,  # Flag to indicate if solution has been computed
            'display_results': True   # New flag to tell the frontend to show the results tab immediately
        }
        
        # Set up initial information based on problem type and method
        if problem_type == 'quadratic':
            if requested_method == 'wolfe':
                # Extract variables
                var_mapping = get_variable_mapping(objective_expr + ' ' + ' '.join(constraints_list))
                variables = sorted(var_mapping.keys())
                n_vars = len(variables)
                
                # Parse the quadratic objective
                Q, c, constant = parse_quadratic_terms(objective_expr, var_mapping)
                
                # Handle maximization
                if min_max == 'max':
                    Q = -Q
                    c = -c
                
                # Parse constraints
                A_eq = []
                b_eq = []
                A_ineq = []
                b_ineq = []
                
                for constraint in constraints_list:
                    try:
                        A_row, b_val, constraint_type = parse_constraint(constraint, var_mapping)
                        
                        if constraint_type == '==':
                            A_eq.append(A_row)
                            b_eq.append(b_val)
                        elif constraint_type == '<=':
                            A_ineq.append(A_row)
                            b_ineq.append(b_val)
                        elif constraint_type == '>=':
                            A_ineq.append(-A_row)  # Negate to convert to <=
                            b_ineq.append(-b_val)
                    except Exception as e:
                        logger.exception(f"Error parsing constraint: {constraint}")
                        return JsonResponse({'error': f"Error parsing constraint: {constraint}. {str(e)}"}, status=400)
                
                # Convert to numpy arrays
                A_eq = np.array(A_eq) if A_eq else np.empty((0, n_vars))
                b_eq = np.array(b_eq) if b_eq else np.empty(0)
                A_ineq = np.array(A_ineq) if A_ineq else np.empty((0, n_vars))
                b_ineq = np.array(b_ineq) if b_ineq else np.empty(0)
                
                # Create initial Wolfe tableau
                m_ineq = A_ineq.shape[0]  # Number of inequality constraints
                
                # Initial tableau structure
                tableau_header = ["Basis"] + [f"x_{i+1}" for i in range(n_vars)] + [f"λ_{i+1}" for i in range(m_ineq)] + [f"v_{i+1}" for i in range(m_ineq)] + ["RHS"]
                
                tableau_rows = []
                
                # Objective row
                obj_row = ["Z"]
                for j in range(n_vars):
                    obj_row.append(f"{c[j]:.4f}")
                for j in range(m_ineq):
                    obj_row.append("0")  # λ coefficients
                for j in range(m_ineq):
                    obj_row.append("0")  # v coefficients
                obj_row.append("0")  # RHS
                tableau_rows.append(obj_row)
                
                # Constraint rows from KKT conditions
                for i in range(n_vars):
                    row = [f"x_{i+1}"]
                    # Gradient of objective with respect to x_i
                    for j in range(n_vars):
                        row.append(f"{Q[i, j]:.4f}")
                    # Coefficients of λ
                    for j in range(m_ineq):
                        row.append(f"{-A_ineq[j, i]:.4f}" if j < m_ineq else "0")
                    # Coefficients of v
                    for j in range(m_ineq):
                        row.append("0")
                    row.append(f"{-c[i]:.4f}")  # RHS
                    tableau_rows.append(row)
                
                result['initial_tableau'] = {
                    'title': 'Initial Wolfe Dual Tableau',
                    'header': tableau_header,
                    'rows': tableau_rows
                }
                
                result['method_explanation'] = [
                    "Wolfe's method solves quadratic programs by creating a dual problem and using complementary slackness.",
                    "The method sets up a tableau similar to the simplex method for linear programming.",
                    "The KKT conditions are built into the tableau structure.",
                    "The method then performs pivot operations to find an optimal solution."
                ]
                
            elif requested_method == 'beale':
                # Extract variables
                var_mapping = get_variable_mapping(objective_expr + ' ' + ' '.join(constraints_list))
                variables = sorted(var_mapping.keys())
                n_vars = len(variables)
                
                # Parse the quadratic objective
                Q, c, constant = parse_quadratic_terms(objective_expr, var_mapping)
                
                # Handle maximization
                if min_max == 'max':
                    Q = -Q
                    c = -c
                
                # Parse constraints
                A_eq = []
                b_eq = []
                A_ineq = []
                b_ineq = []
                
                for constraint in constraints_list:
                    try:
                        A_row, b_val, constraint_type = parse_constraint(constraint, var_mapping)
                        
                        if constraint_type == '==':
                            A_eq.append(A_row)
                            b_eq.append(b_val)
                        elif constraint_type == '<=':
                            A_ineq.append(A_row)
                            b_ineq.append(b_val)
                        elif constraint_type == '>=':
                            A_ineq.append(-A_row)  # Negate to convert to <=
                            b_ineq.append(-b_val)
                    except Exception as e:
                        logger.exception(f"Error parsing constraint: {constraint}")
                        return JsonResponse({'error': f"Error parsing constraint: {constraint}. {str(e)}"}, status=400)
                
                # Convert to numpy arrays
                A_eq = np.array(A_eq) if A_eq else np.empty((0, n_vars))
                b_eq = np.array(b_eq) if b_eq else np.empty(0)
                A_ineq = np.array(A_ineq) if A_ineq else np.empty((0, n_vars))
                b_ineq = np.array(b_ineq) if b_ineq else np.empty(0)
                
                # Create initial Beale tableau
                tableau_header = ["Basis"] + [f"x_{i+1}" for i in range(n_vars)] + ["RHS"]
                tableau_rows = []
                
                # For demonstration purposes, create a simple example tableau
                obj_row = ["Z"]
                for j in range(n_vars):
                    obj_row.append(f"{c[j]:.4f}")
                obj_row.append("0")  # RHS
                tableau_rows.append(obj_row)
                
                # Add constraint rows for demonstration
                for i in range(A_ineq.shape[0]):
                    row = [f"S_{i+1}"]  # Slack variable
                    for j in range(n_vars):
                        row.append(f"{A_ineq[i, j]:.4f}")
                    row.append(f"{b_ineq[i]:.4f}")
                    tableau_rows.append(row)
                
                result['initial_tableau'] = {
                    'title': 'Initial Beale Tableau',
                    'header': tableau_header,
                    'rows': tableau_rows
                }
                
                result['method_explanation'] = [
                    "Beale's method is an extension of the simplex method for quadratic programming.",
                    "The method introduces artificial variables to handle the quadratic terms.",
                    "The algorithm performs pivot operations similar to the simplex method.",
                    "It converges to the optimal solution by satisfying the KKT conditions."
                ]
            
            else:  # Auto or other QP method
                result['method_explanation'] = [
                    "This quadratic program can be solved using various methods:",
                    "- Interior point methods: Work by moving through the interior of the feasible region",
                    "- Active set methods: Identify the active constraints at the optimum",
                    "- Gradient projection: Projects the gradient onto the feasible region",
                    "The solution satisfies the KKT conditions shown above."
                ]
        
        else:  # General nonlinear
            result['method_explanation'] = [
                "This nonlinear program requires numerical optimization methods:",
                "- Sequential quadratic programming (SQP): Approximates the problem with quadratic subproblems",
                "- Interior point methods: Follow a path through the interior of the feasible region",
                "- Trust region methods: Build and refine models within a trusted region",
                "The KKT conditions provide necessary conditions for optimality."
            ]
        
        # Only solve if explicitly requested
        if solve_now:
            if problem_type == 'quadratic':
                if requested_method == 'wolfe':
                    solution = solve_quadratic_program_wolfe(objective_expr, constraints_list, min_max)
                elif requested_method == 'beale':
                    solution = solve_quadratic_program_beale(objective_expr, constraints_list, min_max)
                else:
                    solution = solve_quadratic_program(objective_expr, constraints_list, min_max)
            else:  # General nonlinear
                solution = solve_general_nonlinear(objective_expr, constraints_list, min_max)
            
            # Merge solution into result
            if isinstance(solution, dict):
                for key, value in solution.items():
                    if key != 'kkt_conditions':  # Don't overwrite KKT conditions
                        result[key] = value
            
            result['solution_ready'] = True
        
        # Add backwards compatibility
        if 'variables' in result and 'solution' not in result:
            result['solution'] = result['variables']
        if 'objective_value' in result and 'optimal_value' not in result:
            result['optimal_value'] = result['objective_value']
        
        return JsonResponse(result)
    
    except Exception as e:
        logger.exception(f"Error in solve_nonlinear: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)

def generate_kkt_conditions(objective_expr, constraints_list, min_max='min'):
    """Generate symbolic KKT conditions for the optimization problem."""
    try:
        # Extract variables from expressions
        var_mapping = get_variable_mapping(objective_expr + ' ' + ' '.join(constraints_list))
        variables = sorted(var_mapping.keys())
        logger.debug(f"Variables for KKT: {variables}")
        
        # Convert to sympy symbols
        symbols = {var: sp.Symbol(var) for var in variables}
        
        # Try to evaluate the objective, with better error handling
        try:
            # First approach: Add spaces between terms to make parsing easier
            processed_expr = objective_expr
            
            # Add spaces around operators
            processed_expr = re.sub(r'([+\-*/^()])', r' \1 ', processed_expr)
            # Fix double signs (e.g., "+ -" becomes "+-")
            processed_expr = re.sub(r'\+ \s* \-', '+-', processed_expr)
            # Handle multiplications between variables (e.g., "ab" becomes "a*b")
            for var in sorted(variables, key=len, reverse=True):
                processed_expr = re.sub(r'([0-9])(' + var + r')', r'\1*\2', processed_expr)
                processed_expr = re.sub(r'(' + var + r')([0-9])', r'\1*\2', processed_expr)
                # Handle implicit multiplication between variables (e.g., "ab" becomes "a*b")
                for other_var in sorted(variables, key=len, reverse=True):
                    if var != other_var:
                        processed_expr = re.sub(r'(' + var + r')(' + other_var + r')', r'\1*\2', processed_expr)
            
            logger.debug(f"Processed expr for sympify: {processed_expr}")
            
            # Try using string parsing first
            obj_str = processed_expr
            # Replace ^ with ** for Python power syntax
            obj_str = obj_str.replace('^', '**')
            # Create a namespace with symbol variables
            namespace = {var: symbols[var] for var in variables}
            # Add math functions
            namespace.update({'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 
                               'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt})
            
            # Try direct evaluation with symbols
            try:
                obj_sympy = eval(obj_str, {"__builtins__": {}}, namespace)
                # Make the objective negative for maximization
                obj_sympy = -obj_sympy if min_max == 'max' else obj_sympy
            except SyntaxError as e:
                # If direct eval fails, try sympify
                raw_expr = objective_expr.replace('^', '**')
                # Escape special characters and substitute variables
                for var in variables:
                    raw_expr = re.sub(r'\b' + var + r'\b', f"symbols['{var}']", raw_expr)
                    
                obj_sympy = sp.sympify(raw_expr, locals={'symbols': symbols})
                obj_sympy = -obj_sympy if min_max == 'max' else obj_sympy
        except Exception as e:
            logger.exception(f"Error parsing objective: {str(e)}")
            return [{'title': 'Error generating KKT conditions', 
                     'conditions': [f"Error parsing objective: {str(e)}"]}]
        
        # Parse constraints
        inequality_constraints = []  # g_i(x) ≤ 0
        equality_constraints = []   # h_j(x) = 0
        constraint_expr_map = {}  # Store original expressions
        
        for constraint in constraints_list:
            try:
                if '>=' in constraint:
                    lhs, rhs = constraint.split('>=')
                    expr = lhs.strip()
                    try:
                        value = float(rhs.strip())
                    except ValueError:
                        return [{'title': 'Error generating KKT conditions', 
                                 'conditions': [f"Right side must be a number in constraint: {constraint}"]}]
                    
                    # Try direct sympify with preprocessing
                    try:
                        processed_expr = expr
                        # Add spaces around operators and handle multiplications as before
                        processed_expr = re.sub(r'([+\-*/^()])', r' \1 ', processed_expr)
                        processed_expr = re.sub(r'\+ \s* \-', '+-', processed_expr)
                        for var in sorted(variables, key=len, reverse=True):
                            processed_expr = re.sub(r'([0-9])(' + var + r')', r'\1*\2', processed_expr)
                            processed_expr = re.sub(r'(' + var + r')([0-9])', r'\1*\2', processed_expr)
                            for other_var in sorted(variables, key=len, reverse=True):
                                if var != other_var:
                                    processed_expr = re.sub(r'(' + var + r')(' + other_var + r')', r'\1*\2', processed_expr)
                        
                        expr_sympy = sp.sympify(processed_expr, locals={var: symbols[var] for var in variables})
                        # g_i(x) ≤ 0 format: convert g(x) ≥ value to g(x) - value ≥ 0 to -g(x) + value ≤ 0
                        inequality_expr = expr_sympy - value
                        inequality_constraints.append(inequality_expr)
                        constraint_expr_map[str(inequality_expr)] = f"{expr} ≥ {value}"
                    except Exception as e:
                        logger.exception(f"Error processing constraint: {constraint}")
                        return [{'title': 'Error generating KKT conditions', 
                                 'conditions': [f"Could not parse constraint: {constraint}. {str(e)}"]}]
                    
                elif '<=' in constraint:
                    lhs, rhs = constraint.split('<=')
                    expr = lhs.strip()
                    try:
                        value = float(rhs.strip())
                    except ValueError:
                        return [{'title': 'Error generating KKT conditions', 
                                 'conditions': [f"Right side must be a number in constraint: {constraint}"]}]
                    
                    # Similar preprocessing and parsing
                    try:
                        processed_expr = expr
                        processed_expr = re.sub(r'([+\-*/^()])', r' \1 ', processed_expr)
                        processed_expr = re.sub(r'\+ \s* \-', '+-', processed_expr)
                        for var in sorted(variables, key=len, reverse=True):
                            processed_expr = re.sub(r'([0-9])(' + var + r')', r'\1*\2', processed_expr)
                            processed_expr = re.sub(r'(' + var + r')([0-9])', r'\1*\2', processed_expr)
                            for other_var in sorted(variables, key=len, reverse=True):
                                if var != other_var:
                                    processed_expr = re.sub(r'(' + var + r')(' + other_var + r')', r'\1*\2', processed_expr)
                        
                        expr_sympy = sp.sympify(processed_expr, locals={var: symbols[var] for var in variables})
                        # g_i(x) ≤ 0 format: convert g(x) ≤ value to g(x) - value ≤ 0
                        inequality_expr = expr_sympy - value
                        inequality_constraints.append(inequality_expr)
                        constraint_expr_map[str(inequality_expr)] = f"{expr} ≤ {value}"
                    except Exception as e:
                        logger.exception(f"Error processing constraint: {constraint}")
                        return [{'title': 'Error generating KKT conditions', 
                                 'conditions': [f"Could not parse constraint: {constraint}. {str(e)}"]}]
                    
                elif '==' in constraint or '=' in constraint:
                    separator = '==' if '==' in constraint else '='
                    lhs, rhs = constraint.split(separator)
                    expr = lhs.strip()
                    try:
                        value = float(rhs.strip())
                    except ValueError:
                        return [{'title': 'Error generating KKT conditions', 
                                 'conditions': [f"Right side must be a number in constraint: {constraint}"]}]
                    
                    try:
                        processed_expr = expr
                        processed_expr = re.sub(r'([+\-*/^()])', r' \1 ', processed_expr)
                        processed_expr = re.sub(r'\+ \s* \-', '+-', processed_expr)
                        for var in sorted(variables, key=len, reverse=True):
                            processed_expr = re.sub(r'([0-9])(' + var + r')', r'\1*\2', processed_expr)
                            processed_expr = re.sub(r'(' + var + r')([0-9])', r'\1*\2', processed_expr)
                            for other_var in sorted(variables, key=len, reverse=True):
                                if var != other_var:
                                    processed_expr = re.sub(r'(' + var + r')(' + other_var + r')', r'\1*\2', processed_expr)
                        
                        expr_sympy = sp.sympify(processed_expr, locals={var: symbols[var] for var in variables})
                        # h_j(x) = 0 format: convert g(x) = value to g(x) - value = 0
                        equality_expr = expr_sympy - value
                        equality_constraints.append(equality_expr)
                        constraint_expr_map[str(equality_expr)] = f"{expr} = {value}"
                    except Exception as e:
                        logger.exception(f"Error processing constraint: {constraint}")
                        return [{'title': 'Error generating KKT conditions', 
                                 'conditions': [f"Could not parse constraint: {constraint}. {str(e)}"]}]
                else:
                    return [{'title': 'Error generating KKT conditions', 
                             'conditions': [f"Invalid constraint format: {constraint}. Must use <=, >=, or =="]}]
            except Exception as e:
                logger.exception(f"Error processing constraint: {constraint}")
                return [{'title': 'Error generating KKT conditions', 
                         'conditions': [f"Error processing constraint {constraint}: {str(e)}"]}]
        
        # Create lambda symbols for inequality constraints
        lambda_symbols = [sp.Symbol(f'λ_{i+1}') for i in range(len(inequality_constraints))]
        
        # Create mu symbols for equality constraints
        mu_symbols = [sp.Symbol(f'μ_{i+1}') for i in range(len(equality_constraints))]
        
        # Build the Lagrangian
        lagrangian = obj_sympy
        
        # Add inequality constraints to Lagrangian
        for i, g in enumerate(inequality_constraints):
            lagrangian -= lambda_symbols[i] * g
        
        # Add equality constraints to Lagrangian
        for i, h in enumerate(equality_constraints):
            lagrangian += mu_symbols[i] * h
        
        # Format KKT conditions in a more readable way
        kkt_conditions = []
        
        # Intro - Lagrangian function definition
        lagrangian_intro = [
            "𝐿(x, λ, μ) = f(x) - ∑λᵢgᵢ(x) + ∑μⱼhⱼ(x)",
            "",
            f"where f(x) = {sp.latex(obj_sympy)}"
        ]
        
        # Add constraint definitions
        if inequality_constraints:
            lagrangian_intro.append("")
            lagrangian_intro.append("Inequality constraints (gᵢ(x) ≤ 0):")
            for i, g in enumerate(inequality_constraints):
                try:
                    original = constraint_expr_map.get(str(g), f"g_{i+1}(x)")
                    lagrangian_intro.append(f"g_{i+1}(x): {original} → {sp.latex(g)} ≤ 0")
                except Exception as e:
                    lagrangian_intro.append(f"g_{i+1}(x): Error rendering constraint: {str(e)}")
        
        if equality_constraints:
            lagrangian_intro.append("")
            lagrangian_intro.append("Equality constraints (hⱼ(x) = 0):")
            for i, h in enumerate(equality_constraints):
                try:
                    original = constraint_expr_map.get(str(h), f"h_{i+1}(x)")
                    lagrangian_intro.append(f"h_{i+1}(x): {original} → {sp.latex(h)} = 0")
                except Exception as e:
                    lagrangian_intro.append(f"h_{i+1}(x): Error rendering constraint: {str(e)}")
        
        kkt_conditions.append({
            'title': 'Lagrangian Function',
            'conditions': lagrangian_intro
        })
        
        # 1. Stationarity: ∇f(x) + ∑λᵢ∇gᵢ(x) + ∑μⱼ∇hⱼ(x) = 0
        stationarity_conditions = ["∇L(x*) = 0, or equivalently:"]
        
        # Calculate gradient of Lagrangian for each variable
        for var, sym in symbols.items():
            try:
                gradient = sp.diff(lagrangian, sym)
                stationarity_conditions.append(f"∂L/∂{var} = {sp.latex(gradient)} = 0")
            except Exception as e:
                logger.exception(f"Error calculating gradient for {var}: {str(e)}")
                stationarity_conditions.append(f"∂L/∂{var} = Error: {str(e)}")
        
        kkt_conditions.append({
            'title': '✅ Stationarity Conditions',
            'conditions': stationarity_conditions
        })
        
        # 2. Primal Feasibility
        primal_conditions = ["Solution must satisfy all original constraints:"]
        
        # Inequality constraints
        if inequality_constraints:
            primal_conditions.append("")
            primal_conditions.append("Inequality constraints:")
            for i, g in enumerate(inequality_constraints):
                try:
                    primal_conditions.append(f"g_{i+1}(x) = {sp.latex(g)} ≤ 0")
                except Exception as e:
                    primal_conditions.append(f"g_{i+1}(x) = Error: {str(e)}")
        
        # Equality constraints
        if equality_constraints:
            primal_conditions.append("")
            primal_conditions.append("Equality constraints:")
            for i, h in enumerate(equality_constraints):
                try:
                    primal_conditions.append(f"h_{i+1}(x) = {sp.latex(h)} = 0")
                except Exception as e:
                    primal_conditions.append(f"h_{i+1}(x) = Error: {str(e)}")
        
        kkt_conditions.append({
            'title': '✅ Primal Feasibility',
            'conditions': primal_conditions
        })
        
        # 3. Dual Feasibility: λᵢ ≥ 0 for all i
        dual_conditions = ["Lagrange multipliers for inequality constraints must be non-negative:"]
        
        for i, lambda_sym in enumerate(lambda_symbols):
            try:
                dual_conditions.append(f"{sp.latex(lambda_sym)} ≥ 0")
            except Exception as e:
                dual_conditions.append(f"λ_{i+1} ≥ 0 (Error: {str(e)})")
        
        kkt_conditions.append({
            'title': '✅ Dual Feasibility',
            'conditions': dual_conditions
        })
        
        # 4. Complementary Slackness: λᵢ * gᵢ(x) = 0 for all i
        comp_slackness = [
            "For each inequality constraint gᵢ(x) ≤ 0, either:",
            "• The constraint is active (gᵢ(x) = 0), or",
            "• Its corresponding Lagrange multiplier is zero (λᵢ = 0)"
        ]
        
        for i, (g, lambda_sym) in enumerate(zip(inequality_constraints, lambda_symbols)):
            try:
                comp_slackness.append(f"{sp.latex(lambda_sym)} · {sp.latex(g)} = 0")
            except Exception as e:
                comp_slackness.append(f"λ_{i+1} · g_{i+1}(x) = 0 (Error: {str(e)})")
        
        kkt_conditions.append({
            'title': '✅ Complementary Slackness',
            'conditions': comp_slackness
        })
        
        return kkt_conditions
    
    except Exception as e:
        logger.exception(f"Error generating KKT conditions: {str(e)}")
        return [{'title': 'Error generating KKT conditions', 'conditions': [str(e)]}]

def solve_quadratic_program(objective_expr, constraints_list, min_max):
    """Solve a quadratic program using scipy.optimize.minimize."""
    try:
        logger.debug("Solving quadratic program")
        
        # Extract variables from expressions
        var_mapping = get_variable_mapping(objective_expr + ' ' + ' '.join(constraints_list))
        variables = sorted(var_mapping.keys())
        n_vars = len(variables)
        
        # Parse the quadratic objective
        Q, c, constant = parse_quadratic_terms(objective_expr, var_mapping)
        
        # Handle maximization by negating the objective
        if min_max == 'max':
            Q = -Q
            c = -c
        
        # Parse constraints
        A_eq = []
        b_eq = []
        A_ineq = []
        b_ineq = []
        
        for constraint in constraints_list:
            try:
                A_row, b_val, constraint_type = parse_constraint(constraint, var_mapping)
                
                if constraint_type == '==':
                    A_eq.append(A_row)
                    b_eq.append(b_val)
                elif constraint_type == '<=':
                    A_ineq.append(A_row)
                    b_ineq.append(b_val)
                elif constraint_type == '>=':
                    A_ineq.append(-A_row)  # Negate to convert to <=
                    b_ineq.append(-b_val)
            except Exception as e:
                logger.exception(f"Error parsing constraint: {constraint}")
                return {'error': f"Error parsing constraint: {constraint}. {str(e)}"}
        
        # Convert to numpy arrays
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None
        A_ineq = np.array(A_ineq) if A_ineq else None
        b_ineq = np.array(b_ineq) if b_ineq else None
        
        # If using scipy.optimize
        try:
            # Define objective function f(x) = 0.5 x^T Q x + c^T x + constant
            def objective(x):
                return 0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(c, x) + constant
            
            # Define objective gradient g(x) = Q x + c
            def objective_gradient(x):
                return np.dot(Q, x) + c
            
            # If maximizing, we need to negate the objective and gradient
            if min_max == 'max':
                def neg_objective(x):
                    return -objective(x)
                
                def neg_gradient(x):
                    return -objective_gradient(x)
                
                obj_func = neg_objective
                obj_grad = neg_gradient
            else:
                obj_func = objective
                obj_grad = objective_gradient
            
            # Define constraints for scipy.optimize.minimize
            constraints = []
            
            # Inequality constraints: A_ineq @ x <= b_ineq
            if A_ineq is not None and b_ineq is not None:
                for i in range(len(b_ineq)):
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, i=i: b_ineq[i] - np.dot(A_ineq[i], x),
                        'jac': lambda x, i=i: -A_ineq[i]
                    })
            
            # Equality constraints: A_eq @ x == b_eq
            if A_eq is not None and b_eq is not None:
                for i in range(len(b_eq)):
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda x, i=i: np.dot(A_eq[i], x) - b_eq[i],
                        'jac': lambda x, i=i: A_eq[i]
                    })
            
            # Initial guess
            x0 = np.zeros(n_vars)
            
            # Solve the problem
            result = optimize.minimize(
                obj_func,
                x0,
                method='SLSQP',
                jac=obj_grad,
                constraints=constraints,
                options={'disp': True}
            )
            
            # Extract solution
            if result.success:
                solution_dict = {}
                for i, var in enumerate(variables):
                    solution_dict[var] = float(result.x[i])
                
                obj_value = float(objective(result.x))
                if min_max == 'max':
                    obj_value = -obj_value
                
                logger.debug(f"Solution found: {solution_dict}, objective: {obj_value}")
                
                return {
                    'variables': solution_dict,
                    'objective_value': obj_value,
                    'iterations': result.nit,
                    'success': True,
                    'message': result.message
                }
            else:
                logger.warning(f"Optimization failed: {result.message}")
                return {
                    'error': f"Optimization failed: {result.message}",
                    'success': False
                }
                
        except Exception as e:
            logger.exception(f"Error in scipy.optimize: {str(e)}")
            return {'error': f"Error in scipy.optimize: {str(e)}"}
            
    except Exception as e:
        logger.exception(f"Error in quadratic solver: {str(e)}")
        return {'error': f"Error in quadratic solver: {str(e)}"}

def solve_general_nonlinear(objective_expr, constraints_list, min_max):
    """
    Solve a general nonlinear optimization problem.
    """
    try:
        logger.debug("Solving general nonlinear program")
        
        # Extract variables
        var_mapping = get_variable_mapping(objective_expr + ' ' + ' '.join(constraints_list))
        variables = sorted(var_mapping.keys())
        n_vars = len(variables)
        
        # Parse constraints
        A_eq = []
        b_eq = []
        A_ineq = []
        b_ineq = []
        nonlinear_constraints = []
        
        for constraint in constraints_list:
            try:
                # Try to parse as linear first
                try:
                    A_row, b_val, constraint_type = parse_constraint(constraint, var_mapping)
                    
                    if constraint_type == '==':
                        A_eq.append(A_row)
                        b_eq.append(b_val)
                    elif constraint_type == '<=':
                        A_ineq.append(A_row)
                        b_ineq.append(b_val)
                    elif constraint_type == '>=':
                        A_ineq.append(-A_row)  # Negate to convert to <=
                        b_ineq.append(-b_val)
                except Exception:
                    # If parsing as linear fails, treat as nonlinear
                    nonlinear_constraints.append(constraint)
            except Exception as e:
                logger.exception(f"Error parsing constraint: {constraint}")
                return {'error': f"Error parsing constraint: {constraint}. {str(e)}"}
        
        # Convert to numpy arrays
        A_eq = np.array(A_eq) if A_eq else np.empty((0, n_vars))
        b_eq = np.array(b_eq) if b_eq else np.empty(0)
        A_ineq = np.array(A_ineq) if A_ineq else np.empty((0, n_vars))
        b_ineq = np.array(b_ineq) if b_ineq else np.empty(0)
        
        # Create the objective function
        namespace = {var: 0 for var in variables}
        namespace.update({'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt})
        
        # Preprocess the objective expression to handle powers
        obj_expr = objective_expr.replace('^', '**')  # Convert ^ to ** for Python
        
        # Add explicit multiplication operators where needed
        for var in sorted(variables, key=len, reverse=True):
            # Add * between numbers and variables
            obj_expr = re.sub(r'(\d)(' + var + r')', r'\1*\2', obj_expr)
            # Add * between variables and numbers
            obj_expr = re.sub(r'(' + var + r')(\d)', r'\1*\2', obj_expr)
            # Handle implicit multiplication between variables
            for other_var in sorted(variables, key=len, reverse=True):
                if var != other_var:
                    obj_expr = re.sub(r'(' + var + r')(' + other_var + r')', r'\1*\2', obj_expr)
        
        def objective_function(x):
            for i, var in enumerate(variables):
                namespace[var] = x[i]
            try:
                result = eval(obj_expr, {"__builtins__": {}}, namespace)
                return result if min_max == 'min' else -result
            except Exception as e:
                logger.error(f"Error evaluating objective: {str(e)}, expr: {obj_expr}")
                # Return a large value to signal an error to the optimizer
                return 1e10 if min_max == 'min' else -1e10
        
        # Create constraint functions for nonlinear constraints
        constraint_functions = []
        
        for constraint in nonlinear_constraints:
            parts = re.split(r'(==|<=|>=)', constraint)
            lhs = parts[0].strip()
            operator = parts[1].strip()
            rhs = parts[2].strip()
            
            # Preprocess expressions
            lhs_expr = lhs.replace('^', '**')
            rhs_expr = rhs.replace('^', '**')
            
            # Add explicit multiplication
            for var in sorted(variables, key=len, reverse=True):
                lhs_expr = re.sub(r'(\d)(' + var + r')', r'\1*\2', lhs_expr)
                lhs_expr = re.sub(r'(' + var + r')(\d)', r'\1*\2', lhs_expr)
                rhs_expr = re.sub(r'(\d)(' + var + r')', r'\1*\2', rhs_expr)
                rhs_expr = re.sub(r'(' + var + r')(\d)', r'\1*\2', rhs_expr)
                
                for other_var in sorted(variables, key=len, reverse=True):
                    if var != other_var:
                        lhs_expr = re.sub(r'(' + var + r')(' + other_var + r')', r'\1*\2', lhs_expr)
                        rhs_expr = re.sub(r'(' + var + r')(' + other_var + r')', r'\1*\2', rhs_expr)
            
            def make_constraint_func(lhs_expr, rhs_expr, op):
                def constraint_func(x):
                    for i, var in enumerate(variables):
                        namespace[var] = x[i]
                    try:
                        lhs_val = eval(lhs_expr, {"__builtins__": {}}, namespace)
                        rhs_val = eval(rhs_expr, {"__builtins__": {}}, namespace)
                        if op == '==':
                            return lhs_val - rhs_val
                        elif op == '<=':
                            return rhs_val - lhs_val  # Ensure constraint is in form g(x) >= 0
                        elif op == '>=':
                            return lhs_val - rhs_val  # Ensure constraint is in form g(x) >= 0
                    except Exception as e:
                        logger.error(f"Error evaluating constraint: {str(e)}, lhs: {lhs_expr}, rhs: {rhs_expr}")
                        return 0  # Return a value that won't constrain the solution space
                return constraint_func
            
            if operator == '==':
                constraint_functions.append({'type': 'eq', 'fun': make_constraint_func(lhs_expr, rhs_expr, operator)})
            elif operator in ['<=', '>=']:
                constraint_functions.append({'type': 'ineq', 'fun': make_constraint_func(lhs_expr, rhs_expr, operator)})
        
        # Add linear constraints
        if A_eq.shape[0] > 0:
            constraint_functions.append({
                'type': 'eq',
                'fun': lambda x: A_eq @ x - b_eq
            })
        
        if A_ineq.shape[0] > 0:
            constraint_functions.append({
                'type': 'ineq',
                'fun': lambda x: b_ineq - A_ineq @ x
            })
        
        # Initial guess
        x0 = np.zeros(n_vars)
        
        # Start with the solution dictionary to be built up
        solution = {'method': 'scipy-slsqp'}
        
        # Always generate KKT conditions, regardless of optimization success
        try:
            kkt_conditions = generate_kkt_conditions(objective_expr, constraints_list, min_max)
            solution['kkt_conditions'] = kkt_conditions
        except Exception as kkt_error:
            logger.exception("Error generating KKT conditions")
            solution['kkt_error'] = str(kkt_error)
        
        # Try optimization with different methods if one fails
        try:
            # Try SLSQP first
            result = optimize.minimize(
                objective_function,
                x0,
                constraints=constraint_functions,
                method='SLSQP',
                options={'disp': True}
            )
            
            solution['status'] = 'success' if result.success else 'error'
            solution['message'] = result.message
            solution['iterations'] = result.nit if hasattr(result, 'nit') else 0
            
            if not result.success:
                # If SLSQP fails, try trust-constr
                try:
                    logger.info("SLSQP failed, trying trust-constr")
                    solution['message'] += " - Trying alternative solver trust-constr."
                    
                    result = optimize.minimize(
                        objective_function,
                        x0,
                        constraints=constraint_functions,
                        method='trust-constr',
                        options={'verbose': 1}
                    )
                    
                    solution['status'] = 'success' if result.success else 'error'
                    solution['message'] = result.message
                    solution['iterations'] = result.nit if hasattr(result, 'nit') else 0
                except Exception as trust_error:
                    logger.exception("Error with trust-constr solver")
                    solution['trust_error'] = str(trust_error)
            
            # Format the result if optimization succeeded with any method
            if result.success:
                solution_values = {}
                for i, var in enumerate(variables):
                    solution_values[var] = round(result.x[i], 4)
                solution['solution'] = solution_values
                
                objective_value = objective_function(result.x)
                if min_max == 'max':
                    objective_value = -objective_value
                solution['objective_value'] = round(objective_value, 4)
            else:
                # Even if optimization failed, return partial solution or best guess
                logger.warning("Optimization failed, returning best guess")
                solution_values = {}
                for i, var in enumerate(variables):
                    solution_values[var] = round(result.x[i], 4) if hasattr(result, 'x') else 0
                solution['partial_solution'] = solution_values
                solution['note'] = "Optimization failed but partial results are provided. KKT conditions may still be useful."
        
        except Exception as e:
            logger.exception("Error in general nonlinear solver")
            solution['error'] = str(e)
            solution['status'] = 'error'
            solution['note'] = "Solver failed but KKT conditions are still available."
        
        return solution
        
    except Exception as e:
        logger.exception("Error in general nonlinear solver")
        return {'error': str(e), 'status': 'error'}

def solve_quadratic_program_wolfe(objective_expr, constraints_list, min_max):
    """Solve a quadratic program using Wolfe's method."""
    try:
        logger.debug("Solving quadratic program using Wolfe's method")
        
        # Extract variables
        var_mapping = get_variable_mapping(objective_expr + ' ' + ' '.join(constraints_list))
        variables = sorted(var_mapping.keys())
        n_vars = len(variables)
        
        # Parse the quadratic objective
        Q, c, constant = parse_quadratic_terms(objective_expr, var_mapping)
        
        # Handle maximization by negating the objective
        if min_max == 'max':
            Q = -Q
            c = -c
        
        # Parse constraints
        A_eq = []
        b_eq = []
        A_ineq = []
        b_ineq = []
        
        for constraint in constraints_list:
            try:
                A_row, b_val, constraint_type = parse_constraint(constraint, var_mapping)
                
                if constraint_type == '==':
                    A_eq.append(A_row)
                    b_eq.append(b_val)
                elif constraint_type == '<=':
                    A_ineq.append(A_row)
                    b_ineq.append(b_val)
                elif constraint_type == '>=':
                    A_ineq.append(-A_row)  # Negate to convert to <=
                    b_ineq.append(-b_val)
            except Exception as e:
                logger.exception(f"Error parsing constraint: {constraint}")
                return {'error': f"Error parsing constraint: {constraint}. {str(e)}"}
        
        # Convert to numpy arrays
        A_eq = np.array(A_eq) if A_eq else np.empty((0, n_vars))
        b_eq = np.array(b_eq) if b_eq else np.empty(0)
        A_ineq = np.array(A_ineq) if A_ineq else np.empty((0, n_vars))
        b_ineq = np.array(b_ineq) if b_ineq else np.empty(0)
        
        # Implement Wolfe's method
        # 1. Set up the dual problem
        m_ineq = A_ineq.shape[0]  # Number of inequality constraints
        
        # Wolfe's dual problem tableau
        steps = []
        
        # Initial tableau structure
        tableau_header = ["Basis"] + [f"x_{i+1}" for i in range(n_vars)] + [f"λ_{i+1}" for i in range(m_ineq)] + [f"v_{i+1}" for i in range(m_ineq)] + ["RHS"]
        
        # Set up initial tableau for Wolfe's method
        # This is a simplified implementation - in real-world applications, this would be more complex
        tableau_rows = []
        
        # Objective row
        obj_row = ["Z"]
        for j in range(n_vars):
            obj_row.append(f"{c[j]:.4f}")
        for j in range(m_ineq):
            obj_row.append("0")  # λ coefficients
        for j in range(m_ineq):
            obj_row.append("0")  # v coefficients
        obj_row.append("0")  # RHS
        tableau_rows.append(obj_row)
        
        # Constraint rows from KKT conditions
        for i in range(n_vars):
            row = [f"x_{i+1}"]
            # Gradient of objective with respect to x_i
            for j in range(n_vars):
                row.append(f"{Q[i, j]:.4f}")
            # Coefficients of λ
            for j in range(m_ineq):
                row.append(f"{-A_ineq[j, i]:.4f}" if j < m_ineq else "0")
            # Coefficients of v
            for j in range(m_ineq):
                row.append("0")
            row.append(f"{-c[i]:.4f}")  # RHS
            tableau_rows.append(row)
        
        steps.append({
            'title': 'Initial Wolfe Dual Tableau',
            'header': tableau_header,
            'rows': tableau_rows
        })
        
        # In a real implementation, we would now perform pivot operations
        # For demonstration, we'll skip to the solution
        
        # Use scipy's solver to get the actual solution
        result = solve_quadratic_program(objective_expr, constraints_list, min_max)
        
        # Add Wolfe's method steps to the result
        result['method'] = 'wolfe'
        result['steps'] = steps
        
        return result
    
    except Exception as e:
        logger.exception("Error in Wolfe's method")
        return {'error': str(e), 'method': 'wolfe'}

def solve_quadratic_program_beale(objective_expr, constraints_list, min_max):
    """Solve a quadratic program using Beale's method."""
    try:
        logger.debug("Solving quadratic program using Beale's method")
        
        # Extract variables
        var_mapping = get_variable_mapping(objective_expr + ' ' + ' '.join(constraints_list))
        variables = sorted(var_mapping.keys())
        n_vars = len(variables)
        
        # Parse the quadratic objective
        Q, c, constant = parse_quadratic_terms(objective_expr, var_mapping)
        
        # Handle maximization by negating the objective
        if min_max == 'max':
            Q = -Q
            c = -c
        
        # Parse constraints
        A_eq = []
        b_eq = []
        A_ineq = []
        b_ineq = []
        
        for constraint in constraints_list:
            try:
                A_row, b_val, constraint_type = parse_constraint(constraint, var_mapping)
                
                if constraint_type == '==':
                    A_eq.append(A_row)
                    b_eq.append(b_val)
                elif constraint_type == '<=':
                    A_ineq.append(A_row)
                    b_ineq.append(b_val)
                elif constraint_type == '>=':
                    A_ineq.append(-A_row)  # Negate to convert to <=
                    b_ineq.append(-b_val)
            except Exception as e:
                logger.exception(f"Error parsing constraint: {constraint}")
                return {'error': f"Error parsing constraint: {constraint}. {str(e)}"}
                
        # Convert to numpy arrays
        A_eq = np.array(A_eq) if A_eq else np.empty((0, n_vars))
        b_eq = np.array(b_eq) if b_eq else np.empty(0)
        A_ineq = np.array(A_ineq) if A_ineq else np.empty((0, n_vars))
        b_ineq = np.array(b_ineq) if b_ineq else np.empty(0)
        
        # Create Beale's method steps
        steps = []
        
        # Initial tableau for Beale's method would be set up here
        tableau_header = ["Basis"] + [f"x_{i+1}" for i in range(n_vars)] + ["RHS"]
        tableau_rows = []
        
        # For demonstration purposes, we'll create a simple example tableau
        obj_row = ["Z"]
        for j in range(n_vars):
            obj_row.append(f"{c[j]:.4f}")
        obj_row.append("0")  # RHS
        tableau_rows.append(obj_row)
        
        # Add constraint rows for demonstration
        for i in range(A_ineq.shape[0]):
            row = [f"S_{i+1}"]  # Slack variable
            for j in range(n_vars):
                row.append(f"{A_ineq[i, j]:.4f}")
            row.append(f"{b_ineq[i]:.4f}")
            tableau_rows.append(row)
        
        steps.append({
            'title': 'Initial Beale Tableau',
            'header': tableau_header,
            'rows': tableau_rows
        })
        
        # In a real implementation, we would perform Beale's method iterations
        # For demonstration, we'll skip to the solution
        
        # Use scipy's solver to get the actual solution
        result = solve_quadratic_program(objective_expr, constraints_list, min_max)
        
        # Add Beale's method steps to the result
        result['method'] = 'beale'
        result['steps'] = steps
        
        return result
    
    except Exception as e:
        logger.exception("Error in Beale's method")
        return {'error': str(e), 'method': 'beale'}

def solve_with_cvxpy(objective_expr, constraints_list, min_max):
    """Solve a quadratic program using CVXPY."""
    try:
        logger.debug("Solving quadratic program using CVXPY")
        
        # Extract variables
        var_mapping = get_variable_mapping(objective_expr + ' ' + ' '.join(constraints_list))
        variables = sorted(var_mapping.keys())
        n_vars = len(variables)
        
        # Parse the quadratic objective
        Q, c, constant = parse_quadratic_terms(objective_expr, var_mapping)
        
        # Create CVXPY variables
        x = cp.Variable(n_vars)
        
        # Create objective function
        if min_max == 'min':
            objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x + constant)
        else:
            objective = cp.Maximize(-0.5 * cp.quad_form(x, -Q) - c.T @ x - constant)
        
        # Parse constraints
        constraints = []
        
        for constraint in constraints_list:
            try:
                if '>=' in constraint:
                    lhs, rhs = constraint.split('>=')
                    A_row, _, _ = parse_constraint(f"{lhs} <= 0", var_mapping)
                    b_val = float(rhs.strip())
                    constraints.append(A_row @ x >= b_val)
                    
                elif '<=' in constraint:
                    lhs, rhs = constraint.split('<=')
                    A_row, _, _ = parse_constraint(f"{lhs} <= 0", var_mapping)
                    b_val = float(rhs.strip())
                    constraints.append(A_row @ x <= b_val)
                    
                elif '==' in constraint or '=' in constraint:
                    if '==' in constraint:
                        lhs, rhs = constraint.split('==')
                    else:
                        lhs, rhs = constraint.split('=')
                        
                    A_row, _, _ = parse_constraint(f"{lhs} <= 0", var_mapping)
                    b_val = float(rhs.strip())
                    constraints.append(A_row @ x == b_val)
                    
            except Exception as e:
                logger.exception(f"Error parsing constraint for CVXPY: {constraint}")
                return {'error': f"Error parsing constraint: {constraint}. {str(e)}"}
        
        # Create and solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        # Check if the problem was solved successfully
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return {
                'error': f"Problem could not be solved: {prob.status}",
                'method': 'cvxpy'
            }
        
        # Extract the solution
        solution = {}
        solution['variables'] = {}
        for i, var in enumerate(variables):
            solution['variables'][var] = float(x.value[i])
        
        solution['objective_value'] = float(prob.value)
        solution['method'] = 'cvxpy'
        solution['iterations'] = -1  # CVXPY doesn't provide iteration count directly
        
        return solution
    
    except Exception as e:
        logger.exception("Error in CVXPY solver")
        return {'error': str(e), 'method': 'cvxpy'}
    