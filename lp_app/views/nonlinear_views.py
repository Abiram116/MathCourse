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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@ensure_csrf_cookie
def nonlinear_solver_view(request):
    """Render the nonlinear programming solver page."""
    return render(request, 'nonlinear.html')

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
    
    # For nonlinear expressions, return as is
    return {
        'expression': left_side,
        'value': right_side,
        'type': sign
    }

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
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            logger.debug(f"Received data: {data}")
            
            # Validate required fields
            required_fields = ['problem_type', 'objective', 'constraints', 'min_max']
            if not all(key in data for key in required_fields):
                missing = [field for field in required_fields if field not in data]
                logger.error(f"Missing required fields: {missing}")
                return JsonResponse({
                    'success': False, 
                    'error': f'Missing required fields: {missing}'
                }, status=400)
            
            problem_type = data.get('problem_type', 'quadratic')
            min_max = data.get('min_max', 'min')
            objective = data.get('objective', '').strip()
            constraints = data.get('constraints', '')
            
            # Validate inputs
            if not objective:
                logger.error("Empty objective function")
                return JsonResponse({
                    'success': False,
                    'error': 'Objective function cannot be empty'
                }, status=400)
                
            if not constraints:
                logger.error("No constraints provided")
                return JsonResponse({
                    'success': False,
                    'error': 'At least one constraint must be provided'
                }, status=400)
            
            # Split constraints by semicolon
            constraints_list = constraints.split(';')
            constraints_list = [c.strip() for c in constraints_list if c.strip()]
                
            # Check for invalid characters or potential code injection
            for expr in [objective] + constraints_list:
                if any(c in expr for c in ['#', '`', '&', '$', '%', '@', '!']):
                    logger.error(f"Invalid character in expression: {expr}")
                    return JsonResponse({
                        'success': False,
                        'error': f'Invalid character in expression: {expr}'
                    }, status=400)
            
            # Use expressions as-is since we're now handling explicit variable names (a, b, c, etc.)
            # instead of auto-indexing x variables
            logger.info(f"Processing {problem_type} problem to {min_max}imize '{objective}' with constraints: {constraints_list}")
            
            # Dispatch to appropriate solver
            if problem_type == 'quadratic':
                return solve_quadratic_program(objective, constraints_list, min_max)
            else:
                return solve_general_nonlinear(objective, constraints_list, min_max)
                
        except json.JSONDecodeError:
            logger.exception("Invalid JSON in request body")
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON format in request body'
            }, status=400)
        except Exception as e:
            import traceback
            logger.exception("Error processing nonlinear solver request")
            return JsonResponse({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method. Use POST to submit optimization problems.'
    }, status=405)

def solve_quadratic_program(objective_expr, constraints_list, min_max):
    """Solve a quadratic programming problem."""
    try:
        logger.debug(f"Starting quadratic solve with objective: {objective_expr}, constraints: {constraints_list}, min_max: {min_max}")
        
        # Extract variables from objective and constraints
        all_expressions = [objective_expr] + constraints_list
        combined_expr = ' '.join(all_expressions)
        var_mapping = get_variable_mapping(combined_expr)
        
        if not var_mapping:
            logger.error("No variables detected in the expressions")
            return JsonResponse({
                'success': False,
                'error': 'No variables detected in the expressions'
            })
        
        # Parse the objective function
        Q, c, constant = parse_quadratic_terms(objective_expr, var_mapping)
        
        # Parse constraints
        constraint_objects = []
        for constraint_expr in constraints_list:
            constraint = parse_constraint(constraint_expr, var_mapping)
            
            # For quadratic programming, we only support linear constraints
            # Extract the left and right side
            left = constraint['expression']
            right = constraint['value']
            
            # Try to convert right side to a number
            try:
                b = float(right)
                logger.debug(f"Right side parsed as number: {b}")
            except ValueError:
                logger.error(f"Right side isn't a number: {right}")
                return JsonResponse({
                    'success': False,
                    'error': f'Right side of constraint must be a number: {constraint_expr}'
                })
            
            # Extract coefficients for each variable in the constraint
            A = np.zeros(len(var_mapping))
            terms = left.replace('-', '+-').split('+')
            for term in terms:
                if not term:
                    continue
                
                # Match term pattern: coefficient*variable or just variable
                match = re.match(r'([-]?\d*\.?\d*)?([a-zA-Z][a-zA-Z0-9]*)', term)
                if match:
                    coef = 1.0 if not match.group(1) or match.group(1) == '-' else float(match.group(1))
                    if match.group(1) == '-':
                        coef = -1.0
                    var = match.group(2)
                    logger.debug(f"Constraint term: {term}, coef: {coef}, var: {var}")
                    if var in var_mapping:
                        i = var_mapping[var]
                        A[i] = coef
                        logger.debug(f"Setting A[{i}] = {coef}")
            
            # Create constraint based on type
            if constraint['type'] == '<=':
                constraint_objects.append(optimize.LinearConstraint(A, -np.inf, b))
                logger.debug(f"Added <= constraint: {A} <= {b}")
            elif constraint['type'] == '>=':
                constraint_objects.append(optimize.LinearConstraint(A, b, np.inf))
                logger.debug(f"Added >= constraint: {A} >= {b}")
            elif constraint['type'] == '==' or constraint['type'] == '=':
                constraint_objects.append(optimize.LinearConstraint(A, b, b))
                logger.debug(f"Added == constraint: {A} == {b}")
        
        # Set default bounds to non-negative
        bounds = optimize.Bounds(0, np.inf)
        logger.debug("Set bounds to non-negative")
        
        # Define the objective function
        def objective(x):
            val = 0.5 * np.dot(x, np.dot(Q, x)) + np.dot(c, x) + constant
            logger.debug(f"Objective function call with x={x}, result={val}")
            return val
            
        def objective_gradient(x):
            grad = np.dot(Q, x) + c
            logger.debug(f"Gradient function call with x={x}, result={grad}")
            return grad
        
        # Solve the quadratic programming problem
        if min_max == 'max':
            # For maximization, negate the objective function
            def neg_objective(x):
                val = -objective(x)
                logger.debug(f"Negated objective function call with x={x}, result={val}")
                return val
                
            def neg_gradient(x):
                grad = -objective_gradient(x)
                logger.debug(f"Negated gradient function call with x={x}, result={grad}")
                return grad
            
            logger.debug("Starting maximization solver")
            result = optimize.minimize(
                neg_objective,
                np.zeros(len(var_mapping)),
                method='SLSQP',
                jac=neg_gradient,
                bounds=bounds,
                constraints=constraint_objects,
                options={'disp': True, 'maxiter': 1000}
            )
            objective_value = -result.fun
        else:
            logger.debug("Starting minimization solver")
            result = optimize.minimize(
                objective,
                np.zeros(len(var_mapping)),
                method='SLSQP',
                jac=objective_gradient,
                bounds=bounds,
                constraints=constraint_objects,
                options={'disp': True, 'maxiter': 1000}
            )
            objective_value = result.fun
        
        logger.debug(f"Optimization result: {result}")
        logger.debug(f"Solution: {result.x}")
        logger.debug(f"Objective value: {objective_value}")
        
        if result.success:
            # Format solution with variable names
            solution = {}
            for var, idx in var_mapping.items():
                solution[var] = float(result.x[idx])
            
            # Format constraints for display
            formatted_constraints = []
            for i, constraint_expr in enumerate(constraints_list):
                formatted_constraints.append(constraint_expr)
            
            return JsonResponse({
                'success': True,
                'solution': solution,
                'objectiveValue': float(objective_value),
                'iterations': result.nit,
                'message': result.message,
                'problem': {
                    'objective': objective_expr,
                    'constraints': formatted_constraints,
                    'minMax': min_max,
                    'problemType': 'quadratic'
                }
            })
        else:
            logger.error(f"Optimization failed: {result.message}")
            return JsonResponse({
                'success': False,
                'error': f"Optimization failed: {result.message}"
            })
            
    except Exception as e:
        import traceback
        logger.exception("Exception in quadratic solver")
        return JsonResponse({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

def solve_general_nonlinear(objective_expr, constraints_list, min_max):
    """Solve a general nonlinear programming problem using the KKT conditions."""
    try:
        # Extract variables from objective and constraints
        all_expressions = [objective_expr] + constraints_list
        combined_expr = ' '.join(all_expressions)
        var_mapping = get_variable_mapping(combined_expr)
        
        if not var_mapping:
            return JsonResponse({
                'success': False,
                'error': 'No variables detected in the expressions'
            })
        
        # Create sympy symbols for variables
        variables = {}
        for var_name in var_mapping.keys():
            variables[var_name] = sp.Symbol(var_name)
        
        # Parse the objective function
        try:
            # Replace ** with ^ for easier parsing
            sympy_expr = objective_expr.replace('**', '^').replace('^', '**')
            
            # Make the expression more sympy-friendly
            sympy_expr = sympy_expr.replace('sin', 'sp.sin')
            sympy_expr = sympy_expr.replace('cos', 'sp.cos')
            sympy_expr = sympy_expr.replace('tan', 'sp.tan')
            sympy_expr = sympy_expr.replace('exp', 'sp.exp')
            sympy_expr = sympy_expr.replace('log', 'sp.log')
            sympy_expr = sympy_expr.replace('sqrt', 'sp.sqrt')
            
            # Replace variable names with sympy symbols
            for var_name, var_symbol in variables.items():
                # Use word boundaries to avoid partial matches
                sympy_expr = re.sub(r'\b' + var_name + r'\b', f'symbols["{var_name}"]', sympy_expr)
            
            # Create namespace for eval
            symbols = variables
            
            try:
                obj_expr = eval(sympy_expr)
            except Exception as e:
                logger.exception(f"Error evaluating expression: {sympy_expr}")
                return JsonResponse({
                    'success': False,
                    'error': f'Error parsing objective function: {str(e)}. Check for correct syntax.'
                })
                
        except Exception as e:
            logger.exception(f"Error parsing objective function: {objective_expr}")
            return JsonResponse({
                'success': False,
                'error': f'Error parsing objective function: {str(e)}'
            })
        
        # Parse constraints
        eq_constraints = []
        ineq_constraints = []
        
        for constraint_expr in constraints_list:
            try:
                constraint = parse_constraint(constraint_expr, var_mapping)
                
                # Parse left and right sides
                left_expr = constraint['expression'].replace('**', '^').replace('^', '**')
                right_expr = constraint['value'].replace('**', '^').replace('^', '**')
                
                # Make the expressions more sympy-friendly
                left_expr_sympy = left_expr
                right_expr_sympy = right_expr
                for func in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
                    left_expr_sympy = left_expr_sympy.replace(func, f'sp.{func}')
                    right_expr_sympy = right_expr_sympy.replace(func, f'sp.{func}')
                
                # Replace variable names with sympy symbols
                for var_name, var_symbol in variables.items():
                    left_expr_sympy = re.sub(r'\b' + var_name + r'\b', f'symbols["{var_name}"]', left_expr_sympy)
                    right_expr_sympy = re.sub(r'\b' + var_name + r'\b', f'symbols["{var_name}"]', right_expr_sympy)
                
                # Evaluate expressions
                try:
                    left_side = eval(left_expr_sympy)
                except Exception as e:
                    logger.exception(f"Error evaluating left side: {left_expr}")
                    return JsonResponse({
                        'success': False,
                        'error': f'Error parsing left side of constraint {constraint_expr}: {str(e)}. Check for correct syntax.'
                    })
                
                # Try to convert right side to a number first
                try:
                    right_side = float(right_expr)
                except ValueError:
                    # If not a number, evaluate as expression
                    try:
                        right_side = eval(right_expr_sympy)
                    except Exception as e:
                        logger.exception(f"Error evaluating right side: {right_expr}")
                        return JsonResponse({
                            'success': False,
                            'error': f'Error parsing right side of constraint {constraint_expr}: {str(e)}. Check for correct syntax.'
                        })
                
                # Create constraint based on type
                if constraint['type'] == '<=':
                    ineq_constraints.append(left_side - right_side)
                elif constraint['type'] == '>=':
                    ineq_constraints.append(right_side - left_side)
                elif constraint['type'] == '==' or constraint['type'] == '=':
                    eq_constraints.append(left_side - right_side)
            except Exception as e:
                logger.exception(f"Error parsing constraint: {constraint_expr}")
                return JsonResponse({
                    'success': False,
                    'error': f'Error parsing constraint {constraint_expr}: {str(e)}'
                })
        
        # Set up KKT conditions
        try:
            kkt_conditions, solution = solve_with_kkt(list(variables.values()), obj_expr, eq_constraints, ineq_constraints, min_max)
            
            if solution:
                # Format the solution
                formatted_solution = {}
                for i, var_name in enumerate(variables.keys()):
                    formatted_solution[var_name] = float(solution[0][i])
                
                # Calculate objective value
                obj_value = float(obj_expr.subs({variables[var_name]: solution[0][i] for i, var_name in enumerate(variables.keys())}))
                if min_max == 'max':
                    obj_value = -obj_value
                
                return JsonResponse({
                    'success': True,
                    'solution': formatted_solution,
                    'objectiveValue': obj_value,
                    'kktConditions': kkt_conditions,
                    'message': "Solution found using KKT conditions",
                    'problem': {
                        'objective': objective_expr,
                        'constraints': constraints_list,
                        'minMax': min_max,
                        'problemType': 'nonlinear'
                    }
                })
            else:
                # If KKT fails, try numerical optimization
                return fallback_numerical_optimization(variables, obj_expr, eq_constraints, ineq_constraints, min_max, objective_expr, constraints_list)
        except Exception as e:
            logger.exception(f"Error in KKT solver: {str(e)}")
            # If KKT fails with an exception, try numerical optimization
            return fallback_numerical_optimization(variables, obj_expr, eq_constraints, ineq_constraints, min_max, objective_expr, constraints_list)
            
    except Exception as e:
        import traceback
        logger.exception(f"Exception in nonlinear solver: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

def fallback_numerical_optimization(variables, obj_expr, eq_constraints, ineq_constraints, min_max, objective_expr, constraints_list):
    """Fallback to numerical optimization if KKT fails."""
    try:
        var_list = list(variables.values())
        
        # Define objective function for numerical optimization
        def objective(x):
            # Substitute variable values
            substitutions = {var: val for var, val in zip(var_list, x)}
            return float(obj_expr.subs(substitutions))
        
        # Create constraints for scipy
        constraints = []
        
        # Equality constraints
        for eq in eq_constraints:
            def eq_constraint(x, eq=eq):
                substitutions = {var: val for var, val in zip(var_list, x)}
                return float(eq.subs(substitutions))
            
            constraints.append({'type': 'eq', 'fun': eq_constraint})
        
        # Inequality constraints
        for ineq in ineq_constraints:
            def ineq_constraint(x, ineq=ineq):
                substitutions = {var: val for var, val in zip(var_list, x)}
                return -float(ineq.subs(substitutions))  # Negative because scipy uses g(x) >= 0
            
            constraints.append({'type': 'ineq', 'fun': ineq_constraint})
        
        # Try multiple starting points to avoid local minima
        best_result = None
        best_objective = float('inf') if min_max == 'min' else float('-inf')
        
        # Define several starting points
        starting_points = [
            np.zeros(len(var_list)),  # Origin
            np.ones(len(var_list)),   # All ones
            np.random.rand(len(var_list)),  # Random point
            np.random.rand(len(var_list)) * 10,  # Scaled random point
        ]
        
        for start_point in starting_points:
            try:
                # Solve the optimization problem
                if min_max == 'max':
                    def neg_objective(x):
                        return -objective(x)
                    
                    result = optimize.minimize(
                        neg_objective,
                        start_point,
                        method='SLSQP',
                        constraints=constraints,
                        options={'maxiter': 1000, 'disp': False}
                    )
                    current_obj = -result.fun
                else:
                    result = optimize.minimize(
                        objective,
                        start_point,
                        method='SLSQP',
                        constraints=constraints,
                        options={'maxiter': 1000, 'disp': False}
                    )
                    current_obj = result.fun
                
                # Keep the best result
                if result.success:
                    if ((min_max == 'min' and current_obj < best_objective) or 
                        (min_max == 'max' and current_obj > best_objective)):
                        best_result = result
                        best_objective = current_obj
            except Exception as e:
                logger.warning(f"Optimization failed for starting point {start_point}: {str(e)}")
                continue
        
        # Use the best result if found
        if best_result is not None:
            result = best_result
            objective_value = best_objective
            
            # Format solution with variable names
            solution = {}
            for i, var_name in enumerate(variables.keys()):
                solution[var_name] = float(result.x[i])
            
            return JsonResponse({
                'success': True,
                'solution': solution,
                'objectiveValue': float(objective_value),
                'iterations': result.nit,
                'message': "KKT method failed; solution found using numerical optimization with multiple starting points.",
                'problem': {
                    'objective': objective_expr,
                    'constraints': constraints_list,
                    'minMax': min_max,
                    'problemType': 'nonlinear'
                }
            })
        else:
            return JsonResponse({
                'success': False,
                'error': "Optimization failed from all starting points. Try reformulating the problem."
            })
    except Exception as e:
        logger.exception("Exception in fallback numerical optimization")
        return JsonResponse({
            'success': False,
            'error': f"Fallback optimization failed: {str(e)}"
        })

def solve_with_kkt(variables, objective, eq_constraints, ineq_constraints, min_max='min'):
    """Apply KKT conditions to solve the nonlinear programming problem."""
    try:
        # Define Lagrange multipliers
        n_eq = len(eq_constraints)
        n_ineq = len(ineq_constraints)
        
        lambda_symbols = sp.symbols(f'lambda1:{n_eq+1}') if n_eq > 0 else []
        mu_symbols = sp.symbols(f'mu1:{n_ineq+1}') if n_ineq > 0 else []
        
        # Set up Lagrangian
        L = objective if min_max == 'min' else -objective
        
        # Add equality constraints
        for i, eq in enumerate(eq_constraints):
            L += lambda_symbols[i] * eq
        
        # Add inequality constraints
        for i, ineq in enumerate(ineq_constraints):
            L += mu_symbols[i] * ineq
        
        # Compute gradients for KKT conditions
        gradient_eqs = [sp.diff(L, var) for var in variables]
        
        # Complementary slackness
        complementary_slackness = [mu_symbols[i] * ineq_constraints[i] for i in range(n_ineq)]
        
        # Dual feasibility
        dual_feasibility = [mu >= 0 for mu in mu_symbols]
        
        # KKT conditions text representation
        kkt_conditions = []
        kkt_conditions.append("Stationarity conditions (gradient of Lagrangian = 0):")
        for i, eq in enumerate(gradient_eqs):
            kkt_conditions.append(f"∂L/∂{variables[i]} = {eq} = 0")
        
        kkt_conditions.append("\nPrimal feasibility:")
        for i, eq in enumerate(eq_constraints):
            kkt_conditions.append(f"h{i+1}(x) = {eq} = 0")
        for i, ineq in enumerate(ineq_constraints):
            kkt_conditions.append(f"g{i+1}(x) = {ineq} ≤ 0")
        
        if n_ineq > 0:
            kkt_conditions.append("\nDual feasibility:")
            for i, _ in enumerate(mu_symbols):
                kkt_conditions.append(f"μ{i+1} ≥ 0")
            
            kkt_conditions.append("\nComplementary slackness:")
            for i, cs in enumerate(complementary_slackness):
                kkt_conditions.append(f"μ{i+1}·g{i+1}(x) = {cs} = 0")
        
        # Try to solve the KKT system
        all_vars = list(variables) + list(lambda_symbols) + list(mu_symbols)
        all_eqs = gradient_eqs + eq_constraints + complementary_slackness
        
        try:
            solutions = sp.solve(all_eqs, all_vars, dict=True)
            
            # Filter solutions for feasibility
            feasible_solutions = []
            for sol in solutions:
                is_feasible = True
                
                # Check primal feasibility for inequalities
                for ineq in ineq_constraints:
                    val = ineq.subs(sol)
                    if val > 1e-6:  # Some tolerance for numerical issues
                        is_feasible = False
                        break
                
                # Check dual feasibility
                for mu in mu_symbols:
                    if mu in sol and sol[mu] < -1e-6:  # Negative tolerance for numerical issues
                        is_feasible = False
                        break
                
                if is_feasible:
                    var_values = [sol.get(var, 0) for var in variables]
                    lambda_values = [sol.get(lam, 0) for lam in lambda_symbols]
                    mu_values = [sol.get(mu, 0) for mu in mu_symbols]
                    feasible_solutions.append((var_values, lambda_values, mu_values))
            
            if feasible_solutions:
                # Evaluate objective for each solution and pick the best
                min_obj = float('inf') if min_max == 'min' else float('-inf')
                best_sol = None
                
                for sol in feasible_solutions:
                    var_vals = sol[0]
                    obj_val = objective.subs({variables[i]: var_vals[i] for i in range(len(variables))})
                    
                    if (min_max == 'min' and obj_val < min_obj) or (min_max == 'max' and obj_val > min_obj):
                        min_obj = obj_val
                        best_sol = sol
                
                return kkt_conditions, best_sol
            
            return kkt_conditions, None
        except Exception as e:
            logger.exception(f"Error solving KKT system: {str(e)}")
            return [f"Error solving KKT conditions: {str(e)}"] + kkt_conditions, None
    except Exception as e:
        logger.exception(f"Error in KKT setup: {str(e)}")
        return [f"Error in KKT setup: {str(e)}"], None
    