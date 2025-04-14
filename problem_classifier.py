import sympy as sp
from typing import Tuple, List, Dict

def analyze_objective_function(expr_str: str, variables: List[str]) -> Dict:
    """
    Analyze the objective function and classify the optimization problem.
    
    Args:
        expr_str: String representation of the objective function
        variables: List of variable names
        
    Returns:
        Dictionary containing problem classification and characteristics
    """
    # Create symbols for variables
    sym_vars = {var: sp.Symbol(var) for var in variables}
    
    # Parse the expression
    expr = sp.sympify(expr_str)
    
    # Initialize result dictionary
    result = {
        'degree': None,
        'has_cross_terms': False,
        'problem_type': None,
        'is_quadratic': False,
        'is_nlp': False
    }
    
    # Get the polynomial degree
    result['degree'] = sp.degree(expr)
    
    # Check for cross terms using a simpler method
    # First, expand the expression to ensure all terms are visible
    expanded_expr = sp.expand(expr)
    
    # Check for cross terms between all pairs of variables
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            var1 = sym_vars[variables[i]]
            var2 = sym_vars[variables[j]]
            
            # Check if the coefficient of var1*var2 is non-zero
            if expanded_expr.coeff(var1 * var2) != 0:
                result['has_cross_terms'] = True
                break
        if result['has_cross_terms']:
            break
    
    # Classify the problem
    if result['degree'] > 2:
        result['problem_type'] = 'NLP'
        result['is_nlp'] = True
        result['is_quadratic'] = False
    elif result['degree'] == 2:
        if result['has_cross_terms']:
            result['problem_type'] = 'NLP'
            result['is_nlp'] = True
            result['is_quadratic'] = False
        else:
            result['problem_type'] = 'Quadratic'
            result['is_quadratic'] = True
            result['is_nlp'] = False
    else:
        result['problem_type'] = 'Linear'
        result['is_quadratic'] = False
        result['is_nlp'] = False
    
    return result

def classify_problem(expr_str: str, variables: List[str]) -> str:
    """
    Classify the optimization problem based on the objective function.
    
    Args:
        expr_str: String representation of the objective function
        variables: List of variable names
        
    Returns:
        String describing the problem type
    """
    analysis = analyze_objective_function(expr_str, variables)
    
    if analysis['degree'] > 2:
        return "This is a Nonlinear Programming (NLP) problem because the degree is greater than 2"
    elif analysis['degree'] == 2:
        if analysis['has_cross_terms']:
            return "This is a Nonlinear Programming (NLP) problem because it has cross terms"
        else:
            return "This is a Quadratic Programming problem"
    else:
        return "This is a Linear Programming problem" 