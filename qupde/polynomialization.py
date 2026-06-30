from typing import Optional
from sympy import symbols, Function, exp, sin, cos, Pow, Symbol, Integer, nsimplify
from sympy import Derivative as D
from .quadratization import quadratize, print_quad

non_pol_funcs = [exp, sin, cos, Pow]

def polynomialize(func_eq, first_indep=symbols('t'), second_indep=symbols('x'), is_rat=False):
    """Transforms a system of nonlinear and non-polynomial equations into a polynomial 
    system by introducing auxiliary variables.

    Parameters
    ----------
    func_eq
        Tuples with the unknown functions and corresponding equations of the PDE
    first_indep : optional
        The first independent variable of the PDE
    second_indep : optional
        The second independent variable of the PDE
    is_rat : optional
        If True, negative integer powers are considered polynomial and are not
        replaced by auxiliary variables

    Returns
    -------
    tuple[list[tuple], list[tuple]]
        a tuple with the polynomialized system of equations and the list of
        substitutions (original expression, new auxiliary function) introduced
    """
    count = 0
    j = 0
    new_vars = []
    func_syms = [first_indep, second_indep]
    poly_func_eq = func_eq.copy()
    while j in range(len(poly_func_eq)):
        while True:
            new_var = get_non_pol_var(poly_func_eq[j][1], is_rat=is_rat, func_syms=func_syms)
            if not new_var:
                break
            new_vars.append((new_var, Function(f'p_{count}')(first_indep, second_indep)))
            for i in range(len(poly_func_eq)):
                poly_func_eq[i] = (poly_func_eq[i][0], poly_func_eq[i][1].subs(new_vars))     
            new_eq = get_new_eq(new_var, poly_func_eq, first_indep).subs(new_vars)
            poly_func_eq.append((Function(f'p_{count}')(first_indep, second_indep), new_eq))
            count += 1
        j += 1
    return poly_func_eq, new_vars
    
def get_non_pol_var(expr, is_rat, func_syms, new_var = None):
    """Recursively searches an expression tree for the innermost non-polynomial
    sub-expression (exp, sin, cos, or non-integer/negative power).

    Parameters
    ----------
    expr
        The sympy expression to search
    is_rat
        If True, negative integer powers are considered polynomial
    func_syms
        List of independent variable symbols
    new_var : optional
        The current candidate non-polynomial variable found during recursion

    Returns
    -------
    expr or None
        The non-polynomial sub-expression found, or None if the expression is
        already polynomial
    """
    args = expr.args
    if expr.func in non_pol_funcs:
        if expr.func == Pow: 
            exp = nsimplify(args[1]) if args[1].is_Float else args[1]
            if exp.is_integer:
                if exp>0 or (exp<0 and is_rat): 
                    return get_non_pol_var(args[0], is_rat, func_syms, new_var=new_var)
        for arg in args:
            if bool(arg.free_symbols & set(func_syms)):
                return get_non_pol_var(arg, is_rat, func_syms, new_var=expr)
            # else: return new_var
    else: 
        if expr.func == Symbol or expr.func == Integer: return new_var
        else: 
            for arg in args:
                new_var = get_non_pol_var(arg, is_rat, func_syms, new_var=new_var)
                if new_var is not None: return new_var
    return new_var

def get_new_eq(new_var, func_eq, first_indep):
    """Derives the evolution equation for a newly introduced auxiliary variable by
    differentiating it with respect to the first independent variable and substituting
    the original system's equations.

    Parameters
    ----------
    new_var
        The non-polynomial sub-expression introduced as a new variable
    func_eq
        The current system of equations as tuples (function, expression)
    first_indep
        The first independent variable

    Returns
    -------
    sp.Expr
        The derived equation for the new auxiliary variable
    """
    refac = [(D(func, first_indep), eq) for func, eq in func_eq]
    wt = D(new_var, first_indep).doit().subs(refac).doit()
    return wt

def polynomialize_and_quadratize(
    func_eq, diff_ord: None, 
    first_indep=symbols('t'),
    second_indep=symbols('x'), 
    nvars_bound=10,
    printing: Optional[str] = "",  #'pprint' or 'latex'
):
    """Combines polynomialization and quadratization into a single step by first
    transforming the system into polynomial form and then finding a quadratization.

    Parameters
    ----------
    func_eq
        Tuples with the unknown functions and corresponding equations of the PDE
    diff_ord
        The differentiation order of the quadratization
    first_indep : optional
        The first independent variable of the PDE
    second_indep : optional
        The second independent variable of the PDE
    nvars_bound : optional
        The maximum number of variables in the quadratization

    Returns
    -------
    PDESys
        The result of the quadratization applied to the polynomialized system
    """
    poly_sys, aux_poly = polynomialize(func_eq, first_indep, second_indep, is_rat=True)
    
    quadratic_sys = quadratize(poly_sys, diff_ord=diff_ord, first_indep=symbols('t'), nvars_bound=nvars_bound)
    nonpol_vars = ((name, expr) for expr, name in aux_poly)
    quadratic_sys.set_new_vars(nonpol_vars=nonpol_vars)
    if printing:
        print_quad(quadratic_sys, p_style=printing)
    