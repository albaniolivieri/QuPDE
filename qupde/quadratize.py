from typing import Optional, Callable
import sympy as sp
from sympy.polys.rings import PolyElement
from .rat_sys import RatSys
from .search_quad import bnb, nearest_neighbor
from .mon_heuristics import by_fun, by_degree_order, by_order_degree

def quadratize(
    func_eq: list[tuple[sp.Function, sp.Expr]],
    diff_ord: Optional[int] = 3,
    sort_fun: Optional[str] = 'by_fun',
    nvars_bound: Optional[int] = 10,
    first_indep: Optional[sp.Symbol] = sp.symbols("t"),
    max_der_order: Optional[int] = None,
    search_alg: Optional[str] = 'bnb', # 'bnb' or 'inn'
    printing: Optional[str] = '', #'pprint' or 'latex'
) -> tuple[list[PolyElement], list[PolyElement], int]:
    """Quadratizes a given PDE

    Parameters
    ----------
    func_eq
        Tuples with the symbol and equations of the PDE
    diff_ord : optional
        The differentiation order of the quadratization 
    sort_fun : optional
        The function to sort the proposed new variables
    nvars_bound : optional
        The maximum number of variables in the quadratization
    first_indep : optional
        The first independent variable of the PDE
    max_der_order : optional
        The maximum order of derivatives allowed in the new variables
    search_alg : optional
        The search algorithm to use. 'bnb' for branch and bound, 'nn' for incremental nearest neighbor
    print_quad : optional
        If 'pprint', prints the quadratization in a human-readable format.
        If 'latex', prints the quadratization in LaTeX format.

    Returns
    -------
    tuple[list[PolyElement], tuple[sp.Symbol, PolyElement], int]
        a tuple with the best quadratization found, the variables introduced
        from rational expressions and the total number of traversed nodes
    """
    undef_fun = [symbol for symbol, _, in func_eq]
    x_var = [
        symbol for symbol in undef_fun[0].free_symbols if symbol != first_indep
    ].pop()
    
    poly_syst = RatSys(func_eq, diff_ord, (first_indep, x_var))
    vars_frac_intro = poly_syst.get_frac_vars()
    quad = []
    nodes = 0
    
    if sort_fun == 'by_fun':
        sort_fun = by_fun 
    elif sort_fun == 'by_degree_order':
        sort_fun = by_degree_order
    elif sort_fun == 'by_order_degree':
        sort_fun = by_order_degree
    else: 
        raise ValueError(f"Unknown sorting function: {sort_fun}")

    if search_alg == 'inn':
        quad, nodes = nearest_neighbor(poly_syst, sort_fun, new_vars=[])
    elif search_alg == 'bnb':
        quad, _, nodes = bnb([], nvars_bound, poly_syst, sort_fun, max_der_order)
    if not quad and not vars_frac_intro:
        print("Quadratization not found")
        return vars_frac_intro, nodes
    
    if printing:
        print_quad(func_eq, quad, vars_frac_intro, diff_ord, first_indep, p_style=printing)
        
    return quad, vars_frac_intro, nodes


def check_quadratization(
    func_eq: list[tuple[sp.Function, sp.Expr]],
    new_vars: list[PolyElement],
    n_diff: int,
    first_indep: Optional[sp.Symbol] = sp.symbols("t"),
) -> bool: 
    """Checks if a given set of new variables is a quadratization for the provided PDE

    Parameters
    ----------
    func_eq 
        Tuples with the symbol and equations of the PDE
    new_vars
        List of proposed new variables
    n_diff
        The number of second variable differentiations to do
    first_indep : optional
        The first independent variable of the PDE

    Returns
    -------
    bool
        True if the proposed quadratization is valid, False otherwise
    """
    undef_fun = [symbol for symbol, _, in func_eq]
    x_var = [
        symbol for symbol in undef_fun[0].free_symbols if symbol != first_indep
    ].pop()

    poly_syst = RatSys(func_eq, n_diff, (first_indep, x_var), new_vars)

    return poly_syst.try_make_quadratic()

def print_quad(pde, new_vars, vars_frac_intro, n_diff, first_indep, p_style):
    _, quad = check_quadratization(pde, new_vars, n_diff, first_indep = first_indep)
    new_vars_named = [(sp.symbols(f'w_{i}'), pol)
                          for i, pol in enumerate(new_vars)]
    print("\nQuadratization:")
    for name, var in new_vars_named:
        if p_style == 'latex':
            print(sp.latex(sp.Eq(name, var.as_expr())))
        else:
            sp.pprint(sp.Eq(name, var.as_expr()))
    for name, var in vars_frac_intro:
        if p_style == 'latex':
            print(sp.latex(sp.Eq(name, 1/var.as_expr())))
        else:
            sp.pprint(sp.Eq(name, 1/var.as_expr()))
    print("\nQuadratic PDE:")
    for exprs in quad:
        if p_style == 'latex':
            print(sp.latex(exprs))
        else:
            sp.pprint(exprs)
    