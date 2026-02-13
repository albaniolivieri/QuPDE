from typing import Optional, Union
import sympy as sp
from sympy.polys.rings import PolyElement
from .pde_sys import PDESys
from .search_quad import bnb, nearest_neighbor
from .mon_heuristics import by_fun, by_degree_order, by_order_degree
from .utils import get_sys_order


def quadratize(
    func_eq: list[tuple[sp.Function, sp.Expr]],
    diff_ord: Optional[int] = None,
    sort_fun: Optional[str] = "by_fun",
    nvars_bound: Optional[int] = 10,
    first_indep: Optional[Union[sp.Symbol, str]] = sp.symbols("t"),
    max_der_order: Optional[int] = None,
    search_alg: Optional[str] = "bnb",  # 'bnb' or 'inn'
    printing: Optional[str] = "",  #'pprint' or 'latex'
    show_nodes: bool = False,
) -> tuple[list[PolyElement], list[PolyElement], int]:
    """Quadratizes a given PDE

    Parameters
    ----------
    func_eq
        Tuples with the unknown functions and corresponding equations of the PDE
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
    show_nodes : optional
        If True, returns the number of nodes traversed by the algorithm

    Returns
    -------
    tuple[list[PolyElement], tuple[sp.Symbol, PolyElement], int]
        a tuple with the best quadratization found, the variables introduced
        from rational expressions and the total number of traversed nodes
    """
    if not func_eq:
        raise ValueError("The differential equations list is empty")
    if not isinstance(func_eq[0], tuple):
        raise ValueError(
            "The differential equations list must be a list of tuples of the type (Function, Expression)"
        )

    undef_fun = [symbol for symbol, _ in func_eq]
    x_var = [
        symbol for symbol in undef_fun[0].free_symbols if symbol != first_indep
    ].pop()

    if diff_ord is None:
        diff_ord = 3 * get_sys_order([expr for _, expr in func_eq])
    elif not isinstance(diff_ord, int) or diff_ord < 0:
        raise ValueError("The differentiation order must be a non-negative integer")

    if isinstance(first_indep, str):
        first_indep = sp.symbols(first_indep)

    poly_syst = PDESys(func_eq, diff_ord, (first_indep, x_var))
    quad = []
    nodes = 0

    dic_sort_fun = {
        "by_fun": by_fun,
        "by_degree_order": by_degree_order,
        "by_order_degree": by_order_degree,
    }

    try:
        sort_fun = dic_sort_fun[sort_fun]
    except KeyError:
        raise ValueError(f"Unknown sorting function: {sort_fun}")

    if not isinstance(nvars_bound, int) or nvars_bound <= 0:
        raise ValueError(
            "The bound on the number of variables must be a positive integer"
        )
    if max_der_order is not None:
        if not isinstance(max_der_order, int) or max_der_order < 0:
            raise ValueError("The maximum derivative order must be a positive integer")

    if search_alg == "inn":
        quad, nodes = nearest_neighbor(poly_syst, sort_fun, new_vars=[])
    elif search_alg == "bnb":
        quad, _, nodes = bnb([], nvars_bound, poly_syst, sort_fun, max_der_order)
    else:
        raise ValueError(f"Unknown search algorithm: {search_alg}")
    if quad is None:
        print("Quadratization not found")
        if show_nodes:
            return [], nodes
        else:
            return []
    poly_syst.set_new_vars(quad)
    _, quad_syst = poly_syst.try_make_quadratic()
    poly_syst.set_quad_sys(quad_syst)

    if printing:
        print_quad(poly_syst, p_style=printing)

    if not isinstance(show_nodes, bool):
        raise ValueError("The show_nodes parameter must be a boolean")

    if show_nodes:
        print("Nodes traversed:", nodes)
        return poly_syst, nodes

    return poly_syst


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
    undef_fun = [symbol for symbol, _ in func_eq]
    x_var = [
        symbol for symbol in undef_fun[0].free_symbols if symbol != first_indep
    ].pop()

    poly_syst = PDESys(func_eq, n_diff, (first_indep, x_var), new_vars)

    return poly_syst.try_make_quadratic()


def print_quad(poly_syst, p_style):
    if p_style not in ["pprint", "latex"]:
        raise ValueError(f"Unknown printing style: {p_style}")
    new_pde = poly_syst.get_quad_sys()
    new_vars_named = [
        (sp.symbols(f"w_{i}"), pol) for i, pol in enumerate(poly_syst.get_aux_vars()[0])
    ]
    print("\nQuadratization:")
    for name, var in new_vars_named:
        if p_style == "latex":
            print(sp.latex(sp.Eq(name, var.as_expr())))
        else:
            sp.pprint(sp.Eq(name, var.as_expr()))
    for name, var in poly_syst.get_aux_vars()[1]:
        if p_style == "latex":
            print(sp.latex(sp.Eq(name, 1 / var.as_expr())))
        else:
            sp.pprint(sp.Eq(name, 1 / var.as_expr()))
    print("\nQuadratic PDE:")
    for exprs in new_pde:
        if p_style == "latex":
            print(sp.latex(exprs))
        else:
            sp.pprint(exprs)
