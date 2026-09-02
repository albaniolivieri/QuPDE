from typing import Optional, Union
import sympy as sp
from .quadratization import quadratize, print_quad
from .utils import get_sys_order

non_pol_funcs = [sp.exp, sp.sin, sp.cos, sp.Pow]


def polynomialize(
    func_eq: list[tuple[sp.Function, sp.Expr]],
    first_indep: sp.Symbol = sp.symbols("t"),
    second_indep: sp.Symbol = sp.symbols("x"),
    is_rat: bool = False,
) -> tuple[list[tuple[sp.Function, sp.Expr]], list[tuple[sp.Expr, sp.Expr]]]:
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
    while j < len(poly_func_eq):
        while True:
            new_eqs = []
            new_var = get_non_pol_var(
                poly_func_eq[j][1], is_rat=is_rat, func_syms=func_syms
            )
            if not new_var:
                break
            if (
                new_var.func == sp.Pow
                and new_var.args[1].is_number
                and not new_var.args[1].is_integer
                and new_var.args[1] < 1
            ):
                new_eqs, count = handle_rational_power(
                    new_var,
                    new_vars,
                    poly_func_eq,
                    count,
                    is_rat,
                    first_indep,
                    second_indep,
                )
            else:
                var_name = sp.Function(f"p_{count}")(first_indep, second_indep)
                new_vars.append((new_var, var_name))
                new_eqs.append(
                    (
                        var_name,
                        get_new_eq(new_var, poly_func_eq, first_indep).subs(new_vars),
                    )
                )
                count += 1

            poly_func_eq.extend(new_eqs)
            for i in range(len(poly_func_eq)):
                poly_func_eq[i] = (
                    poly_func_eq[i][0],
                    poly_func_eq[i][1].subs(new_vars),
                )
        j += 1
    return poly_func_eq, new_vars


def get_non_pol_var(
    expr: sp.Expr,
    is_rat: bool,
    func_syms: list[sp.Symbol],
    new_var: Optional[sp.Expr] = None,
) -> Optional[sp.Expr]:
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
        if expr.func == sp.Pow:
            exp = sp.nsimplify(args[1]) if args[1].is_Float else args[1]
            if exp.is_integer:
                if exp > 0 or (exp < 0 and is_rat):
                    return get_non_pol_var(args[0], is_rat, func_syms, new_var=new_var)
        for arg in args:
            if bool(arg.free_symbols & set(func_syms)):
                return get_non_pol_var(arg, is_rat, func_syms, new_var=expr)
    else:
        if expr.func == sp.Symbol or expr.func == sp.Integer:
            return new_var
        else:
            for arg in args:
                new_var = get_non_pol_var(arg, is_rat, func_syms, new_var=new_var)
                if new_var is not None:
                    new_var = new_var
    return new_var


def handle_rational_power(
    new_var: sp.Expr,
    new_vars: list[tuple[sp.Expr, sp.Expr]],
    poly_func_eq: list[tuple[sp.Function, sp.Expr]],
    count: int,
    is_rat: bool,
    first_indep: sp.Symbol,
    second_indep: sp.Symbol,
) -> tuple[list[tuple[sp.Function, sp.Expr]], int]:
    """Derives auxiliary variables for a fractional-power non-polynomial expression.

    Handles two cases depending on the sign of the exponent (alpha):
    - alpha < 0: introduces base^(alpha + neg_power) as the new variable
    - alpha > 0: introduces base^alpha directly as the new variable

    Parameters
    ----------
    new_var
        The fractional-power expression
    new_vars
        Accumulated list of auxiliary variables in the form (expression, symbol)
    poly_func_eq
        Current system of equations as (function, rhs) tuples
    count
        Current auxiliary variable counter
    is_rat
        If False, 1/base is introduced as p_i
    first_indep
        First independent variable
    second_indep
        Second independent variable

    Returns
    -------
    tuple[list[tuple], int]
        New equations introduced and the updated counter
    """
    new_eqs = []
    rat_var = new_var.args[0] ** -1
    neg_power = sp.ceiling(abs(new_var.args[1]))
    dummy = sp.Dummy("inv")
    if is_rat:
        rat_var_tup = (rat_var, dummy)
    else:
        if not any(var[0].equals(rat_var) for var in new_vars):
            var_name = sp.Function(f"p_{count}")(first_indep, second_indep)
            new_vars.append((rat_var, var_name))
            count += 1
            poly_func_eq[:] = [(func, rhs.subs(new_vars)) for func, rhs in poly_func_eq]
            new_eqs.append(
                (
                    var_name,
                    get_new_eq(
                        rat_var,
                        poly_func_eq,
                        first_indep,
                        frac_var=(rat_var, var_name),
                        orig_var=new_var,
                        neg_power=neg_power,
                    ).subs(new_vars),
                )
            )
        rat_var_tup = next(t for t in new_vars if t[0] == rat_var)
    if new_var.args[1] < 0 and not any(
        var[0].equals(new_var.args[0] ** (new_var.args[1] + neg_power))
        for var in new_vars
    ):
        var_name = sp.Function(f"p_{count}")(first_indep, second_indep)
        float_var = new_var.args[0] ** (new_var.args[1] + neg_power)
        new_vars.append((float_var, var_name))
        count += 1
        new_eqs.append(
            (
                var_name,
                get_new_eq(
                    float_var,
                    poly_func_eq,
                    first_indep,
                    frac_var=rat_var_tup,
                    neg_power=neg_power,
                )
                .subs(new_vars)
                .subs(dummy, rat_var),
            )
        )
        poly_func_eq[:] = [(func, rhs.subs(new_vars)) for func, rhs in poly_func_eq]
    elif new_var.args[1] > 0 and not any(
        var[0].equals(new_var.args[0] ** (new_var.args[1])) for var in new_vars
    ):
        var_name = sp.Function(f"p_{count}")(first_indep, second_indep)
        new_vars.append((new_var.args[0] ** (new_var.args[1]), var_name))
        count += 1
        new_eqs.append(
            (
                var_name,
                get_new_eq(
                    new_var.args[0] ** (new_var.args[1]),
                    poly_func_eq,
                    first_indep,
                    frac_var=rat_var_tup,
                    neg_power=neg_power,
                )
                .subs(new_vars)
                .subs(dummy, rat_var),
            )
        )
        poly_func_eq[:] = [(func, rhs.subs(new_vars)) for func, rhs in poly_func_eq]
    return new_eqs, count


def get_new_eq(
    new_var: sp.Expr,
    func_eq: list[tuple[sp.Function, sp.Expr]],
    first_indep: sp.Symbol,
    frac_var: Optional[tuple[sp.Expr, sp.Expr]] = None,
    orig_var: Union[sp.Expr, int] = 0,
    neg_power: int = 1,
) -> sp.Expr:
    """Derives the evolution equation for a newly introduced auxiliary variable

    Parameters
    ----------
    new_var
        The non-polynomial sub-expression being introduced as a new variable
    func_eq
        The current system of equations as tuples (function, rhs expression)
    first_indep
        The first independent variable (differentiation variable)
    frac_var : optional
        Tuple (rat_expr, sym) where rat_expr is base^(-1) and sym is the symbol
        representing 1/base in the chain rule computation
    orig_var : optional
        The original non-polynomial variable that triggered the introduction of an
        auxiliary variable of type base^(-1). Default is 0
    neg_power : optional
        Ceiling of the absolute value of the fractional exponent of new_var.
        Sets the power of the 1/base symbol. Default is 1

    Returns
    -------
    sp.Expr
        The derived equation for the new auxiliary variable
    """
    refac = [(sp.Derivative(func, first_indep), eq) for func, eq in func_eq]
    wt_expr = sp.nsimplify(
        sp.Derivative(new_var, first_indep).doit().subs(refac).doit()
    )
    wt = wt_expr
    if frac_var is not None and orig_var == 0:
        base = new_var.args[0]
        alpha = sp.nsimplify(new_var.args[1])
        for atom in wt_expr.atoms(sp.Pow):
            if atom.args[0] == base:
                beta = sp.nsimplify(atom.args[1])
                # solves base^(neg_power) * base^(beta) = base^(m*alpha)
                m = (beta + neg_power) / alpha  
                if m.is_integer and m >= 0:
                    wt = wt_expr.subs(
                        atom, frac_var[1] ** neg_power * base ** (m * alpha)
                    )
                break
    elif orig_var != 0:
        wt = wt_expr.subs(
            sp.nsimplify(new_var.args[0] ** (orig_var.args[1] + (new_var.args[1] - 1))),
            frac_var[1] ** (1 + neg_power) * new_var.args[0] ** (orig_var.args[1]),
        )
    return wt


def polynomialize_and_quadratize(
    func_eq,
    diff_ord: Optional[int] = None,
    first_indep=sp.symbols("t"),
    second_indep=sp.symbols("x"),
    nvars_bound=10,
    printing: Optional[str] = "", #'pprint' or 'latex'
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
    printing : optional
        If 'pprint', prints the quadratization in a human-readable format.
        If 'latex', prints the quadratization in LaTeX format.

    Returns
    -------
    PDESys
        The result of the quadratization applied to the polynomialized system
    """
    poly_sys, aux_poly = polynomialize(func_eq, first_indep, second_indep, is_rat=True)
    quadratic_sys = quadratize(
        poly_sys, diff_ord=diff_ord, first_indep=first_indep, nvars_bound=nvars_bound
    )
    if quadratic_sys == []:
        return (poly_sys, aux_poly)
    pde_order = get_sys_order([expr for _, expr in func_eq])
    der_subs = []
    for fun, _ in func_eq:
        der_subs += [
            (
                sp.Derivative(fun, second_indep, i),
                sp.symbols(f"{fun.name}_{second_indep.name}{i}"),
            )
            for i in range(pde_order, 0, -1)
        ] + [(fun, sp.symbols(fun.name))]
    nonpol_vars = [
        (sp.symbols(fun.name), expr.subs(der_subs)) for expr, fun in aux_poly
    ]
    quadratic_sys.set_new_vars(nonpol_vars=nonpol_vars)
    if printing:
        print_quad(quadratic_sys, p_style=printing)
    return quadratic_sys
