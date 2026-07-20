from typing import Optional
import sympy as sp
from .quadratization import quadratize, print_quad
from .utils import get_sys_order

non_pol_funcs = [sp.exp, sp.sin, sp.cos, sp.Pow]

def polynomialize(func_eq, first_indep=sp.symbols('t'), second_indep=sp.symbols('x'), is_rat=False):
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
            new_var = get_non_pol_var(poly_func_eq[j][1], is_rat=is_rat, func_syms=func_syms)
            if not new_var:
                break
            if new_var.func == sp.Pow and new_var.args[1].is_number and not new_var.args[1].is_integer and new_var.args[1] < 1:
                rat_var = new_var.args[0]**-1
                neg_power = sp.ceiling(abs(new_var.args[1]))
                dummy = sp.Dummy('inv')
                if is_rat:
                    rat_var_tup = (rat_var, dummy)  
                else:
                    if not any(var[0].equals(rat_var) for var in new_vars):
                        var_name = sp.Function(f'p_{count}')(first_indep, second_indep)
                        new_vars.append((rat_var, var_name))
                        count += 1
                        poly_func_eq[:] = [(func, rhs.subs(new_vars)) for func, rhs in poly_func_eq]
                        new_eqs.append((var_name, get_new_eq(rat_var, poly_func_eq, first_indep, frac_var=(rat_var, var_name), orig_var=new_var, neg_power=neg_power).subs(new_vars)))
                    rat_var_tup = next(t for t in new_vars if t[0] == rat_var)
                if new_var.args[1] < 0 and not any(var[0].equals(new_var.args[0]**(new_var.args[1]+neg_power)) for var in new_vars):
                    var_name = sp.Function(f'p_{count}')(first_indep, second_indep)
                    float_var = new_var.args[0]**(new_var.args[1]+neg_power)
                    new_vars.append((float_var, var_name))
                    count += 1
                    new_eqs.append((var_name, get_new_eq(float_var, poly_func_eq, first_indep, frac_var=rat_var_tup, neg_power=neg_power).subs(new_vars).subs(dummy, rat_var)))
                    poly_func_eq[:] = [(func, rhs.subs(new_vars)) for func, rhs in poly_func_eq]
                elif new_var.args[1] > 0 and not any(var[0].equals(new_var.args[0]**(new_var.args[1])) for var in new_vars):
                    var_name = sp.Function(f'p_{count}')(first_indep, second_indep)
                    new_vars.append((new_var.args[0]**(new_var.args[1]), var_name))
                    count += 1
                    new_eqs.append((var_name, get_new_eq(new_var.args[0]**(new_var.args[1]), poly_func_eq, first_indep, frac_var=rat_var_tup, neg_power=neg_power).subs(new_vars).subs(dummy, rat_var)))
                    poly_func_eq[:] = [(func, rhs.subs(new_vars)) for func, rhs in poly_func_eq]
                else:
                    pass
            else:
                var_name = sp.Function(f'p_{count}')(first_indep, second_indep)
                new_vars.append((new_var, var_name))
                new_eqs.append((var_name, get_new_eq(new_var, poly_func_eq, first_indep).subs(new_vars)))
                count += 1

            poly_func_eq.extend(new_eqs)
            for i in range(len(poly_func_eq)):
                poly_func_eq[i] = (poly_func_eq[i][0], poly_func_eq[i][1].subs(new_vars))
        j += 1
    return poly_func_eq, new_vars

def get_non_pol_var(expr, is_rat, func_syms, new_var=None):
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
        if expr.func == sp.Symbol or expr.func == sp.Integer: return new_var
        else:
            for arg in args:
                new_var = get_non_pol_var(arg, is_rat, func_syms, new_var=new_var)
                if new_var is not None:
                    new_var = new_var
    return new_var

def get_new_eq(new_var, func_eq, first_indep, frac_var=None, orig_var=0, neg_power=1):
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
    refac = [(sp.Derivative(func, first_indep), eq) for func, eq in func_eq]
    wt_expr = sp.nsimplify(sp.Derivative(new_var, first_indep).doit().subs(refac).doit())
    wt = wt_expr
    if frac_var is not None and orig_var == 0:
        base = new_var.args[0]
        alpha = sp.nsimplify(new_var.args[1])
        for atom in wt_expr.atoms(sp.Pow):
            if atom.args[0] == base:
                beta = sp.nsimplify(atom.args[1])
                m = (beta + neg_power) / alpha
                if m.is_integer and m >= 0:
                    wt = wt_expr.subs(atom, frac_var[1]**neg_power * base**(m * alpha))
                break
    elif orig_var != 0:
        wt = wt_expr.subs(
            sp.nsimplify(new_var.args[0]**(orig_var.args[1]+(new_var.args[1]-1))),
            frac_var[1]**(1+neg_power)*new_var.args[0]**(orig_var.args[1]))        
    return wt

def polynomialize_and_quadratize(
    func_eq, diff_ord: Optional[int] = None,
    first_indep=sp.symbols('t'),
    second_indep=sp.symbols('x'),
    nvars_bound=10,
    printing: Optional[str] = "",
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
    quadratic_sys = quadratize(poly_sys, diff_ord=diff_ord, first_indep=first_indep, nvars_bound=nvars_bound)
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
    nonpol_vars = [(sp.symbols(fun.name), expr.subs(der_subs)) for expr, fun in aux_poly]
    quadratic_sys.set_new_vars(nonpol_vars=nonpol_vars)
    if printing:
        print_quad(quadratic_sys, p_style=printing)
    return quadratic_sys
