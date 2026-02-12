from __future__ import annotations

from typing import Optional, Callable, TYPE_CHECKING
import re
from functools import reduce
import sympy as sp
from sympy import Derivative as D
from sympy.polys.rings import PolyElement

if TYPE_CHECKING:
    from .fraction_decomp import FractionDecomp


def get_sys_order(set_derivs: list[sp.Symbol]) -> int:
    """Gets the maximum order of derivatives in a list of derivatives

    Parameters
    ----------
    set_derivs
        List with the derivatives as expressions

    Returns
    -------
    int
        the maximum order of derivatives
    """
    max_order = 0
    for deriv in set_derivs:
        max_order = reduce(max, [der.args[1][1] for der in deriv.atoms(D)], max_order)
    return max_order


def reduction(
    pol1: tuple[sp.Expr, PolyElement], pol2: tuple[sp.Expr, PolyElement]
) -> tuple[sp.Expr, PolyElement, PolyElement]:
    """Reduces the polynomial pol1 by the polynomial pol2

    Parameters
    ----------
    pol1
        the polynomial to be reduced as a tuple with the symbol, the polynomial and the leading monomial
    pol2
        the polynomial to reduce by as a tuple with the symbol, the polynomial and the leading monomial

    Returns
    -------
    tuple[sp.Expr, PolyElement, PolyElement]
        the reduced polynomial as a tuple with the symbol, the polynomial and the leading monomial
    """
    if pol2[1] != 0:
        coef = pol1[1].coeff(pol2[2])
        if coef != 0:
            new_pol = pol1[1] - coef * pol2[1]
            return (
                pol1[0] - coef.as_expr() * pol2[0],
                new_pol,
                new_pol.leading_monom(),
            )
    return pol1


def diff_dict(
    pol: PolyElement,
    dic: dict,
    frac_decomp: Optional[FractionDecomp] = None,
    order: Optional[int] = 1,
) -> PolyElement:
    """differentiates a polynomial by a dictionary of variables

    Parameters
    ----------
    pol
        polynomial to be differentiated
    dic
        dictionary with the variables to differentiate by
    frac_decomp : optional
        the fraction decomposition of the PDE if it has one (default is None)
    order : optional
        the order of differentiation (default is 1)

    Returns
    -------
    PolyElement
        the differentiated polynomial
    """
    deriv = pol
    for _ in range(1, order + 1):
        deriv = sum(deriv.diff(k) * v for (k, v) in dic.items())
    if frac_decomp:
        return pol.ring(frac_decomp.try_reduce(deriv))
    return deriv


def get_decompositions(monomial: tuple) -> set[tuple]:
    """Returns the decompositions of a monomial to turn it into linear or quadratic

    Parameters
    ----------
    monomial
        Monomial to decompose

    Returns
    -------
    set[tuple]
        the decompositions of the monomial
    """
    if len(monomial) == 0:
        return {(tuple(), tuple())}
    result = set()
    prev_result = get_decompositions(tuple(monomial[:-1]))
    for r in prev_result:
        for i in range(monomial[-1] + 1):
            a, b = tuple(list(r[0]) + [i]), tuple(list(r[1]) + [monomial[-1] - i])
            result.add((min(a, b), max(a, b)))
    return result


def sort_vars(var_list: list[PolyElement], fun: Callable) -> list[PolyElement]:
    """Sorts the list of variables according to the function fun

    Parameters
    ----------
    var_list
        List of variables to sort
    fun
        Function to sort the variables

    Returns
    -------
    list[PolyElement]
        the sorted list of variables
    """
    sorted_list = sorted(var_list, key=fun)
    return sorted_list


def remove_vars(
    list_vars: list[PolyElement], accum_vars: list[PolyElement]
) -> list[PolyElement]:
    """Removes variables if they are already proposed in the quadratization or if they are of degree less than two

    Parameters
    ----------
    list_vars
        Set of variables to check
    accum_vars
        List with all variables proposed up to this point

    Returns
    -------
    list[tuple]
        the filtered list of variables
    """
    for i in range(len(list_vars)):
        tup_list = list(list_vars[i])
        new_tup = list(tup_list)
        if tup_list[0] in accum_vars or sum(tup_list[0].degrees()) <= 1:
            new_tup.remove(tup_list[0])
        if (
            tup_list[1] in accum_vars
            or sum(tup_list[1].degrees()) <= 1
            or tup_list[0] == tup_list[1]
        ):
            new_tup.remove(tup_list[1])
        list_vars[i] = tuple(new_tup)
    return list_vars


def get_pol_diff_order(pol: PolyElement) -> int:
    """Gets the derivative order of a polynomial

    Parameters
    ----------
    pol
        polynomial to check

    Returns
    -------
    int
        the derivative order of the polynomial (highest derivative order)
    """
    pattern = r"\w+_[0-9]*[A-Za-z]+[0-9]+$"

    derivs = [x for x in pol.ring.gens if pol.diff(x) != 0]
    order = 0
    for var in derivs:
        var_str = str(var)
        if bool(re.match(pattern, var_str)):
            var_ord = ""
            for char in reversed(var_str):
                if char.isalpha():
                    break
                else:
                    var_ord = char + var_ord
            order = max(order, int(var_ord))
    return order
