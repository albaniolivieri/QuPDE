from sympy.polys.rings import PolyElement
from .utils import get_pol_diff_order

def by_degree_order(vars_tup: tuple[PolyElement, PolyElement]) -> tuple[int, int]:
    """Function to sort the variables by the sum of their degrees and the
    maximum order of differentiation

    Parameters
    ----------
    vars_tup
        Tuple with the variables to be sorted

    Returns
    -------
    tuple[int, int]
        tuple with the sorting criteria given by the sum of the degrees and
        the maximum order of differentiation of the variables
    """
    if len(vars_tup) > 1:
        deg, order = (
            max([sum(vars_tup[0].degrees()), sum(vars_tup[1].degrees())]),
            max([get_pol_diff_order(vars_tup[0]), get_pol_diff_order(vars_tup[1])]),
        )
    else:
        deg, order = (sum(vars_tup[0].degrees()), get_pol_diff_order(vars_tup[0]))
    return (deg, order)


def by_order_degree(vars_tup: tuple[PolyElement, PolyElement]) -> tuple[int, int]:
    """Function to sort the variables by the maximum order of differentiation and the
    sum of their degrees

    Parameters
    ----------
    vars_tup
        Tuple with the variables to be sorted

    Returns
    -------
    tuple
        tuple with sorting criteria given by the maximum order of differentiation and
        the sum of the degrees of the variables
    """
    deg, order = 0, 0
    if len(vars_tup) > 1:
        order, deg = (
            max([get_pol_diff_order(vars_tup[0]), get_pol_diff_order(vars_tup[1])]),
            max([sum(vars_tup[0].degrees()), sum(vars_tup[1].degrees())]),
        )
    else:
        order, deg = (get_pol_diff_order(vars_tup[0]), sum(vars_tup[0].degrees()))
    return (order, deg)


def by_fun(vars_tup: tuple[PolyElement, PolyElement]) -> int:
    """Function to sort the variables by the function: degree + 2 * order

    Parameters
    ----------
    vars_tup: tuple
        Tuple with the variables to be sorted

    Returns
    -------
    int
        sorting criteria given by the value of the function degree + 2 * order
    """
    if len(vars_tup) > 1:
        return max([sum(vars_tup[0].degrees()), sum(vars_tup[1].degrees())]) + 2 * max([get_pol_diff_order(vars_tup[0]), get_pol_diff_order(vars_tup[1])])
    else:
        return sum(vars_tup[0].degrees()) + 2 * get_pol_diff_order(vars_tup[0])


def by_fun2(vars_tup: tuple[PolyElement, PolyElement]) -> int:
    """
    Function to sort the variables by the function: degree + 4 * order

    Parameters
    ----------
    vars_tup
        Tuple with the variables to be sorted

    Returns
    -------
    int
        sorting criteria given by the value of the function degree + 4 * order
    """
    if len(vars_tup) > 1:
        return sum([sum(vars_tup[0].degrees()), sum(vars_tup[1].degrees())]) + 4 * sum([get_pol_diff_order(vars_tup[0]), get_pol_diff_order(vars_tup[1])])
    else:
        return sum(vars_tup[0].degrees()) + 4 * get_pol_diff_order(vars_tup[0])
