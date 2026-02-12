import math
from typing import Optional, Callable
from queue import PriorityQueue
from collections import deque
from itertools import chain, combinations
from sympy.polys.rings import PolyElement
from .utils import get_pol_diff_order
from .pde_sys import PDESys


def pruning_rule_nvars(nvars: int, global_nvars: int) -> bool:
    """Pruning rule based on the number of variables in the quadratization found.

    Parameters
    ----------
    nvars
        The number of variables in the quadratization found
    global_nvars
        The minimum number of variables allowed

    Returns
    -------
    bool
        True if the number of variables of the quadratization found is greater
        than the global, False otherwise
    """
    if nvars >= global_nvars:
        return True
    return False


def pruning_rule_order(
    new_vars: list[PolyElement], max_order: int, pde_sys: PDESys
) -> bool:
    """Pruning rule based on the maximum order of derivatives allowed.

    Parameters
    ----------
    new_vars
        List of proposed new variables
    max_order
        The maximum order allowed
    pde_sys
        PDE system

    Returns
    -------
    bool
        True if the maximum order of the derivatives in the new vars proposed is greater
        than the maximum order allowed or the differential order of the quadratization, False otherwise
    """
    for var in new_vars:
        var_order = get_pol_diff_order(var)
        if (var_order > max_order) or (
            (pde_sys.get_diff_quad_order() - (var_order + pde_sys.get_pde_order())) < 0
        ):
            return True
    return False


def shrink_quad(quad_vars: list[PolyElement], poly_syst: PDESys) -> list[PolyElement]:
    """Checks if the quadratization can be shrunk to a smaller set of variables.

    Parameters
    ----------
    quad_vars
        List of variables in the quadratization
    poly_syst
        The polynomial system

    Returns
    -------
    list[PolyElement]
        a list with a quadratization of an equal or lesser order than the original
    """
    final_vars = quad_vars
    subsets = chain.from_iterable(
        combinations(quad_vars, r) for r in range(1, len(quad_vars))
    )
    for var_group in subsets:
        poly_syst.set_new_vars(var_group)
        res, _ = poly_syst.try_make_quadratic()
        if res:
            return list(var_group)
    return final_vars


def bnb(
    new_vars: list[PolyElement],
    best_nvars: int,
    poly_syst: PDESys,
    sort_fun: Callable,
    max_der_order: int,
) -> tuple[list[PolyElement], int, int]:
    """Branch and bound algorithm to find the best quadratization of a polynomial system.

    Parameters
    ----------
    new_vars
        List of proposed new variables
    best_nvars
        The minimum number of variables found in a quadratization
    poly_syst
        The polynomial system to quadratize
    sort_fun
        The function to sort the proposed new variables
    max_der_order
        The maximum order of derivatives allowed in the quadratic transformation

    Returns
    -------
    tuple[list[PolyElement], int, int]
        a tuple with the best quadratization found, the number of variables in the
        quadratization and the total number of traversed nodes
    """
    if pruning_rule_nvars(len(new_vars), best_nvars):
        return None, math.inf, 1

    if max_der_order is None:
        if pruning_rule_order(new_vars, poly_syst.get_diff_quad_order(), poly_syst):
            return None, math.inf, 1
    else:
        if pruning_rule_order(new_vars, max_der_order, poly_syst):
            return None, math.inf, 1

    poly_syst.set_new_vars(new_vars)
    result_quad = poly_syst.try_make_quadratic()

    if result_quad[0]:
        shrinked_quad = shrink_quad(new_vars, poly_syst)
        return shrinked_quad, len(shrinked_quad), 1
    else:
        if len(new_vars) >= best_nvars - 1:
            return None, math.inf, 1

    min_nvars = best_nvars
    best_quad_vars = None
    traversed_total = 1
    prop_vars = poly_syst.prop_new_vars(sort_fun)

    for p_vars in prop_vars:
        quad_vars, nvars, traversed = bnb(
            new_vars + list(p_vars), min_nvars, poly_syst, sort_fun, max_der_order
        )
        traversed_total += traversed
        if nvars < min_nvars:
            min_nvars = nvars
            best_quad_vars = quad_vars

    return best_quad_vars, min_nvars, traversed_total


def nearest_neighbor(
    poly_syst: PDESys, sort_fun: Callable, new_vars: Optional[list[PolyElement]] = []
) -> tuple[list[PolyElement], int]:
    """Incremental nearest neighbor algorithm to find the best quadratization of a polynomial system.

    Parameters
    ----------
    poly_syst
        The polynomial system to quadratize
    sort_fun
        The function to sort the proposed new variables
    new_vars : optional
        List of proposed new variables

    Returns
    -------
    tuple[list[PolyElement], int]
        a tuple with the best quadratization found and the total number of traversed nodes
    """
    pq = PriorityQueue()

    pq.put((len(new_vars), 0, new_vars))
    # FIFO queue to explore subproblems in a lazy approach
    NS_queue = deque()
    node_count, count = 0, 0
    quad_temp = None

    while not pq.empty():
        new_vars = pq.get()[2]

        if quad_temp:
            if len(new_vars) >= len(quad_temp):
                continue

        poly_syst.set_new_vars(new_vars)
        result_quad = poly_syst.try_make_quadratic()
        node_count += 1

        if result_quad[0]:
            shrinked_quad = shrink_quad(new_vars, poly_syst)
            if quad_temp:
                if len(shrinked_quad) < len(quad_temp):
                    quad_temp = shrinked_quad
            else:
                quad_temp = shrinked_quad
            # we check if the subproblems in the FIFO queue are worth exploring
            # after finding a quadratization
            while len(NS_queue) > 0:
                new_vars_ns, NS = NS_queue.popleft()
                if len(new_vars_ns) + 1 < len(quad_temp):
                    poly_syst.set_new_vars(new_vars_ns)
                    poly_syst.set_NS_list(NS)
                    prop_vars = poly_syst.prop_new_vars(sort_fun)
                    for p_vars in prop_vars:
                        if len(new_vars_ns + list(p_vars)) < len(quad_temp):
                            pq.put(
                                (
                                    len(new_vars_ns + list(p_vars)),
                                    count,
                                    new_vars_ns + list(p_vars),
                                )
                            )
                            count += 1
        else:
            if not quad_temp:
                NS_queue.append((new_vars, result_quad[1]))
                if pq.qsize() <= 1:
                    new_vars, NS = NS_queue.popleft()
                    poly_syst.set_new_vars(new_vars)
                    poly_syst.set_NS_list(NS)
                    prop_vars = poly_syst.prop_new_vars(sort_fun)
                    for p_vars in prop_vars:
                        pq.put(
                            (
                                len(new_vars + list(p_vars)),
                                count,
                                new_vars + list(p_vars),
                            )
                        )
                        count += 1
    return quad_temp, node_count
