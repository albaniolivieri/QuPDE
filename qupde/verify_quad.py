import sympy as sp
from sympy.polys.rings import PolyElement
from .utils import reduction
from .fraction_decomp import FractionDecomp


def is_quadratization(
    V: list[PolyElement],
    deriv: list[PolyElement],
    frac_decomp: FractionDecomp,
) -> tuple[bool, list[PolyElement]]:
    """Verifies if all variables in V are a quadratization for the system in deriv.
    It also prints the quadratization.

    Parameters
    ----------
    V
        List V with all variables/symbols of the system
    deriv
        List with the equations of the PDE system
    frac_decomp
        Fraction decomposition of the PDE system

    Returns
    -------
    tuple[bool, list[PolyElement]]
        a tuple with a boolean value and the quadratization found (if not found, returns the NS set)
    """
    V2 = list(set((m1[0] * m2[0], m1[1] * m2[1]) for m1 in V for m2 in V))

    if frac_decomp.groeb_rels:
        V2 = [(m[0], frac_decomp.try_reduce(m[1])) for m in V2]

    V2_poly, names = [], []
    for name, polyn in V2:
        names.append(name)
        V2_poly.append(polyn)
    quad, NS = [], []

    V2_red = reduce_set(V2)
    for name, pol in deriv:
        if pol not in V2_poly:
            result = is_linear_combination(V2_red, pol)
            if not result[0]:
                NS.append((name, result[1]))
            else:
                quad.append(sp.Eq(name, result[1]))
        else:
            quad.append(sp.Eq(name, names[V2_poly.index(pol)]))

    if NS != []:
        # for printing problematic polynomials, uncomment next line:
        # for i in range(len(NS)): pprint(f'NS for expr {NS[i][0]}: {NS[i][1]}')
        return (False, NS)
    return (True, quad)


def reduce_set(V2: list[PolyElement]) -> list[tuple[sp.Expr, PolyElement]]:
    """Reduces the V^2 set following the Gauss elimination method

    Parameters
    ----------
    V2
        List with V^2 set

    Returns
    -------
    list[tuple[sp.Expr, PolyElement]]
        a list with the reduced V^2 set
    """
    for i in range(len(V2)):
        for j in range(i):
            V2[i] = reduction(V2[i], V2[j])
        if V2[i][1] != 0:
            lead_coeff = V2[i][1].coeff(V2[i][1].leading_monom())
            V2[i] = (
                V2[i][0] / lead_coeff.as_expr(),
                V2[i][1] * (1 / lead_coeff),
                V2[i][1].leading_monom(),
            )
            for j in range(i):
                V2[j] = reduction(V2[j], V2[i])
    return [(a[0], a[1]) for a in V2]


def is_linear_combination(
    V2: list[PolyElement], der_pol: PolyElement
) -> tuple[bool, PolyElement]:
    """Checks if a certain polynomial is a linear combination of the set V^2

    Parameters
    ----------
    V2
        List with V^2 set
    der_pol
        Polynomial to check

    Returns
    -------
    tuple[bool, PolyElement]
        if der_pol is a linear combination of V^2, returns a tuple with the boolean True and the resulting polynomial
        from the algebraic operations. If that is not the case, returns a tuple with the boolean False and the residual
        polynomial.
    """
    der_tuple = (0, der_pol, der_pol.leading_monom())
    V2 = [(name, pol, pol.leading_monom()) for name, pol in V2]
    for i in range(len(V2)):
        der_tuple = reduction(der_tuple, V2[i])
        if der_tuple[1] == 0:
            return (True, -der_tuple[0])
    return (False, der_tuple[1])
