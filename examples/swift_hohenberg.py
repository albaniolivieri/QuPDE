import sympy as sp
from sympy import Derivative as D
import time
import statistics
from qupde.quadratization import quadratize

"""
The Swift-Hohenberg equation is a partial differential equation for a scalar field which has been widely
used as a model for the study of various issues in pattern formation:
    u_t = r * u - (u_xx + q_c^2)^2*u + v * u^2 - g * u^3.
References:
    Burke, J., & Knobloch, E. (2006). Localized states in the generalized Swift-Hohenberg equation. Physical Review E, 73(5).
"""
t, x = sp.symbols("t x")
u = sp.Function("u")(t, x)
r, qc, v, g = sp.symbols("r qc v g", constant=True)

u_t = (
    r * u
    - D(u, (x, 2)) ** 2 * u
    - 2 * qc**2 * D(u, (x, 2)) ** 2 * u
    - qc**4 * u
    + v * u**2
    - g * u**3
)

# we run QuPDE for the Swift-Hohenberg equation
if __name__ == "__main__":
    times = []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t)], diff_ord=4, search_alg="bnb", max_der_order=4)
        times.append(time.time() - ti)
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])

    quadratize(
        [(u, u_t)], diff_ord=4, search_alg="bnb", max_der_order=4, printing="latex"
    )

    print("Average time", avg)
    print("Standard deviation", std)
