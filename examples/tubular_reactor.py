import sympy as sp
from sympy import Derivative as D
import time
import statistics
import sys
sys.path.append("..")
from qupde.quadratize import quadratize
from qupde.mon_heuristics import *

"""
The non-adiabatic tubular reactor model describes species concentration and temperature evolution in a single reaction:
    psi_t = (1/Pe) * u_ss - u_s - D * psi * f(theta),
    theta_t = (1/Pe) * theta_ss - theta_s - beta * (theta + theta_ref) + B * D * psi * f(theta),
where f(theta) = c_0 + c_1 * theta + c_2 * theta^2 + c_3 * theta^3 + ...
References:
    Heinemann, R. F., & Poore, A. B. (1981). Multiplicity, stability, and oscillatory dynamics of the tubular reactor. 
    Chemical Engineering Science, 36(8), 1411–1419. https://doi.org/10.1016/0009-2509(81)80175-3
"""

t, s = sp.symbols("t s")
psi = sp.Function("psi")(t, s)
theta = sp.Function("theta")(t, s)
Pe = sp.symbols("Pe", constant=True)
B = sp.symbols("B", constant=True)
D_ct = sp.symbols("D", constant=True)
beta = sp.symbols("beta", constant=True)
theta_ref = sp.symbols("theta_ref", constant=True)
c_0, c_1, c_2, c_3, c_4, c_5, c_6 = sp.symbols("c_0 c_1 c_2 c_3 c_4 c_5 c_6", constant=True)

f = (c_3 * theta**3 + c_2 * theta**2 + c_1 * theta + c_0)

psi_t = (
    (1 / Pe) * D(psi, s, 2)
    - D(psi, s)
    - D_ct * psi * f
)
theta_t = (
    (1 / Pe) * D(theta, s, 2)
    - D(theta, s)
    - beta * (theta - theta_ref)
    + B * D_ct * psi * f
)

# we run QuPDE for the tubular reactor model
if __name__ == "__main__":
    times = []
    for i in range(10):
        ti = time.time()
        quadratize(
            [(psi, psi_t), (theta, theta_t)],
            diff_ord=2,
            nvars_bound=7,
            max_der_order=1,
            search_alg="bnb",
        )
        times.append(time.time() - ti)
    avg = statistics.mean(times)
    std = statistics.stdev(times)

    print("Average time", avg)
    print("Standard deviation", std)

    print(quadratize(
            [(psi, psi_t), (theta, theta_t)],
            diff_ord=2,
            nvars_bound=8,
            max_der_order=1,
            search_alg="bnb",
            printing="latex",
        ))


