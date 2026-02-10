import sympy as sp
from sympy import Derivative as D
import statistics
import time
from qupde import quadratize

"""
The nonlinear Schrödinger equation is a nonlinear partial differential equation, applicable to classical and quantum mechanics:
    u_t = -1/2 * u_xx + kappa * u**3
References:
    Zakharov, V.E., Manakov, S.V. On the complete integrability of a nonlinear Schrödinger equation. 
    Theor Math Phys 19, 551–559 (1974). https://doi.org/10.1007/BF01035568
"""

t, x = sp.symbols("t x")
u = sp.Function("u")(t, x)

u_t = -0.5 * D(u, x, 2) + u**3

# we run QuPDE for the Dym equation
if __name__ == "__main__":
    times = []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t)], 2, search_alg="bnb")
        times.append(time.time() - ti)
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])

    quadratize([(u, u_t)], 2, search_alg="bnb", printing="pprint")

    print("Average time", avg)
    print("Standard deviation", std)
