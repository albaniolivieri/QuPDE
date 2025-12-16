import sympy as sp
from sympy import Derivative as D
import time
import statistics
from qupde import quadratize

"""
The Gray-Scott equations are a system of reaction-diffusion equations that model the interaction of two generic chemical species:
    u_t = epsilon_1 * Delta(u) - u * v^2 + F(1 - u)
    v_t = epsilon_2 * Delta(v) + u * v^2 - (F + k) * v
References:
    P. Gray, S.K. Scott (1983). Autocatalytic reactions in the isothermal, continuous stirred tank reactor: Isolas and other forms of
    multistability, Chemical Engineering Science, Volume 38.
"""

t, x = sp.symbols("t x")
e_1, e_2, F, k = sp.symbols("epsilon_1 epsilon_2 F k", constant=True)
v = sp.Function("v")(t, x)
u = sp.Function("u")(t, x)

u_t = e_1 * D(u, x, 2) - u * v**2 + F * (1 - u)
v_t = e_2 * D(v, x, 2) + u * v**2 - (F + k) * v

# we run QuPDE for the Gray-Scott equations
if __name__ == "__main__":
    times = []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t), (v, v_t)], 3, search_alg="bnb")
        times.append(time.time() - ti)
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])

    quadratize([(u, u_t), (v, v_t)], 3, search_alg="bnb", printing="pprint")

    print("Average", avg)
    print("Standard deviation", std)
