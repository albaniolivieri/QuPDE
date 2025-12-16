import sympy as sp
from sympy import Derivative as D
import time
import statistics
from qupde import quadratize

"""
The 1D nonlinear heat equation is one the most widely studied models, and it presents a rich mathematical 
structure for studying “blow-up” functions:
    u_t = u_xx + u^p, p>1
References: 
    Bandle, C., & Brunner, H. (1998). Blowup in diffusion equations: A survey. Journal of Computational and 
    Applied Mathematics, 97(1–2), 3–22. https://doi.org/10.1016/S0377-0427(98)00100-9
"""

t, x = sp.symbols("t x")
u = sp.Function("u")(t, x)

p = 6
u_t = D(u, x, 2) + u**p

# we run QuPDE for the nonlinear heat equation
if __name__ == "__main__":
    times = []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t)], diff_ord=2, max_der_order=0)
        times.append(time.time() - ti)
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])

    quadratize([(u, u_t)], diff_ord=2, max_der_order=0, printing="pprint")

    print("Average time", avg)
    print("Standard deviation", std)
