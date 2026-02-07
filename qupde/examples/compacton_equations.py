import sympy as sp
from sympy import Derivative as D
import time
import statistics
from qupde import quadratize

"""
The compacton equation is generalization of the KdV equation in which the dispersion too is nonlinear. The solutions of this equation
are compactons, which are solitary waves with compact support. The compacton equation (K(m,n)) is given by:
    u_t = -(u^m)_x - (u^n)_x, m>0 and 1<n<=3.
References: 
    Rosenau, P., & Hyman, J. (1993). Compactons: Solitons with finite wavelength. Physical Review Letters, 70(4), 564–567. 
    https://doi.org/10.1103/PhysRevLett.70.564
"""
t, x = sp.symbols("t x")
u = sp.Function("u")(t, x)

m = 3
n = 3
u_t = -D(u**m, x).doit() - D(u**n, x).doit()

# we run QuPDE for the compacton equation
if __name__ == "__main__":
    times = []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t)], 1, search_alg="bnb")
        times.append(time.time() - ti)
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])

    quadratize([(u, u_t)], 1, search_alg="bnb", printing="pprint")

    print("Average time", avg)
    print("Standard deviation", std)
