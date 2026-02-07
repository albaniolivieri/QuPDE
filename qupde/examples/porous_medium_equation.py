import sympy as sp
from sympy import Derivative as D
import time
import statistics
from qupde import quadratize

"""
The porous medium equation is a nonlinear diffusion equation that has applications in biology, notably in models of animal and
insect dispersal, and some are in plasma physics. Another, as indicated by the name, is in the study of the ow of a gas in a 
porous medium. The porous medium equation is given by:
    u_t = \Delta(u^m), m>1.
References: 
    Vázquez, J. L. (2007). The porous medium equation: Mathematical theory. Clarendon Press.
    https://books.google.com/books?id=kZQUDAAAQBAJ
"""
t, x = sp.symbols("t x")
u = sp.Function("u")(t, x)

m = 7
u_t = D(u**m, x, 2).doit()

# we run QuPDE for the porous medium equation
if __name__ == "__main__":
    times = []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t)], 3, search_alg="bnb")
        times.append(time.time() - ti)
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])

    quadratize([(u, u_t)], 3, search_alg="bnb", printing="pprint")

    print("Average time", avg)
    print("Standard deviation", std)
