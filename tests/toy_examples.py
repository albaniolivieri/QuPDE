from sympy import *
from sympy import Derivative as D
import time
import statistics
import sys

sys.path.append("..")
from qupde.quadratization import quadratize

# tests
t, x = symbols("t x")
u = Function("u")(t, x)

orders = []

# u_t = u**2 * ux
ut1 = u**2 * D(u, x)

# u_t = u**2 * ux + u**3
ut2 = u**2 * D(u, x) + u**3

# u_t = ux**2 * u
ut3 = u * D(u, x) ** 2

# u_t = ux**3
ut4 = D(u, x) ** 3

# u_t = u**2 * uxx
ut5 = u**2 * D(u, x, 2)

# u_t = u * (ux**2 + u * uxx)
ut6 = u * (D(u, x) ** 2 + u * D(u, x, 2))

# u_t = u * (2 ux * uxx + u * uxxx + 1)
ut7 = u * (3 * D(u, x) * D(u, x, 2) + u * D(u, x, 3) + 1)

# u_t = ux**3 + u**3
ut8 = D(u, x) ** 3 + u**3

# u_t = ux**4
ut9 = D(u, x) ** 4

# u_t = ux**3 + u**5
ut10 = D(u, x) ** 3 + u**5

# u_t = ux + u^3
ut11 = D(u, x) + u**3


if __name__ == "__main__":
    pdes = [ut1, ut2, ut3, ut4, ut5, ut6, ut7, ut8, ut9, ut10, ut11]
    for eq in pdes:
        times = []
        for i in range(2):
            ti = time.time()
            quad = quadratize(
                [(u, eq)],
                2,
                search_alg="bnb",
                max_der_order=10,
                printing="pprint",
                nvars_bound=11,
            )[0]
            times.append(time.time() - ti)
        if quad:
            print(
                f"result for {eq}",
                f"new variables: {quad}",
                f"order: {len(quad)}",
                f"avg: {statistics.mean(times)}",
                f"std dev: {statistics.stdev(times)}",
            )
