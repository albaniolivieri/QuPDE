import sympy as sp
from sympy import Derivative as D
import time
import statistics
from qupde import quadratize


"""
The Allen-Cahn equation is a partial differential equation that describes the phase separation of 
a binary alloy at a fixed temperature. In one space dimension:
    u_t = u_xx + u - u^3.
References:
    Allen, S. M., & Cahn, J. W. (1979). A microscopic theory for antiphase boundary motion and its 
    application to antiphase domain coarsening. Acta Metallurgica, 27(6), 1085–1095.
    https://doi.org/10.1016/0001-6160(79)90196-2
"""

t, x = sp.symbols('t x')
u = sp.Function('u')(t,x)

u_t = D(u, x, 2) + u - u**3

# we run QuPDE for the Allen-Cahn equation
if __name__ == '__main__':
    times= []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t)], diff_ord=2)
        times.append(time.time() - ti) 
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])
    
    quadratize([(u, u_t)], diff_ord=2, printing='pprint')

    print('Average time', avg)
    print('Standard deviation', std)
