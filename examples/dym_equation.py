import sympy as sp
from sympy import Derivative as D
import statistics
import sys
import time
sys.path.append("..")
from qupde.quadratize import quadratize

"""
The Harry Dym equation is an important dynamical equation that is integrable and finds applications 
in several physical systems. The Dym equation represents a system in which dispersion and nonlinearity 
are coupled together:
    u_t = u^3 * u_xxx
References:
    Kumar, S., Tripathi, M. P., & Singh, O. P. (2013). A fractional model of Harry Dym equation and its 
    approximate solution. Ain Shams Engineering Journal, 4(1), 111–115. https://doi.org/10.1016/j.asej.2012.07.001
"""

t, x = sp.symbols('t x')
u = sp.Function('u')(t,x)

u_t = u**3 * D(u, x, 3)

# we run QuPDE for the Dym equation
if __name__ == '__main__':
    times= []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t)], search_alg = 'bnb')
        times.append(time.time() - ti) 
    print(times)
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])
    
    print('Average time', avg)
    print('Standard deviation', std)

    print(quadratize([(u, u_t)], sort_fun="by_fun", printing = 'latex', max_der_order=7))
    
    