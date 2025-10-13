import sympy as sp
from sympy import Derivative as D
import sys
import time
import statistics
sys.path.append("..")
from qupde import quadratize

"""
The Korteweg-de Vries (KDV) equation is a generic model for the study of weakly nonlinear long waves, 
incorporating leading nonlinearity and dispersion. Also, it describes surface waves of long 
wavelength and small amplitude in shallow water: 
    u_t = a * u^2 * u_x - u_xxx
References:
    Wazwaz, A.-M. (2008). Chapter 9: The KdV equation. In Handbook of Differential Equations: Evolutionary 
    Equations (pp. 485–568). Elsevier. https://doi.org/10.1016/s1874-5717(08)00009-1
"""

t, x = sp.symbols('t x')
u = sp.Function('u')(t,x)
a = sp.symbols('a', constant=True)

u_t = - D(u, x, 3) + a * u**2 * D(u, x)
 
# we run QuPDE for the KDV equation
if __name__ == '__main__':
    times= []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t)], 2, search_alg = 'bnb')
        times.append(time.time() - ti) 
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])
    
    quadratize([(u, u_t)], 2, search_alg = 'bnb', printing = 'pprint')

    print('Average time', avg)
    print('Standard deviation', std)