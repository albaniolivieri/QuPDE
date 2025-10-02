import sympy as sp
from sympy import Derivative as D
import time
import statistics
import sys
sys.path.append("..")
from qupde.quadratize import quadratize

"""
The Cahn-Hilliard equation is a partial differential equation was proposed to model certain phenomena 
of phase separation (also known as spinodal decomposition) in binary alloys:
    u_t = Delta(u^3 - u) - epsilon * Delta^2(u).
References:
    Cahn, J. W., & Hilliard, J. E. (1958). Free Energy of a Nonuniform System. I. Interfacial Free Energy.
    The Journal of Chemical Physics, 28(2), 258–267. doi:10.1063/1.1744102 
"""

t, x = sp.symbols('t x')
u = sp.Function('u')(t,x)
epsilon = sp.symbols('epsilon', constant=True)

u_t = D(u**3 - u, (x, 2)).doit() - epsilon * D(u, x, 4)

# we run QuPDE for the Cahn-Hilliard equation
if __name__ == '__main__':   
    times= []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t)], 3, search_alg='bnb')
        times.append(time.time() - ti) 
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])

    quadratize([(u, u_t)], 3, search_alg='bnb', printing='pprint')
    
    print('Average time', avg)
    print('Standard deviation', std)
