from sympy import *
from sympy import Derivative as D
import time
import statistics
import sys
sys.path.append("..")
from qupde.quadratize import quadratize

#tests
t, x = symbols('t x')
u = Function('u')(t,x)

orders = []

# u_t = uxx/u^3
ut1 = D(u, x)**2/u


if __name__ == '__main__':
    print('Testing quadratization of rational PDEs')
    pdes = [ut1]
    for eq in pdes: 
        times = []
        for i in range(2):
            ti = time.time()
            quad, vfrac = quadratize([(u, eq)], 3, search_alg='bnb', max_der_order=5, printing='pprint')
            print(f'Quadratization for {eq}:', quad, vfrac)
            times.append(time.time() - ti)
        if quad:
            print(f'result for {eq}', f'new variables: {quad}', f'order: {len(quad)}', f'avg: {statistics.mean(times)}',
                f'std dev: {statistics.stdev(times)}')
        else: 
            print(f'no quadratization found for {eq}')
            