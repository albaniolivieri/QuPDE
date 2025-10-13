from sympy import *
from sympy import Derivative as D
import time
import statistics
import sys
import math 
sys.path.append("..")
from qupde.polynomialization import polynomialize 
from qupde import quadratize as quad_bb

t, zetav = symbols('t zeta')
alpha = Function('alpha')(t, zetav)
v = Function('v')(t, zetav)
A = symbols('A', constant = True)
gammac = pi/2

alpha_t = D(v, zetav)
v_t = gammac * A * alpha**(gammac-1) * D(alpha, zetav)

new_pde, new_vars = (polynomialize([(alpha, alpha_t), (v, v_t)], second_indep=zetav))
print('New PDE\n', new_pde, '\nNew variables\n', new_vars)

# print(quad_bb(new_pde, 5))