from sympy import *
from sympy import Derivative as D
import time
import statistics
import sys
from qupde.quadratization import quadratize, check_quadratization

t, x = symbols('t x')
z = symbols('z')
u = Function('u')(t,x)
v = Function('v')(z,x)
u_x1, u_x2, u_x4, u_x3= symbols('u_x1 u_x2 u_x4 u_x3')

orders = []

# u_t = u**2 * ux 
# ut1 = u**2 * D(u, x) + u**3

# ut2 = u - Derivative(u, x)*u - 2 * Derivative(u, (x, 3))**2 * u - u + u**2 - u**3 
# print('PDE:', str(ut1))
# quadratize([(u, ut1)], max_der_order=3, printing = 'pprint')

# ut= Derivative(v, (x, 2))**2
# vt = u**2* Derivative(u, (x, 1))


# ut = v*D(v, x,1)* D(v, x, 3)
# ut = 1/((D(u, x)+1)*(u+2))
# ut = D(u, x, 2)*u**2

z=symbols('z')
y=symbols('y')
v=Function('v')(z,y)

vz = v**2 * D(v, y) - D(v, y, 3)

quad_pde =quadratize([(v, vz)], first_indep=z, printing='pprint')


# quad_pde = quadratize([(v, ut)], diff_ord=3, first_indep = z, printing = 'pprint')

# quadratize([(u, D(v, x, 2)**2), (v, D(u, x) * u**2)], max_der_order=5, printing='pprint')

# quad_pde = quadratize([(u, ut)], 2, search_alg='bnb', printing="latex")
print(quad_pde)
# w1 = quad[0].ring(symbols("u")*u_x1)
# w2 = quad[0].ring(u_x1**2)
# w3 = quad[0].ring(symbols("u")*u_x3)
# print(check_quadratization([(u, ut)], [w1, w2, w3], 3))
