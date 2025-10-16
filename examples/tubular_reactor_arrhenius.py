import sympy as sp
from sympy import Derivative as D
import time
import statistics
import sys
sys.path.append("..")
from qupde import quadratize

"""
The non-adiabatic tubular reactor model describes species concentration and temperature evolution in a single reaction:
    psi_t = (1/Pe) * u_ss - u_s - D * psi * f(theta),
    theta_t = (1/Pe) * theta_ss - theta_s - beta * (theta + theta_ref) + B * D * psi * f(theta),
where f(theta) = exp(psi-psi/theta)
References:
    Heinemann, R. F., & Poore, A. B. (1981). Multiplicity, stability, and oscillatory dynamics of the tubular reactor. 
    Chemical Engineering Science, 36(8), 1411–1419. https://doi.org/10.1016/0009-2509(81)80175-3
"""

t, s = sp.symbols("t s")
psi = sp.Function("psi")(t, s)
theta = sp.Function("theta")(t, s)
y = sp.Function("y")(t, s)
Pe = sp.symbols("Pe", constant=True)
B = sp.symbols("B", constant=True)
D_ct = sp.symbols("D", constant=True)
beta = sp.symbols("beta", constant=True)
theta_ref = sp.symbols("theta_ref", constant=True)
gamma = sp.symbols("gamma", constant=True)

psi_t = (
    (1 / Pe) * D(psi, s, 2)
    - D(psi, s)
    - D_ct * psi * y
)
theta_t = (
    (1 / Pe) * D(theta, s, 2)
    - D(theta, s)
    - beta * (theta - theta_ref)
    + B * D_ct * psi * y
)
y_t = gamma/theta**2 * y * theta_t

# we run QuPDE for the tubular reactor model
if __name__ == "__main__":
    times = []
    for i in range(5):
        ti = time.time()
        quadratize(
            [(psi, psi_t), (theta, theta_t), (y, y_t)],
            nvars_bound=7,
            max_der_order=0
        )
        times.append(time.time() - ti)

    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])
    print(time.time())
    quadratize(
        [(psi, psi_t), (theta, theta_t), (y, y_t)],
        nvars_bound=7,
        max_der_order=0,
        printing="pprint",
    )
    
    print("Average time", avg)
    print("Standard deviation", std)
    
    


