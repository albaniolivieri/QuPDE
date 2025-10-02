
# QuPDE
QuPDE is a Python library that finds an optimal and monomial quadratic transformation (quadratization) for nonquadratic PDEs using Sympy. QuPDE handles one-dimensional PDEs that are polynomial or rational. 

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Running Tests](#running-tests)
- [Credits](#credits)

## Overview

A quadratization for a PDE is the set of auxiliary variables we introduce to rewrite the right-hand-side differential equations as quadratic. An optimal quadratization refers to the minimum number of new variables to achieve this transformation. QuPDE outputs this optimal set of new variables and gives the corresponding transformation of the differential equations.

## Installation
### Install using PyPI: 
1. With pip installed, run 
```console  
pip install qupde
```


### Install by cloning the repository from Github:
1. Run the command
```console  
git clone https://github.com/albaniolivieri/pde-quad.git`
```
2. Install requirements by running:

```console 
pip install -r requirements.txt
```
    

## Usage

For interactive usage examples, go to [Colab notebook](https://colab.research.google.com/drive/1qbMoZTL0SMJ5tdp8dHXULBjnxvWJjmo_?usp=sharing).


<!-- To find a quadratization for a PDE, we need to provide the algorithm with two parameters, the PDE and the number of differentiations with respect to the spatial variable to be performed on the auxiliary variables. -->

To find a quadratization for the PDE $$u_t = u_x^3u_{xxx}$$ (Dym equation), we first write the differential equation:

```python 
from sympy import symbols, Function, Derivative

t, x = symbols('t x')
u = Function('u')(t,x)

u_t = u**3 * Derivative(u, x, 3)
```
<!-- 
We choose how many times we want to differentiate the new variables to be introduced, in this case let us say three. 

```python 
n = 3
``` -->

Now we call the main function of the software *quadratize*. This function receives a list of tuples representing each undefined function with its corresponding differential equation within the PDE system. In our example: 

```python 
quadratize([(u, u_t)])
```

This function returns the optimal set of new variables (polynomial and rational) introduced to obtain a quadratic representation of the PDE, and the number of nodes visited. In this case, it returns 
 
```console 
([u**3, u*u_x1**2], [], 21)
```

Besides the PDE input, users can decide to include in the quadratic transformation the spatial derivatives of the auxiliary variables up to order k through the parameter *n_diff*. By default this value is set to 3. 
If we set this value in the previous example to 2, we obtain: 
```console 
Quadratization not found
```
indicating an unsuccesful search. This shows that the number of derivatives taken directly affects the algorithm's ability to find a quadratization. Therefore, incresing this parameter may help when encountered with an unsuccesful search.

In addition, we can print the new variables with their corresponding transformations by calling the same function but with the optional *printing* parameter set with the available printing options: 
- `'pprint'` for pretty printing (Sympy's functionality) 
- `'latex'` for printing the result in latex code. 
The command

```python 
quadratize([(u, u_t)], printing='pprint')
```

outputs
```console 
Quadratization:
      3
w₀ = u 
          2
w₁ = u⋅uₓ₁ 

Quadratic PDE:
w₀ₜ = w₀⋅w₀ₓ₃ - 2⋅w₀ₓ₁⋅w₀ₓ₂ + 10⋅w₀ₓ₁⋅w₁
                2⋅w₀ₓ₂⋅w₀ₓ₃                                       
w₁ₜ = w₀⋅w₁ₓ₃ - ─────────── + 4⋅w₀ₓ₂⋅w₁ₓ₁ + 4⋅w₀ₓ₃⋅w₁ - 24⋅w₁⋅w₁ₓ₁
                     3                                            
uₜ = uₓ₃⋅w₀
```


## Examples
We show a complete example using QuPDE's main function *quadratize* to find a quadratization for the Allen-Cahn equation: $$u_t = u_{xx} + u - u^3.$$ 

```python 
import sympy as sp
from sympy import Derivative as D
from qupde import *

t, x = sp.symbols('t x')
u = sp.Function('u')(t,x)

# define the PDE 
u_t = D(u, x, 2) + u - u**3 

# run QuPDE for the Allen-Cahn equation
quadratize([(u, u_t)], search_alg='bnb', printing='pprint')
```
This example outputs
```console 
Quadratization:
      2
w₀ = u 

Quadratic PDE:
         2                 2
w₀ₜ = 2⋅u  + 2⋅u⋅uₓ₂ - 2⋅w₀ 
uₜ = -u⋅w₀ + u + uₓ₂
```


## Running Tests

To run tests, execute the following command in the pde-quad directory
```bash
  python -m unittest tests/test_quadratization.py 
```

In this module, we provided tests for: 
- The branch-and-bound search framework
- The incremental nearest neighbor search framework 
- The module that handles quadratization for rational PDEs
- PDEs with symbolic coefficients

