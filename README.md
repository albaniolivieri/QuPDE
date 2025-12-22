# QuPDE

QuPDE is a Python library that finds a quadratic transformation (quadratization) for nonquadratic PDEs built using Sympy objects. QuPDE handles spatially one-dimensional PDEs that are polynomial or rational.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Running Tests](#running-tests)

## Overview

A quadratization for a PDE is the set of auxiliary variables we introduce to rewrite the right-hand side differential equations as quadratic. QuPDE outputs set of a low number of new variables and gives the corresponding transformation of the differential equations.

## Installation

### Install using PyPI:

1. With pip installed, run

```console
pip install qupde
```

### Install by cloning the repository from Github:

1. Run the command

```console
git clone https://github.com/albaniolivieri/pde-quad.git
cd pde-quad
```

2. (Optional) Install [`uv`](https://docs.astral.sh/uv/) if you don't have it:

```console
# macOS / Linux (installer script)
curl -LsSf https://astral.sh/uv/install.sh | sh

# or with pip
pip install uv
```

3. Using `uv` (recommended):

```console
uv sync
```

This will create (or update) a `.venv` virtual environment and install `qupde` and its dependencies. You can then run commands with `uv run`, as shown in the **Running Tests** section.

4. Without `uv` (alternative):

```console
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

For interactive usage examples, go to qupde_usage_examples.ipynb file or [Colab notebook](https://colab.research.google.com/drive/1C5gPkOkO14fQwsmXRx91QRsaEFt8nwWl?usp=sharing).

To find a quadratization for the PDE $$u_t = a u^2 * u_x - u_{xxx}$$ (Korteweg-de Vries equation), we first write the differential equation:

```python
from sympy import symbols, Function, Derivative
from qupde import quadratize

t, x = symbols('t x')
u = Function('u')(t,x)
a = symbols('a', constant=True)

u_t =  a * u**2 * Derivative(u, x) - Derivative(u, x, 3)
```

Now we call the main function of the software _quadratize_. This function receives a list of tuples representing each undefined function with its corresponding differential equation within the PDE system. In our example:

```python
new_pde = quadratize([(u, u_t)])
```

_quadratize_ returns an object with the PDE quadratic transformation that stores the new PDE and the auxiliary variables introduced (polynomial and rational). We can get the auxiliary variables and the quadratic transformation by running

```python
new_pde.get_aux_vars()
```

```console
([u**2], [])
```

```python
new_pde.get_quad_sys()
```

```console
[Eq(w_0t, a*w_0*w_0x1 + 6*u_x1*u_x2 - w_0x3), Eq(u_t, a*u_x1*w_0 - u_x3)]
```

Besides the PDE input, users can provide a regularity restriction for the quadratic transformation through the parameter _diff_ord_. This number determines the differential order of the quadratization: the maximum spatial-derivative order of the PDE's original variables allowed. By default, this value is set to the maximum order of derivatives found for the unknown functions in the PDE.
If we set this value in the previous example to 0

```python
quadratize([(u, u_t)], diff_ord=0)
```

we obtain:

```console
Quadratization not found
```

which indicates an unsuccesful search. This shows how the order of derivatives allowed directly affects the algorithm's ability to find a quadratization. Therefore, incresing this parameter may help when encountering an unsuccesful search.

Additionally, we can print the new variables and their corresponding transformations in a more readable format by calling the same function with the optional printing parameter set to one of the available printing options.

- `'pprint'` for pretty printing (Sympy's functionality)
- `'latex'` for printing the result in latex code.
  The command

```python
quadratize([(u, u_t)], printing='pprint')
```

outputs

```console
Quadratization:
      2
w₀ = u

Quadratic PDE:
w₀ₜ = a⋅w₀⋅w₀ₓ₁ - 2⋅u⋅uₓ₃
uₜ = a⋅uₓ₁⋅w₀ - uₓ₃
```

## Command-line interface

Installing QuPDE now provides a small Typer-based CLI:

```bash
qupde examples
qupde run --example allen-cahn --printing pprint
```

Use `--diff-ord`, `--search-alg`, or `--max-der-order` to override defaults for an example, and `--printing latex` to emit LaTeX code.

You can also pass your own PDEs without touching Python code. Provide the independent variables, functions, and one or more equations:

```bash
# SymPy syntax
qupde run \
  --vars "t,x" \
  --funcs "u" \
  --eq "Derivative(u(t,x), t) = Derivative(u(t,x),(x,2)) + u(t,x) - u(t,x)**3" \
  --diff-ord 2 --max-der-order 2 --printing pprint

# Mathematica-style syntax
qupde run \
  --format mathematica \
  --vars "t,x" \
  --funcs "u" \
  --eq "D[u[t,x], t] == D[u[t,x], {x, 2}] + u[t,x] - u[t,x]^3" \
  --printing none
```

Notes:
- Declare exactly two independent variables via `--vars` (first is treated as time).
- Each `--eq` must have a left-hand side derivative in the first variable (e.g., `Derivative(u(t,x), t)` or `D[u[t,x], t]`).
- Any symbol not listed in `--vars`/`--funcs` is treated as a constant.

## Examples

We show a complete example using QuPDE's main function _quadratize_ to find a quadratization for the Allen-Cahn equation: $$u_t = u_{xx} + u - u^3.$$

```python
import sympy as sp
from sympy import Derivative as D
from qupde import *

t, x = sp.symbols('t x')
u = sp.Function('u')(t,x)

# define the PDE
u_t = D(u, x, 2) + u - u**3

# run QuPDE for the Allen-Cahn equation
quadratize([(u, u_t)], printing='pprint')
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

The table below shows some of the PDE examples for which QuPDE has found quadratizations.

| PDE                                  | Quadratization variables                               |
| ------------------------------------ | ------------------------------------------------------ |
| Solar wind model                     | $1/u$                                                  |
| Modified KdV                         | $u^2$                                                  |
| Allen-Cahn equation                  | $u^2$                                                  |
| Schl{\"o}gl model                    | $u^2$                                                  |
| Euler equations                      | $1/\\rho$                                              |
| FHN system                           | $v^2$                                                  |
| Brusselator system                   | $u^2$, $uv$                                            |
| Dym equation                         | $u^3$, $u_{x}^2u$                                      |
| Schnakenberg equations               | $uv$, $u^2$                                            |
| Nonlinear heat equation              | $u^3$, $u_xu$, $u^5$                                   |
| Polynomial tubular reactor (deg = 3) | $uv$, $v^2$, $v^2u$, $v^3$                             |
| Arrhenius-type tubular reactor       | $1/v$, $1/v^2$, $uv$, $uy/v$, $uy/v^2$, $y/v$, $y/v^2$ |

## Running Tests

To run tests, execute one of the following commands in the pde-quad directory:

Using `uv`:

```bash
uv run pytest
```

Using `python` directly:

```bash
python -m pytest
```

### Linting and formatting (Ruff)

To check code style and automatically format the code (if you have installed the development dependencies, e.g. with `uv sync --group dev`), run:

```bash
uv run ruff check .
uv run ruff format .
```

In this module, we provided tests for:

- The branch-and-bound search framework
- The incremental nearest neighbor search framework
- The module that handles quadratization for rational PDEs
- PDEs with symbolic coefficients
