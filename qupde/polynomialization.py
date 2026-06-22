from sympy import symbols, Function, exp, sin, cos, Pow, Symbol, Integer
from sympy import Derivative as D
from .quadratization import quadratize

non_pol_funcs = [exp, sin, cos, Pow]

def polynomialize(func_eq, first_indep=symbols('t'), second_indep=symbols('x'), is_rat=False):
    count = 0
    j = 0
    new_vars = []
    func_syms = [first_indep, second_indep]
    while j in range(len(func_eq)):
        while True:
            new_var = get_non_pol_var(func_eq[j][1], is_rat=is_rat, func_syms=func_syms)
            if not new_var:
                break
            new_vars.append((new_var, Function(f'p_{count}')(first_indep, second_indep)))
            for i in range(len(func_eq)):
                func_eq[i] = (func_eq[i][0], func_eq[i][1].subs(new_vars))     
            new_eq = get_new_eq(new_var, func_eq, first_indep).subs(new_vars)
            func_eq.append((Function(f'p_{count}')(first_indep, second_indep), new_eq))
            count += 1
        j += 1
    return func_eq, new_vars
    
def get_non_pol_var(expr, is_rat, func_syms, new_var = None):
    args = expr.args
    if expr.func in non_pol_funcs:
        if expr.func == Pow:
            try: 
                if args[1] == int(args[1]):
                    if args[1]>0: 
                        return get_non_pol_var(args[0], is_rat, func_syms, new_var=new_var)
                    elif args[1]<0 and is_rat: 
                        for arg in args:
                            return get_non_pol_var(arg, is_rat, func_syms, new_var=new_var)
            except TypeError:
                pass
        for arg in args:
            if bool(arg.free_symbols & set(func_syms)):
                return get_non_pol_var(arg, is_rat, func_syms, new_var=expr)
            else: return new_var
    else: 
        if expr.func == Symbol or expr.func == Integer: return new_var
        else: 
            for arg in args:
                new_var = get_non_pol_var(arg, is_rat, func_syms, new_var=new_var)
                if new_var != None: return new_var
    return new_var

def get_new_eq(new_var, func_eq, first_indep):
    refac = [(D(func, first_indep), eq) for func, eq in func_eq]
    wt = D(new_var, first_indep).doit().subs(refac).doit()
    return wt

def polynomialize_and_quadratize(func_eq, diff_ord: None, first_indep=symbols('t'), second_indep=symbols('x'), nvars_bound=10):
    poly_sys, new_vars = polynomialize(func_eq, first_indep, second_indep, is_rat=True)
    print("poly_sys", poly_sys, new_vars)
    
    return quadratize(poly_sys, diff_ord=diff_ord, first_indep=symbols('t'), nvars_bound=nvars_bound)
    