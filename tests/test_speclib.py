import numpy as np
from numpy import testing
import sympy as sy
import pseudospec as ps
import expected_values as ev

NW = 15
xsym = sy.var('x', real=True)

def test_NtoJ():
    pow = 3
    J = ps.NWtoJ(NW, pow=3)
    assert type(J) == int and J > pow*NW + 1

def check_each_funcs(prod,i):
    N2 = prod.N2
    J = prod.J
    xnum = prod.get_x()
    w = prod.zeros()
    w[:N2] = np.asarray(ev.fft_coeffs[i])*J
    f = ev.fs[i]
    expected = sy.lambdify(xsym, f, 'numpy')(xnum)
    expected = np.real(expected)
    observed = prod.wave2phys(w)

    testing.assert_allclose(observed, expected)

def test_each_funcs():
    pow = 1
    prod1 = ps.SpecCalc(NW,pow)
    
    for i in range(len(ev.fs)):
        yield check_each_funcs, prod1, i