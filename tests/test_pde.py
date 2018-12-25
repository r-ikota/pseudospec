import numpy as np
from numpy import testing
import sympy as sy
import pseudospec as ps
import expected_values_pde as evpde
from expected_values_pde import t, x, zero, one

NW = 15
K = 5
xsym = sy.var('x', real=True)




def sym2num(sc,f):
        xnum = sc.get_x()
        _fnum = sy.lambdify(xsym, f, 'numpy')(xnum)
        _fnum = np.real(_fnum)
        return _fnum

def coeff2wave(sc,coeff):
        N2 = sc.N2
        J = sc.J
        w = sc.zeros()
        w[:N2] = np.asarray(coeff)*J
        return w

def check_exact_sol(pde):
        assert (pde.LHS - pde.f).simplify()\
                == zero

def test_exact_sol():
        pdes = [evpde.PDEsol1(K),
                evpde.PDEsol2(K)]

        for pde in pdes:
                yield check_exact_sol, pde

