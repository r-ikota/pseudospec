import numpy as np
from numpy import testing
import sympy as sy
import pseudospec as ps
import expected_values_pde as evpde
from expected_values_pde import t,  zero, one
from pseudospec import SpecEQ
import tempfile, h5py, os


NW = 50
K = 5
dt = 1.0e-4
trange = np.linspace(0.0, 1.0, 21)
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

# def check_exact_sol(pde):
#         assert (pde.LHS - pde.f).simplify()\
#                 == zero

# def test_exact_sol():
#         pdes = [evpde.PDEsol1(K),
#                 evpde.PDEsol2(K)]

#         for pde in pdes:
#                 yield check_exact_sol, pde

class Heat(SpecEQ):
    def __init__(self, N, NC=1, pow=1, **par):
        SpecEQ.__init__(self, N, NC=NC, pow=pow, **par)
        self._paramNames = ()
        self._paramDefault = []

    def eq(self, u, t):
        '''
        du/dt = u_xx
        '''

        sc = self.sc

        return sc.sdiff2(u)

tmpdir = tempfile.gettempdir()


def check_pde(pde,expected_sol,i):
    try:
        # tmpfname = os.path.join(tmpdir, 'test.hdf5')
        tmpfname = 'test.hdf5'
        x = pde.get_x()
        _ic = pde.sc.phys2wave(expected_sol(0.0, x))
        with h5py.File(tmpfname, 'w') as fh:
            pde.mkInitDataSet(_ic,fh,dt)
            pde.evolve(fh, trange)
            observed_w = fh['u'][()]
        
        observed = pde.sc.wave2phys(observed_w[i])
        expected = expected_sol(trange[i],x)
        # comparison
        testing.assert_allclose(observed[i], expected, rtol=1.0e-4)
    finally:
        os.remove(tmpfname)

def test_pde():
    heat = Heat(NW,pow=1)
    expected_heat = evpde.PDEsol1(K)
    expected_heat_n = expected_heat.getNumSol()
    for i in [6,12,18]:
        yield check_pde, heat, expected_heat_n,i

