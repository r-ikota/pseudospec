import numpy as np
from numpy import testing
import sympy as sy
import pseudospec as ps
import tests.expected_values_pde as evpde
from tests.expected_values_pde import t,  zero, one
from pseudospec import SpecEQ
import tempfile, h5py, os


NW = 50
K = 5
maxdt = 5.0e-3
trange = np.linspace(0.0, 1.0, 21)
xsym = sy.var('x', real=True)

expected_sols = [evpde.PDEsol1(K),
                evpde.PDEsol2(K),
                evpde.PDEsol3()]

expected_sols_n = []
for _ex_sol in expected_sols:
    expected_sols_n.append(_ex_sol.getNumSol())


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

class Heat(SpecEQ):
    def __init__(self, N, NC=1, pow=1, **par):
        SpecEQ.__init__(self, N, NC=NC, pow=pow, **par)
        self._paramNames = ()
        self._paramDefault = []

    def eq(self, t, u):
        '''
        du/dt = u_xx
        '''

        sc = self.sc

        return sc.sdiff2(u)

class HeatNH(SpecEQ):
    def __init__(self, N, NC=1, pow=1, **par):
        SpecEQ.__init__(self, N, NC=NC, pow=pow,**par)
        self._paramNames = ()
        self._paramDefault = []

        _exsol = expected_sols[1]
        self._fp = _exsol.getNumF()
        self.x = self.sc.get_x()


    def eq(self, t, u):
        '''
        du/dt = u_xx + f
        '''

        sc = self.sc
        x = self.x
        f = sc.phys2wave(self._fp(t,x))

        return sc.sdiff2(u) + f

class AdvDiff(SpecEQ):
    def __init__(self, N, NC=1, pow=2, **par):
        SpecEQ.__init__(self,N,NC=NC,pow=pow, **par)
        self._paramNames = ()
        self._paramDefault = []

        _exsol = expected_sols[2]
        self._fp = _exsol.getNumF()
        self._a = _exsol.getNumA()
        self.x = self.sc.get_x()

    def eq(self,t,u):
        '''
        du/dt = (u_x - a*u)_x + f
        '''

        sc = self.sc
        x = self.x
        f = sc.phys2wave(self._fp(t,x))
        a = sc.phys2wave(self._a(t,x))

        return sc.sdiff1(sc.sdiff1(u) - sc.mult2(a,u)) + f


tmpdir = tempfile.gettempdir()


def check_pde(pde,expected_sol):
    try:
        tmpfname = os.path.join(tmpdir, 'test.hdf5')
        
        x = pde.get_x()
        _ic = pde.sc.phys2wave(expected_sol(0.0, x))
        with h5py.File(tmpfname, 'w') as fh:
            pde.mkInitDataSet(_ic,fh,maxdt)
            pde.evolve(fh, trange)
            observed_w = fh['u'][()]
        
        observed = pde.sc.wave2phys(observed_w)

        expected = expected_sol(trange[:,np.newaxis],x[np.newaxis,:])
        # comparison
        testing.assert_allclose(observed, expected, rtol=1.0e-4)
    finally:
        os.remove(tmpfname)

def test_pde():
    heat = Heat(NW,pow=1)
    heatNH = HeatNH(NW,pow=1)
    advdiff = AdvDiff(NW)

    pdes = [heat, heatNH, advdiff]
 
    for pde, exsol in zip(pdes, expected_sols_n):
        yield check_pde, pde, exsol
   

