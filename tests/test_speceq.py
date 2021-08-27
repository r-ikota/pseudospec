import numpy as np
from numpy import testing
import sympy as sy
import pseudospec as ps
import tests.expected_values_pde as evpde
from tests.expected_values_pde import t, zero, one
from pseudospec import SpecEQ
import tempfile, h5py, os


NW = 50
K = 5
trange = np.linspace(0.0, 1.0, 21)
xsym = sy.var("x", real=True)

expected_sols = [evpde.PDEsol1(K), evpde.PDEsol2(K), evpde.PDEsol3()]

expected_sols_n = []
for _ex_sol in expected_sols:
    expected_sols_n.append(_ex_sol.getNumSol())


def sym2num(sc, f):
    xnum = sc.get_x()
    _fnum = sy.lambdify(xsym, f, "numpy")(xnum)
    _fnum = np.real(_fnum)
    return _fnum


def coeff2wave(sc, coeff):
    N2 = sc.N2
    J = sc.J
    w = sc.zeros()
    w[:N2] = np.asarray(coeff) * J
    return w


class Heat(SpecEQ):
    def __init__(self, NW):
        SpecEQ.__init__(self, NW)
        self._paramNames = ()
        self._paramDefault = []

    def eq(self, t, u):
        """
        du/dt = u_xx
        """

        return self.sc.sdiff2(u)


class HeatNH(SpecEQ):
    def __init__(self, NW, NC=1):
        SpecEQ.__init__(self, NW, NC=NC)
        self._paramNames = ()
        self._paramDefault = []

        _exsol = expected_sols[1]
        self._fp = _exsol.getNumF()
        self.x = self.sc.get_x()

    def eq(self, t, u):
        """
        du/dt = u_xx + f
        """

        x = self.x
        f = self.sc.transform_wp2wc(np.array(self._fp(t, x)).reshape(1, self.J))

        return self.sc.sdiff2(u) + f


class AdvDiff(SpecEQ):
    def __init__(self, N, NC=1):
        SpecEQ.__init__(self, N, NC=NC)
        self._paramNames = ()
        self._paramDefault = []

        _exsol = expected_sols[2]
        self._fp = _exsol.getNumF()
        self._a = _exsol.getNumA()
        self.x = self.sc.get_x()

    def eq(self, t, u):
        """
        du/dt = (u_x - a*u)_x + f
        """

        sc = self.sc
        x = self.x
        f = sc.transform_wp2wc(np.array(self._fp(t, x)).reshape(1, self.J))
        a = sc.transform_wp2wc(np.array(self._a(t, x)).reshape(1, self.J))
        return sc.sdiff1(sc.sdiff1(u) - sc.mult2(a, u)) + f


tmpdir = tempfile.gettempdir()


def check_pde(pde, expected_sol):
    try:
        tmpfname = os.path.join(tmpdir, "test.hdf5")

        x = pde.sc.get_x()
        _ic = pde.sc.transform_wp2wc(np.array(expected_sol(0.0, x)).reshape(1, pde.J))
        with h5py.File(tmpfname, "w") as fh:
            pde.mkInitDataSet(_ic, fh)
            pde.evolve(fh, trange, max_step=1.0e-4, method="BDF")
            observed = fh["wp"][()]

        # observed = pde.sc.wave2phys(observed_wp)

        expected = np.array(
            expected_sol(trange[:, np.newaxis], x[np.newaxis, :])
        ).reshape((trange.shape[-1], 1, x.shape[-1]))
        # comparison
        testing.assert_allclose(observed, expected, rtol=1.0e-4)
    finally:
        os.remove(tmpfname)


def test_pde():
    heat = Heat(NW)
    heatNH = HeatNH(NW)
    advdiff = AdvDiff(NW)

    pdes = [heat, heatNH, advdiff]

    for pde, exsol in zip(pdes, expected_sols_n):
        yield check_pde, pde, exsol
