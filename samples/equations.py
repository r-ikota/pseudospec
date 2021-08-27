import numpy as np
from pseudospec import SpecEQ


class FHN(SpecEQ):
    def __init__(self, N, NC=2):
        SpecEQ.__init__(self, N, NC=NC)
        self._paramNames = ("a", "eps", "gamma", "Du", "Dv")
        self._paramDefault = [0.25, 0.05, 3.0, 4.0e-5, 1.0e-7]

        self.one = self.sc.get_one_cwave()

    def eq(self, t, uv, *args):
        """
        du/dt = Du u_xx + (u(1-u)(u-a) - v)/eps
        dv/dt = Dv v_xx + eps(u - gamma v)
        """

        a, eps, gamma, Du, Dv = args

        sc = self.sc

        u, v = uv[0], uv[1]
        DuvDt = sc.get_zero_cwave((2,))

        DuvDt[0] = (
            Du * sc.sdiff2(u) + (sc.mult3(u, self.one - u, u - a * self.one) - v) / eps
        )
        DuvDt[1] = Dv * sc.sdiff2(v) + eps * (u - gamma * v)

        return DuvDt


class KS(SpecEQ):
    def __init__(self, N, NC=1):
        SpecEQ.__init__(self, N, NC=NC)
        self._paramNames = ("nu", "L")
        self._paramDefault = [1.0, 22.0]

        self.one = self.sc.get_one_cwave((1,))

    def eq(self, t, u, *args):
        """
        du/dt = -[nu L**(-4) u_xxxx + L**(-2)u_xx
                    + 1/(2L) (u**2)_x]
        """

        nu, L = args
        sc = self.sc

        return -(
            nu * L ** (-4) * sc.sdiff(u, 4)
            + L ** (-2) * sc.sdiff2(u)
            + 1.0 / (2.0 * L) * sc.sdiff1(sc.mult2(u, u))
        )


class KdV(SpecEQ):
    def __init__(self, N, NC=1):
        SpecEQ.__init__(self, N, NC=NC)
        self._paramNames = ("mu",)
        self._paramDefault = [0.022 ** 2 / 4.0]

    def eq(self, t, u, *args):
        """
        du/dt + u * u_x + mu * u_xxx = 0
        """

        (mu,) = args
        sc = self.sc

        return -sc.mult2(u, sc.sdiff1(u)) - mu * sc.sdiff(u, 3)

