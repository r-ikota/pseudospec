import numpy as np 
import h5py
from pseudospec import SpecEQ


class FHN(SpecEQ):
    def __init__(self, N, NC=2, pow=3, **par):
        SpecEQ.__init__(self, N, NC=NC, pow=pow, **par)
        self._w = np.zeros((3, self.N2))
        self._paramNames = ('a', 'eps', 'gamma', 'Du', 'Dv')
        self._paramDefault = [0.25, 0.05, 3.0, 4.0e-5, 1.0e-7]

    def eq(self, t, uv):
        '''
        du/dt = Du u_xx + (u(1-u)(u-a) - v)/eps
        dv/dt = Dv v_xx + eps(u - gamma v)
        '''

        a, eps, gamma, Du, Dv = self.getArgs()

        sc = self.sc
        N2 = self.N2

        inveps = 1.0/eps

        u,v = uv.reshape((2,N2))
        _w = self._w
        ret = np.zeros(2*N2)
        dudt, dvdt = ret.reshape((2,N2))

        #calc dudt
        sc.sdiff2(u, _w[0])
        sc.mult3(u, self.One - u, u - a*self.One, _w[2])
        dudt[:] = Du*_w[0] + inveps*(_w[2] - v)

        #calc dvdt
        sc.sdiff2(v, _w[0])
        dvdt[:] = Dv*_w[0] + eps*(u - gamma*v)

        return ret


class KS(SpecEQ):
    def __init__(self, N, NC=1, pow=2, **par):
        SpecEQ.__init__(self, N, NC=NC, pow=pow, **par)
        self._w = np.zeros((3, self.N2))
        self._paramNames = ('nu', 'L')
        self._paramDefault = [1.0, 22.0]

    def eq(self, t, u):
        '''
        du/dt = -[nu L**(-4) u_xxxx + L**(-2)u_xx
                    + 1/(2L) (u**2)_x]
        '''

        nu, L = self.getArgs()
        sc = self.sc
        N2 = self.N2

        _w = self._w

        ret = np.zeros(N2)

        sc.sdiff2(u, _w[0])
        sc.sdiff2(_w[0], _w[1])
        sc.mult2(u,u,_w[2])
        sc.sdiff1(_w[2], _w[2])

        ret[:] = -nu*L**(-4)*_w[1] - L**(-2)*_w[0] - _w[2]/2.0/L

        return ret


