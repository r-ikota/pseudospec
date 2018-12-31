import numpy as np 
import h5py
from pseudospec import SpecEQ


class FHN(SpecEQ):
    def __init__(self, N, NC=2, pow=3, **par):
        SpecEQ.__init__(self, N, NC=NC, pow=pow, **par)
        self._w = np.zeros((3, self.N2))
        self._paramNames = ('a', 'eps', 'gamma', 'Du', 'Dv')
        self._paramDefault = [0.25, 0.05, 3.0, 4.0e-5, 1.0e-7]

    def eq(self, uv, t, a, eps, gamma, Du, Dv):
        '''
        du/dt = Du u_xx + (u(1-u)(u-a) - v)/eps
        dv/dt = Dv v_xx + eps(u - gamma v)
        '''

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

class BZ(SpecEQ):

    def __init__(self, N, NC=3, pow=2, **par):
#        setDefault2Par(bz_pardefaults, par)
        SpecEQ.__init__(self, N,NC, pow, **par)
        self._w = np.zeros((3,self.N2))
        
        self._paramNames = ('e', 'ep', 'q', 'f', 'Du', 'Dv', 'Dw')
        self._paramDefault = [9.9e-3, 1.98e-5, 7.62e-5, 1.0,
                    0.2e-5, 0.2e-5, 0.1e-7]


    def eq(self, uvw, t, e, ep, q, f, Du, Dv, Dw):
        '''
        du/dt = Du u_xx + (qv - uv + u(1 - u))/e
        dv/dt = Dv v_xx + (-qv - uv + fw)/ep
        dw/dt = Dw w_xx + u - w
        '''
        sc = self.sc
        N2 = self.N2

#        e, ep, q, f, Du, Dv, Dw = args
        inve = 1.0/e
        invep = 1.0/ep

        u, v, w = uvw.reshape((3,N2))
        _w = self._w
        ret = np.zeros(3*N2)
        dudt, dvdt, dwdt = ret.reshape((3,N2))

        # calc dudt
        #_w[0][:] = u
        sc.sdiff2(u, _w[0])
        sc.mult2(u,v,_w[1])
        sc.mult2(u, (self.One - u), _w[2])
        dudt[:] = Du*_w[0] + inve*(q*v - _w[1] + _w[2])

        # calc dvdt
        #_w[0][:] = v
        sc.sdiff2(v, _w[0])
        dvdt[:] = Dv*_w[0] + invep*(-q*v - _w[1] + f*w)

        #calc dwdt
        #_w[0][:] = w
        sc.sdiff2(w, _w[0])
        dwdt[:] = Dw*_w[0] + u - w

        return ret

class KS(SpecEQ):
    def __init__(self, N, NC=1, pow=2, **par):
        SpecEQ.__init__(self, N, NC=NC, pow=pow, **par)
        self._w = np.zeros((3, self.N2))
        self._paramNames = ('nu', 'L')
        self._paramDefault = [1.0, 22.0]

    def eq(self, u, t, nu, L):
        '''
        du/dt = -[nu L**(-4) u_xxxx + L**(-2)u_xx
                    + 1/(2L) (u**2)_x]
        '''

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


