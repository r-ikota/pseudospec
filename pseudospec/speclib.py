from __future__ import division
import numpy as np
from scipy.fftpack import rfft, irfft

def NWtoJ(NW, pow=2):
    return 2**np.int(np.ceil(np.log2((pow+1)*NW + 1)))

class SpecCalc:

    def __init__(self, NW, pow=2):

        self.pow = pow
        J = NWtoJ(NW, pow)
        self.J = J

        self.NW = NW
        self.N2 = 2*NW+1
        self.k2pi = 2.0*np.pi*np.arange(1.0, self.NW+1)
        self.k2piSecPow = self.zeros(self.N2)
        self.k2piSecPow[1:self.N2:2] = self.k2pi**2
        self.k2piSecPow[2:self.N2:2] = self.k2pi**2
        self._w = np.zeros((4, J))
        self._x = np.arange(0., 1., 1./J)

    def zeros(self, K = None):
        if not K: K = self.N2
        return np.zeros(K)

    def get_x(self):
        return self._x.copy()

    def wave2phys(self,w):
        return irfft(w,n=self.J)
        
    def trunc(self, uh, out=None):
        N2 = self.N2

        ret = None
        if not isinstance(out, np.ndarray):
            out = self.zeros()
            ret = out

        out[:N2] = uh[:N2]
        out[N2:] = 0.0
        return ret

    def sdiff(self, uh, out=None):
        N2 = self.N2
        w = self._w[0]
        ret = None
        if not isinstance(out, np.ndarray):
            out = self.zeros()
            ret = out
#        else:
#            w[:] = 0.

        # Real Part of G
        w[0] = 0.0
        w[1:N2:2] = -self.k2pi*uh[2:N2:2]

    
        # Imaginary Part of G
        w[2:N2:2] = self.k2pi*uh[1:N2:2]

        out[:N2] = w[:N2]
        out[N2:] = 0.0
        return ret

    def sdiff2(self, uh, out=None):
        N2 = self.N2
        #w = self._w[0]
        ret = None
        if not isinstance(out, np.ndarray):
            out = self.zeros()
            ret = out

        out[:N2] = -self.k2piSecPow*uh[:N2]
        out[N2:] = 0.0
        return ret

    def fdiff(self, u):
        return (u[2:] - u[:-2])/2.*self.J

    def mult(self, uh, vh, out=None):
        N2 = self.N2

        ret = None        
        if not isinstance(out, np.ndarray):
            out = self.zeros() 
            ret = out


        wh1 = self._w[0]; wh1[N2:] = 0.0; wh1[:N2] = uh[:N2]
        wh2 = self._w[1]; wh2[N2:] = 0.0; wh2[:N2] = vh[:N2]
        wh3 = self._w[2]

         
        irfft(wh1, overwrite_x = True)
        irfft(wh2, overwrite_x = True)
        np.multiply(wh1, wh2, wh3)
        rfft(wh3, overwrite_x = True)
#        ret[:] = wh3[:N2]
        out[:N2] = wh3[:N2]
        out[N2:] = 0.0

        return ret
        

    def mult3(self, uh, vh, wh, out=None):
        N2 = self.N2

        ret = None        
        if not isinstance(out, np.ndarray):
            out = self.zeros() 
            ret = out


        wh1 = self._w[0]; wh1[N2:] = 0.0; wh1[:N2] = uh[:N2]
        wh2 = self._w[1]; wh2[N2:] = 0.0; wh2[:N2] = vh[:N2] 
        wh3 = self._w[2]; wh3[N2:] = 0.0; wh3[:N2] = wh[:N2]
#        wh4 = self._w[3]        

        irfft(wh1, overwrite_x = True)
        irfft(wh2, overwrite_x = True)
        irfft(wh3, overwrite_x = True)

        np.multiply(wh1, wh2, wh2)
        np.multiply(wh2, wh3, wh3)

        rfft(wh3, overwrite_x = True)
#        ret[:] = wh3[:N2]
        out[:N2] = wh3[:N2]
        out[N2:] = 0.0

        return ret
