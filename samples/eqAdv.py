import numpy as np 
import h5py
from speceq import SpecEQ

class Adv(SpecEQ):


    def __init__(self, N, **par):
        '''
        N: the numver of waves
        a: time-dependent function
        q: the Fourier coefficients of the flow function

        '''

        SpecEQ.__init__(self, N, 2, **par)
        self._paramNames=('a', 'q', 'Du')
        self._paramDefault = [
            'sin(t) etc', 
            '[0, 8/(3*pi)*J, -8/(3*pi)*J, 0, -8.0/(3*pi)*J, -8/(9*pi)*J, -8/(9*pi)*J]']
        
        self._w = np.zeros((2, self.N2))

#        self.setFlow(q)
#        self.setAmp(a)
#        self.setDu(Du)

    def evolve(self, fh, atrange, Du, a, a_txt, q, NTlump=100):
        '''
        a (str): description of the amplitude function a(t) 
        '''

        vlstr = h5py.special_dtype(vlen=str)
        if 'a_txt' in fh: 
            a_txt_dset = fh['a_txt']
            size = a_txt_dset.size
            a_txt_dset.resize((size+1,))
            a_txt_dset[-1] = a_txt
        else:
            a_txt_dset = fh.create_dataset(
                        'a_txt', (1,), 
                        maxshape=(None,), dtype=vlstr
            )
            a_txt_dset[0] = a_txt
            
        self.setAmp(a)


        if 'q' in fh:
            q_dset = fh['q']
            shape = q_dset.shape
            q_dset.resize((shape[0]+1, shape[1]))
            q_dset[-1] = q
        else:
            q_dset = fh.create_dataset(
                'q', (1, self.N2),
                maxshape=(None, self.N2), 
                dtype=np.float64
            )
            q_dset[0] = q

        self.setFlow(q)

        if 'Du' in fh:
            Du_dset = fh['Du']
            size = Du_dset.size
            Du_dset.resize((size+1,))
            Du_dset[-1] = Du
        else:
            Du_dset = fh.create_dataset(
                'Du', (1,),
                maxshape=(None,),
                dtype=np.float64
            )
            Du_dset[0] = Du

        self.setDu(Du)
        
        SpecEQ.evolve(self, fh, self.eq, atrange, NTlump=NTlump)

    def getTypical_q(self):
        J = self.J
        q = self.sc.zeros()

        q[1] = -8.0/(3.0*np.pi)*J
        q[2] = 8.0/(3.0*np.pi)*J
        q[4] = 8.0/(3.0*np.pi)*J
        q[5] = 8.0/(9.0*np.pi)*J
        q[6] = 8.0/(9.0*np.pi)*J

        return q
        
    def setFlow(self,q):
        self._q = q

    def setAmp(self,a):
        self._a = a

    def setDu(self, Du):
        self._Du = Du


    def eq(self, u, t):
        '''
        Du: diffusion constant
        u_t = -(-Du u_x + a(t)q(x)u)_x
            = (Du u_x - a(t)q(x)u)_x

        '''

        sc = self.sc
        N2 = self.N2
        Du = self._Du
        a = self._a
        q = self._q

        _w = self._w
        dudt = np.zeros(N2)
        sc.sdiff(u, _w[0])
        sc.mult(q, u, _w[1])
        sc.sdiff(Du*_w[0] - a(t)*_w[1], dudt)

        return dudt

    def getFlowTS(self, trange):
        '''
        TS means Time Series

        To return a(t)*q [i, :]: fourier coefficients at time t
        '''
        return self._a(trange[:, np.newaxis])*self._q[np.newaxis, :]

    def getFluxTS(self, uTS, trange):
        ret = np.zeros(uTS.shape)
        
        for i, t in enumerate(trange):
            ret[i,:] = self.getFlux(uTS[i], t)

        return ret

    def getFlux(self, u, t):
        sc = self.sc
        _w = self._w
        Du = self._Du
        a = self._a
        q = self._q

        sc.sdiff(u, _w[0])
        sc.mult(q, u, _w[1])

        return -Du*_w[0] + a(t)*_w[1]

    # def mkInitDataSet(self, u, fh, dt, a):

    #     SpecEQ.mkInitDataSet(self, u, fh, dt)


