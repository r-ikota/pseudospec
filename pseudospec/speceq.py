import numpy as np
from scipy.integrate import solve_ivp
from scipy.fftpack import rfft, irfft
from .speclib import SpecCalc

def setDefault2Par(d, par):
    for k in d:
        if not k in par:
            par[k] = d[k]

class SpecEQ():
    def __init__(self, N, NC = 1, pow=2, **par):
        '''
        N: the number of waves
        NC: the number of unknown variables
        pow: the highest degree of the polynomials in nonlinear terms

        self._paramNames: tuple
        self._paramDefault: list
        '''
        
        self.N = N
        self.N2 = 2*N + 1
        self.NC = NC
        self.sc = SpecCalc(N, pow)
        self.J = self.sc.J
        self.get_x = self.sc.get_x
        self._par = par
        self.One = self.sc.zeros(); self.One[0] = self.J

        self._paramNames = ()
        self._paramDefault = []
        

 
    def getParamNames(self):
        return self._paramNames


    def getParamDefault(self):
        return self._paramDefault.copy()

    def evolve(self, fh, atrange, args=()):
        self._setArgs(args)
        dt = fh['dt'][()]
        eq = self.eq

        trange_dset = fh['trange']
        last_t = trange_dset[-1]
        last_idx = trange_dset.size
        size_append = atrange.size - 1
        trange_dset.resize((last_idx + size_append, ))
        ctrange = last_t + atrange
        trange_dset[last_idx:] = ctrange[1:]

        if 't_hist' in fh:
            t_hist_dset = fh['t_hist']
            t_hist_dset.resize((t_hist_dset.size + 1, 2))
        else:
            t_hist_dset = fh.create_dataset(
                't_hist', (1,2),
                maxshape=(None, 2),
                dtype=np.float64
            )
        t_span = (ctrange[0], ctrange[-1])
        t_hist_dset[-1] = t_span


        u_dset = fh['u']
        u0 = np.empty(u_dset.shape[1], dtype=np.float64)
        u0[:] = u_dset[-1,:]        
        u_dset.resize(last_idx + size_append, axis=0)
        
        sol = solve_ivp(
            eq, 
            t_span,
            u0,
            t_eval=ctrange,
            max_step = dt
        )
        u_dset[last_idx:] = np.transpose(sol.y)[1:]

    def _setArgs(self, args):
        self._args = tuple(args)

    def getArgs(self):
        return self._args

    def eq(self):
        pass

            



    def mkPhysData(self, Uresh):
        '''
        Uresh: reshaped data
        '''

        return irfft(Uresh, self.J)

    def reshapeTS(self, U, Nt):
        '''
        To reshape a time series wave datum U
        U(Nt, NC*N2): wave data
        Nt: the number of the time points
        '''
        
        return U.reshape((Nt, self.NC, self.N2))



    def mkInitDataSet(self, u, fh, dt, args=()):
        dataset = fh.create_dataset(
            'u', (1, self.NC*self.N2),
            dtype='float32', 
            maxshape=(None, self.NC*self.N2)
            )
        dataset[0,:] = u
        ds_trange = fh.create_dataset(
            'trange', (1,),
            dtype='float64',
            maxshape=(None,)
        )
        ds_trange[0] = 0.0
        fh['dt'] = dt
        fh['N'] = self.N
        fh['J'] = self.J
        if args: fh['args'] = args

        
