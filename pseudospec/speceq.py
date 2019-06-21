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

    def evolve(self, fh, atrange, args=(), NTlump=100, **keyargs):
        '''
        keyargs: options except max_step that are passed to solve_ivp
        '''

        eq = self.eq

        trange_dset = fh['trange']
        last_t = trange_dset[-1]
        lidx = trange_dset.size - 1
        size_append = atrange.size - 1
        trange_dset.resize((lidx + 1 + size_append, ))
        ctrange = last_t + atrange
        trange_dset[lidx + 1:] = ctrange[1:]

        if 't_hist' in fh:
            t_hist_dset = fh['t_hist']
            t_hist_dset.resize((t_hist_dset.size + 1, 2))
        else:
            t_hist_dset = fh.create_dataset(
                't_hist', (1,2),
                maxshape=(None, 2),
                dtype=np.float64
            )
        
        t_hist_dset[-1] = (ctrange[0], ctrange[-1])


        u_dset = fh['u']
        u0 = np.empty(u_dset.shape[1], dtype=np.float64)
        u0[:] = u_dset[-1,:]        
        u_dset.resize(lidx + 1 + size_append, axis=0)
        
        tsize = ctrange.size
        round = tsize // NTlump
        if round == 0:
            sid = [0]
            eid = [tsize - 1]
        else:
            sid = [NTlump*r for r in range(round)]
            eid = sid[1:]
            eid.append(tsize - 1)

        for s,e in zip(sid, eid):
            t_span = (ctrange[s], ctrange[e])
            t_eval = ctrange[s: e+1]
            sol = solve_ivp(
                lambda t, u: eq(t, u, *args), 
                t_span,
                u0,
                t_eval=t_eval,
                **keyargs
            )
            u_extd = np.transpose(sol.y)
            u_dset[lidx + s + 1: lidx + e + 1] = u_extd[1:]
            u0[:] = u_extd[-1]


    def eq(self, t, u, *args):
        pass

            



    def mkPhysData(self, Uresh):
        '''
        Uresh: reshaped data
        '''

        return irfft(Uresh, self.J)

    def reshapeTS(self, U):
        '''
        To reshape a time series wave datum U
        U(Nt, NC*N2): wave data
        '''
        
        Nt = U.shape[0]
        return U.reshape((Nt, self.NC, self.N2))



    def mkInitDataSet(self, u, fh, args=()):
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
        fh['N'] = self.N
        fh['J'] = self.J
        if args: fh['args'] = args

        
