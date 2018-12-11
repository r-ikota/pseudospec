import numpy as np
from scipy.integrate import odeint
import pseudospec.speclib as sl

def setDefault2Par(d, par):
    for k in d:
        if not k in par:
            par[k] = d[k]

class SpecEQ():
    def __init__(self, N, pow=2, **par):
        '''
        N: the number of waves
        pow: the highest degree of the polynomials in nonlinear terms

        self._paramNames: tuple
        self._paramDefault: list
        '''
        
        self.N = N
        self.N2 = 2*N + 1
        self.sc = sl.SpecCalc(N, pow)
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

    def evolve(self, fh, eq, atrange, args=(), NTlump=100):
        dt = fh['dt'][()]

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
        t_hist_dset[-1] = [ctrange[0], ctrange[-1]]


        u_dset = fh['u']
        u0 = np.empty(u_dset.shape[1], dtype=np.float64)
        u0[:] = u_dset[-1,:]        
        u_dset.resize(last_idx + size_append, axis=0)
        
        

        tno = self.timeArrange(ctrange, dt, NTlump)
        subTranges = self.getSubTRanges(*tno)

        for strange, subidx in subTranges:
            u = odeint(eq, u0, strange, args=args)
            numadd = len(subidx)
            
            if numadd > 0:
                u_dset[last_idx:last_idx + numadd] = u[subidx]
                last_idx += numadd
            u0[:] = u[-1]

    def eq(self):
        pass

    # def mkDataGen4viz(self, U, numvar, Nt):
    #     '''
        # To make a data generator for visualization
        # U(Nt, numvar*N2): wave data
        # numvar: the number of the components of the variables
        # Nt: the number of the time points
        # '''
        # UU = self.ReshapeTS(U, numvar, Nt)
        # uu = self.mkPhysData(UU)

        # def data_gen(i):
        #     return UU[i], uu[i]

        # return data_gen

            



    def mkPhysData(self, Uresh):
        '''
        Uresh: reshaped data
        '''

        return sl.irfft(Uresh, self.J)

    def reshapeTS(self, U, numvar, Nt):
        '''
        To reshape a time series wave datum U
        U(Nt, numvar*N2): wave data
        numvar: the number of the components of the variables
        Nt: the number of the time points
        '''
        
        return U.reshape((Nt, numvar, self.N2))


    def timeArrange(self, trange, dt, NTlump):
        '''
        Arguments
            trange: 
                1D array; time grid points where data are picked

            dt: 
                time incriment for computation

            NTlump: 
                the number of time points during which 
                one round computation is performed

        Returned
            trange_calc:
                time grid points over which computation is performed

            NNTsep:
                separation indices where trange_calc is divided

            org2argd:
                indices of trange_calc 
                which the original time grid points correspond
        '''

        Ndt = np.max((np.int((trange[1] - trange[0])/dt), 1))
        NumDiv = Ndt*(len(trange)-1)
        trange_calc = np.linspace(
                trange[0], trange[-1], NumDiv + 1)
        
        NNT = np.arange(NumDiv + 2)
        NNTsep = NNT[::NTlump]
        NNTsep[-1] = NumDiv + 1
        
        org2argd = Ndt*np.arange(len(trange)) #original to arranged
        
        return trange_calc, NNTsep, org2argd


    def getSubTRanges(self, trange_calc, NNTsep, org2argd):
        NumDiv = len(trange_calc)
        subTranges = []
        for nstart in range(len(NNTsep)-1):
            s_index = NNTsep[nstart]
            e_index = np.min((NNTsep[nstart+1]+1, NumDiv))

            suborgidx = org2argd[
                    (s_index < org2argd) * (org2argd < e_index)
                    ] - s_index

            subTranges.append((trange_calc[s_index:e_index], suborgidx))
        return subTranges

    def mkInitDataSet(self, u, fh, dt, args=()):
        dataset = fh.create_dataset(
            'u', (1, self.N2),
            dtype='float32', 
            maxshape=(None, self.N2)
            )
        dataset[0,:] = u
        ds_trange = fh.create_dataset(
            'trange', (1,),
            dtype='float64',
            maxshape=(None,)
        )
        fh['dt'] = dt
        fh['N'] = self.N
        fh['J'] = self.J
        if args: fh['args'] = args

        
