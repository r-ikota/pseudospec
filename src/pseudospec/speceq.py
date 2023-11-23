import numpy as np
from scipy.integrate import solve_ivp
from .speclib import SpecCalc
from tqdm import tqdm as tqdm_tf  # for terminal and file output
from tqdm.notebook import tqdm as tqdm_nb  # for notebook output


def get_itr(itr, total: int, pb_type=None, pb_file=None):
    if pb_type is None:
        yield from itr

    elif pb_type == "terminal":
        yield from tqdm_tf(itr, total=total)

    elif pb_type == "notebook":
        yield from tqdm_nb(itr, total=total)

    elif pb_type == "file":
        if pb_file is None:
            with open("pb-log", "w", encoding="utf_8") as pfile:
                yield from tqdm_tf(itr, total=total, file=pfile)
        else:
            yield from tqdm_tf(itr, total=total, file=pb_file)


class SpecEQ:
    """
    An abstract class for equations.

    Parameters for initializer
    --------------------------
    NW: int
        The truncation wave number.
    NC: int
        The number of unknown variables.
    sc: SpecCalc instance
    """

    def __init__(self, NW, NC=1):
        self.NW = NW
        self.NC = NC
        self.sc = SpecCalc(NW)
        self.J = self.sc.J
        self.NWrsize = self.sc.NWrsize

        self._paramNames = ()
        self._paramDefault = []

    def reset(self, NW):
        """
        Resets the truncation wave number NW.

        Parameters
        ----------
        NW: int
            A new truncation wave number.
        """
        self.__init__(NW)

    def getParamNames(self):
        return self._paramNames

    def getParamDefault(self):
        return self._paramDefault.copy()

    def wconvert_wrf2wc(self, wrf):
        """
        Converts a wave data in the flatten real format to one in the complex format.

        Parameters
        ----------
        wrf: (..., NC*NWrsize) real ndarray
            A wave data in the flatten real format.

        Returns
        -------
        wc: (..., NC, NW+1) imaginary ndarray
            The wave data converted to the complex format.
        """
        return self.sc.wconvert_r2c(
            np.reshape(wrf, list(wrf.shape[:-1]) + [self.NC, self.NWrsize])
        )

    def wconvert_wc2wrf(self, wc):
        """
        Converts a wave data in the complex format to one in the flatten real format.

        Parameters
        ----------
        wc: (..., NC, NW+1) imaginary ndarray
            A wave data in the complex format.

        Returns
        -------
        wrf: (..., NC*NWrsize) real ndarray
            The wave data converted to the flatten real format.
        """
        return np.reshape(
            self.sc.wconvert_c2r(wc),
            list(wc.shape[:-2]) + [self.NC * self.NWrsize],
        )

    def evolve(
        self,
        fh,
        atrange,
        args=(),
        NTlump=100,
        pb_type=None,
        pb_file="pb-log.txt",
        **keyargs
    ):
        """

        Parameters
        ----------
        atrange: 1D nd array
            Assumed to start from zero.

        pb_type: str
            The type of the output for the progress bar;
            either None, 'terminal', 'notebook', or 'file'.

        pb_file: str
            If pb_type == 'file', then the progress bar will be written to the file
            specified with pb_file.

        keyargs: options except max_step that are passed to solve_ivp

        """

        eq = self.eq

        # Divide time grid points into groups, the number of each is NTlump or less.
        trange_dset = fh["trange"]
        last_t = trange_dset[-1]
        lidx = trange_dset.size
        size_append = atrange.size - 1
        trange_dset.resize((lidx + size_append,))
        ctrange = last_t + atrange
        trange_dset[lidx:] = ctrange[1:]

        if "t_hist" in fh:
            t_hist_dset = fh["t_hist"]
            t_hist_dset.resize((t_hist_dset.size + 1,))
        else:
            t_hist_dset = fh.create_dataset(
                "t_hist", (1,), maxshape=(None,), dtype=np.float64
            )

        t_hist_dset[-1] = ctrange[-1]

        tsize = ctrange.size
        div_num = tsize // NTlump
        if div_num == 0:
            sid = [0]
            eid = [tsize - 1]
        else:
            sid = [NTlump * r for r in range(div_num)]
            eid = sid[1:]
            eid.append(tsize - 1)

        # Prepare the initial data.
        wc_dset = fh["wc"]
        wrf0 = self.wconvert_wc2wrf(wc_dset[-1])  # wc0 = u_dset[-1]

        # Resize the hdf5 dataset.
        wc_dset.resize(lidx + size_append, axis=0)
        wp_dset = fh["wp"]
        wp_dset.resize(lidx + size_append, axis=0)
        powerspec_dset = fh["powerspec"]
        powerspec_dset.resize(lidx + size_append, axis=0)

        # Evolve.
        for s, e in get_itr(
            zip(sid, eid), total=len(sid), pb_type=pb_type, pb_file=pb_file
        ):
            t_span = (ctrange[s], ctrange[e])
            t_eval = ctrange[s : e + 1]
            sol = solve_ivp(
                lambda t, wrf: self.wconvert_wc2wrf(
                    eq(t, self.wconvert_wrf2wc(wrf), *args)
                ),
                t_span,
                wrf0,
                t_eval=t_eval,
                **keyargs
            )
            wrf_extd = np.transpose(sol.y)[1:]
            wc_extd = self.wconvert_wrf2wc(wrf_extd)

            # Store data.
            wc_dset[lidx + s : lidx + e] = wc_extd
            wp_dset[lidx + s : lidx + e] = self.sc.transform_wc2wp(wc_extd)
            powerspec_dset[lidx + s : lidx + e] = np.real(
                wc_extd * np.conjugate(wc_extd)
            )

            # Set the initial data for the next computation.
            wrf0[:] = wrf_extd[-1]

    def eq(self, t, u, *args):
        pass

    def mkInitDataSet(self, wc0, fh):
        """
        Prepare an hd5 dataset for the initial condition of a PDE.

        Parameters
        ----------
        wc0: (NC, NW + 1) complex ndarray
            An initial wave data in the complex format.
        """

        # Create a dataset for a wave data in the complex format.
        dataset_wc = fh.create_dataset(
            "wc",
            (1, self.NC, self.NW + 1),
            dtype=np.complex128,
            maxshape=(None, self.NC, self.NW + 1),
        )
        dataset_wc[0] = wc0

        # Create a dataset for a wave data in the physical space.
        dataset_wp = fh.create_dataset(
            "wp",
            (1, self.NC, self.J),
            dtype=np.float64,
            maxshape=(None, self.NC, self.J),
        )
        dataset_wp[0] = self.sc.transform_wc2wp(wc0)

        # Create a dataset for the power spectrum of a wave data.
        dataset_powerspec = fh.create_dataset(
            "powerspec",
            (1, self.NC, 1 + self.NW),
            dtype=np.float64,
            maxshape=(None, self.NC, self.J),
        )
        dataset_powerspec[0] = np.real(wc0 * np.conjugate(wc0))

        ds_trange = fh.create_dataset(
            "trange", (1,), dtype=np.float64, maxshape=(None,)
        )
        ds_trange[0] = 0.0
        fh["NW"] = self.NW
        fh["J"] = self.J
