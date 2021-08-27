from __future__ import division
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len


class SpecCalc:

    """
    A class for pseudo-spectral method.
    Functions treated are assumed to be real-valued.

    Arguments for instantiation
    ---------------------------
    NW: int
        The truncation wave number.

    Attributes
    ----------
    NW: int
        The truncation wave number.
    NWrsize: int
        The size of wave data in the real format, that is, NWrsize = 2*NW + 1.
        If u = sum(n=-N, ..., N) C(N) * exp(2 * pi * i * n * x), then the wave data is stored as (A(0), A(1), A(2), ..., A(N), B(1), ..., B(N)), where A(n) = Re C(n) and B(n) = Im C(n).
    J: int
        The size of the wave data in the physical space format.
        For a real-valued u, the physical space data is (u(x[0]), u(x[1]), ..., u(x[[J-1])).
    J2: int
        The size of the wave data in the physical space format for the multiplication of two wave data.
    J3: int
        The size of the wave data in the physical space format for the multiplication of three wave data.

    """

    def __init__(self, NW):
        J = next_fast_len(2 * NW + 1)
        self.NW = NW

        self.NWrsize = 2 * self.NW + 1

        self.J = J

        J2 = next_fast_len(3 * self.NW + 1)
        self.J2 = J2
        J3 = next_fast_len(4 * self.NW + 1)
        self.J3 = J3

        self._D1factor = 2.0j * np.pi * np.arange(self.NW + 1)
        self._D2factor = self._D1factor ** 2
        self._x = np.linspace(0.0, 1.0, J, endpoint=False)

    def get_zero_rwave(self, shape=()):
        """
        Returns a zero-padded wave data in the real format.

        Parameters
        ----------
        shape: array_like, optional

        Returns
        -------
        ret: (shape[0], shape[1], ... , shape[-1], 2*NW + 1) ndarray with the dtype float64.
            A zero-padded array with the shape list(shape) + [2*NW + 1].
        """
        return np.zeros(list(shape) + [self.NWrsize], dtype=np.float64)

    def get_zero_cwave(self, shape=()):
        """
        Returns a zero-padded wave data in the complex format.

        Parameters
        ----------
        shape: array_like, optional

        Returns
        -------
        ret: (shape[0], shape[1], ... , shape[-1], NW + 1) ndarray with the dtype complex128.
            A zero-padded array with the shape list(shape) + [NW + 1].
        """
        return np.zeros(list(shape) + [self.NW + 1], dtype=np.complex128)

    def get_one_cwave(self, shape=()):
        """
        Returns a wave data representing the constant function with value one in the complex format.

        Parameters
        ----------
        shape: array_like, optional

        Returns
        -------
        ret: (shape[0], shape[1], ... , shape[-1], NW + 1) ndarray with the dtype complex128.
            An array with value one, that is, ret[..., 0] = 1.0 and otherwize = 0.0. Its shape is list(shape) + [NW + 1].
        """
        ret = self.get_zero_cwave(shape)
        ret[..., 0] = 1.0
        return ret

    def get_zero_pwave(self, shape=()):
        """
        Returns a zero-padded wave data in the physical space.

        Parameters
        ----------
        shape: array_like, optional

        Returns
        -------
        ret: (shape[0], shape[1], ... ,shape[-1], J) ndarray with the dtype complex128.
            A zero-padded array with the shape list(shape) + [NW + 1].        
        """
        return np.zeros(list(shape) + [self.J], dtype=np.float64)

    def wconvert_r2c(self, wr):
        """
        Converts wave data from the real format to the complex format.

        Parameters
        ----------
        wr: (M1, M2, ..., Md, 2*NW + 1) real ndarray 
            Input wave data in the real format.

        Return
        ------
        wc: (M1, M2, ..., Md, NW + 1) complex ndarray
            An array where the output is stored.
        """
        wc = self.get_zero_cwave(wr.shape[:-1])
        wc[..., 0] = wr[..., 0]
        wc[..., 1:] = wr[..., 1 : self.NW + 1] + 1.0j * wr[..., self.NW + 1 :]
        return wc

    def wconvert_c2r(self, wc):
        """
        Converts wave data from the real format to the complex format.

        Parameters
        ----------
        wc: (M1, M2, ..., Md, NW + 1) complex ndarray
            An array where the output is stored.

        Return
        ------
        wr: (M1, M2, ..., Md, 2*NW + 1) real ndarray 
            Input wave data in the real format.
        """
        wr = self.get_zero_rwave(wc.shape[:-1])
        wr[..., 0] = np.real(wc[..., 0])
        wr[..., 1 : self.NW + 1] = np.real(wc[..., 1:])
        wr[..., self.NW + 1 :] = np.imag(wc[..., 1:])
        return wr

    def get_x(self):
        """
        Returns the x-grid points in the physical space.
        """
        return self._x.copy()

    def transform_wc2wp(self, wc):
        """
        Transforms a complex wave data to a physical space wave data.

        Parameters
        ----------
        wc: (M1, M2, ..., Md, NW + 1) complex ndarray
            An array where the output is stored.

        Returns
        -------
        wp: (M1, M2, ..., Md, J) real ndarray 
            Input wave data in the real format.


        Note
        ----
        Expept for the last axis, the shapes of wr and wc are the same.
        """
        return irfft(wc * self.J, n=self.J)

    def transform_wp2wc(self, wp):
        """
        Transforms a physical space wave data to a complex wave data.

        Parameters
        ----------
        wp: (M1, M2, ..., Md, J) real ndarray 
            Input wave data in the real format.

        Returns
        -------
        wc: (M1, M2, ..., Md, NW + 1) complex ndarray
            An array where the output is stored.


        Note
        ----
        Expept for the last axis, the shapes of wr and wc are the same.
        """
        ret = rfft(wp) / wp.shape[-1]
        return np.array(ret[..., : self.NW + 1])

    def sdiff1(self, wc_in):
        """
        Differentiates a wave data once.

        Parameters
        ----------
        wc_in: (M1, M2, ..., Md, NW + 1) complex ndarray
            An array to be differentiated.

        Returns
        -------
        wc_out: (M1, M2, ..., Md, NW + 1) complex ndarray
            An array where the output is stored.
        """

        return wc_in * self._D1factor

    def sdiff2(self, wc_in):
        """
        Differentiates a wave data twice.

        Parameters
        ----------
        wc_in: (M1, M2, ..., Md, NW + 1) complex ndarray
            An array to be differentiated.

        Returns
        -------
        wc_out: (M1, M2, ..., Md, NW + 1) complex ndarray
            An array where the output is stored.
        """

        return wc_in * self._D2factor

    def sdiff(self, wc_in, k):
        """
        Computes a given order derivative of a wave data .

        Parameters
        ----------
        wc_in: (M1, M2, ..., Md, NW + 1) complex ndarray
            An array to be differentiated.
        k: int
            The order of the differentiation.

        Returns
        -------
        wc_out: (M1, M2, ..., Md, NW + 1) complex ndarray
            An array where the output is stored.
        """

        return wc_in * self._D1factor ** k

    def fdiff(self, u):
        """
        Differentiates a physical space wave data with the central difference.
        
        """
        return (u[..., 2:] - u[..., :-2]) / 2.0 * self.J

    def mult2(self, wc_in1, wc_in2):
        """
        Multiply two wave data in the complex format.

        Parameters
        ----------
        wc_in1, wc_in2: (M1, M2, ..., Md, NW + 1) complex ndarray
            Wave data to be multiplied.

        Returns
        -------
        wc_out: (M1, M2, ..., Md, NW + 1) complex ndarray
            An array where the output is stored.
        """

        product_phys = irfft(wc_in1 * self.J2, n=self.J2) * irfft(
            wc_in2 * self.J2, n=self.J2
        )
        return np.array(rfft(product_phys)[..., : self.NW + 1] / self.J2)

    def mult3(self, wc_in1, wc_in2, wc_in3):
        """
        Multiply three wave data in the complex format.

        Parameters
        ----------
        wc_in1, wc_in2, wc_in3: (M1, M2, ..., Md, NW + 1) complex ndarray
            Wave data to be multiplied.

        Returns
        -------
        wc_out: (M1, M2, ..., Md, NW + 1) complex ndarray
            An array where the output is stored.
        """

        product_phys = (
            irfft(wc_in1 * self.J3, n=self.J3)
            * irfft(wc_in2 * self.J3, n=self.J3)
            * irfft(wc_in3 * self.J3, n=self.J3)
        )
        return np.array(rfft(product_phys)[..., : self.NW + 1] / self.J3)
