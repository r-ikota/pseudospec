# coding: utf-8

#%%
import sympy as sy, numpy as np
from sympy import I, pi, exp

#%%
x = sy.var("x", real=True)
z = sy.var("z", complex=True)
one = sy.sympify(1)
zero = sy.sympify(0)


#%%
def phi(n):
    return exp(2 * pi * I * n * x)


#%%


def coeff2wave(_f):
    _g = zero
    for j, c in enumerate(_f[1:]):
        _g += c * phi(j + 1)
    _g += sy.conjugate(_g)
    _g += _f[0]
    return _g


def wave2coeff(_f):
    _f = _f.expand().subs({phi(1): z}).collect(z)
    _nw = sy.degree(_f)
    _coeff = [_f.coeff(z, 0)]
    for i in range(1, _nw + 1):
        _coeff.append(_f.coeff(z, i))
    return _coeff


#%%
f_coeffs = []  # to contain four sets of symbolic coefficients of positive wave numbers.

#%%
# f1
_f = [zero] * 16

_f[0] = one
_f[1] = one / 2
_f[5] = (2 + 3 * I) / 5
_f[12] = (1 - 3 * I) / 36

f_coeffs.append(_f)

#%%
# f2

_f = [zero] * 16

_f[0] = -one / 3
_f[3] = (2 - I) / pi
_f[4] = (-3 + I) / (4 * pi)
_f[15] = one / pi ** 4

f_coeffs.append(_f)


#%%
# f3

_f = [zero] * 16
_f[0] = pi ** 2
_f[6] = -1 + 2 * I
_f[14] = (1 - 3 * I) / pi ** 3

f_coeffs.append(_f)


#%%
# f4

_f = [zero] * 16
_f[0] = -12 * one
_f[9] = 3 - I
_f[15] = one / 128

f_coeffs.append(_f)


#%%
fs = []  # to contain four sets of wave data in symbolic form.
for _f in f_coeffs:
    f = coeff2wave(_f)
    fs.append(f)

#%%
fft_coeffs_r = (
    []
)  # to contain numerical coefficents of fft wave data in the real format.
for _f in f_coeffs:
    _a = _f[0]

    _fft_coeff_real = [sy.re(_a).evalf()]
    _fft_coeff_imag = []
    for _ab in _f[1:]:
        _fft_coeff_real.append(sy.re(_ab).evalf())
        _fft_coeff_imag.append(sy.im(_ab).evalf())
    fft_coeffs_r.append(_fft_coeff_real + _fft_coeff_imag)

#%%
fft_coeffs_c = (
    []
)  # to contain numerical coefficents of fft wave data in the complex format.
for _f in f_coeffs:
    _fft_coeff_c = []
    for _c in _f:
        _fft_coeff_c.append(complex(_c.evalf()))
    fft_coeffs_c.append(np.array(_fft_coeff_c))

#%%
def trunc(f, NW):
    _fexz = f.expand().subs({phi(1): z})
    _fout = _fexz.coeff(z, 0)
    for i in range(1, NW + 1):
        _fout += _fexz.coeff(z, i) * phi(i) + _fexz.coeff(z, -i) * phi(-i)
    return _fout
