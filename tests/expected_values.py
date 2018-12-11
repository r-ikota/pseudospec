
# coding: utf-8

#%%
import sympy as sy
from sympy import I, pi, exp

#%%
sy.var('x y a b', real=True)
one = sy.sympify(1)
zero = sy.sympify(0)

#%%
sy.exp(x**2)

#%%
def phi(n):
    return exp(2*pi*I*n*x)

#%%

def coeff2waves(_f):
    _g = zero
    for j,c in enumerate(_f):
        _g += c*phi(j)
    _g = _g + sy.conjugate(_g)
    return _g


#%%
# f1
f_coeffs = []

_f = [zero]*16

_f[0] = one
_f[1] = one/2
_f[5] = (2 + 3*I)/5
_f[12] = (1 - 3*I)/36

f_coeffs.append(_f)


#%%
#f2

_f = [zero]*16

_f[0] = -one/3
_f[3] = (2 - I)/pi
_f[4] = (-3 + I)/(4*pi)
_f[15] = one/pi**4

f_coeffs.append(_f)


#%%
#f3

_f = [zero]*16
_f[0] = pi**2
_f[6] = (-1 + 2*I)
_f[14] = (1 - 3*I)/pi**3

f_coeffs.append(_f)


#%%
#f4

_f = [zero]*16
_f[0] = -12*one
_f[9] = 3 - I
_f[15] = one/128

f_coeffs.append(_f)



#%%
fs = []
for _f in f_coeffs:
    f = coeff2waves(_f)
    fs.append(f)
