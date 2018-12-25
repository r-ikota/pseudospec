
# coding: utf-8

#%%
import sympy as sy
from sympy import I, pi, exp, sin

#%%
t, x = sy.var('t x', real=True)
z = sy.var('z', complex=True)
one = sy.sympify(1)
zero = sy.sympify(0)

#%%
sy.exp(x**2)

#%%
def phi(n):
    return exp(2*pi*I*n*x)

#%%

def coeff2wave(_f):
    _g = zero
    for j,c in enumerate(_f[1:]):
        _g += c*phi(j+1)
    _g += sy.conjugate(_g)
    _g += _f[0]
    return _g

def wave2coeff(_f):
    _f = _f.expand().subs({phi(1): z}).collect(z)
    _nw = sy.degree(_f)
    _coeff = [_f.coeff(z,0)]
    for i in range(1,_nw+1):
        _coeff.append(_f.coeff(z,i))
    return _coeff

    

#%%
def trunc(f,NW):
    _fexz = f.expand().subs({phi(1): z})
    _fout = _fexz.coeff(z,0)
    for i in range(1,NW+1):
        _fout += _fexz.coeff(z,i)*phi(i)\
                    + _fexz.coeff(z,-i)*phi(-i)
    return _fout

class PDEsol():
    def __init__(self):
        self.ic = None
        self.sol = None
        self.f = None
        self.LHS = None

    def _setIC(self):
        self.ic = self.sol.subs({t: zero})

    def getNumIC(self):
        _numic = sy.lambdify(x, self.ic)
        return _numic

    def getNumSOL(self):
        _numsol = sy.lambdify((t,x), self.sol)
        return _numsol

    def getNumF(self):
        _numf = sy.lambdify((t,x), self.f)
        return _numf

class PDEsol1(PDEsol):
    def __init__(self,N):
        sol = zero
        for n in range(1,N+1):
            _wave = exp(-n)*(one + sin(n*pi/3))\
                    *phi(n)
            sol += _wave*exp(-4*pi**2*n**2*t)
        sol += sy.conjugate(sol)
        sol += one
        self.sol = sol
        self.f = zero
        self.LHS = (sol.diff(t) - sol.diff(x,2)).simplify()

        self._setIC()

class PDEsol2(PDEsol):
    def __init__(self,N):
        sol = zero
        f = zero
        for n in range(1,N+1):
            omegan = sin(n*pi/4)
            fn = exp(-n)
            _wave = fn/(4*pi**2*n**2 + I*omegan)*\
                    phi(n)
            sol += _wave*exp(I*omegan*t)
            f += fn*exp(I*omegan*t)*phi(n)
        sol += sy.conjugate(sol)
        f += sy.conjugate(f)
        sol += one
        self.sol = sol
        self.f = f
        self.LHS = (sol.diff(t) - sol.diff(x,2)).simplify()

        self._setIC()
    




    

