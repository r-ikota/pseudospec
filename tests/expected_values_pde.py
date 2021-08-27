# coding: utf-8

#%%
import sympy as sy
from sympy import I, pi, exp, sin

#%%
t, x = sy.var("t x", real=True)
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


def sym2numfuc(args, sfun):
    return sy.lambdify(args, sy.re(sy.expand_complex(sfun)), "numpy")


#%%
def trunc(f, NW):
    _fexz = f.expand().subs({phi(1): z})
    _fout = _fexz.coeff(z, 0)
    for i in range(1, NW + 1):
        _fout += _fexz.coeff(z, i) * phi(i) + _fexz.coeff(z, -i) * phi(-i)
    return _fout


class PDEsol:
    def __init__(self):
        self.ic = None
        self.sol = None
        self.f = None

    def _setIC(self):
        self.ic = self.sol.subs({t: zero})

    def getNumSol(self):
        # _numsol = sy.lambdify((t,x), sy.re(self.sol), 'numpy')
        _numsol = sym2numfuc((t, x), self.sol)
        return _numsol

    def getNumF(self):
        # _numf = sy.lambdify((t,x), sy.re(self.f), 'numpy')
        _numf = sym2numfuc((t, x), self.f)
        return _numf


class PDEsol1(PDEsol):
    def __init__(self, K):
        sol = zero
        for k in range(1, K + 1):
            _wave = exp(-k) * (one + sin(k * pi / 3)) * phi(k)
            sol += _wave * exp(-4 * pi ** 2 * k ** 2 * t)
        sol += sy.conjugate(sol)
        sol += one
        self.sol = sol
        self.f = (sol.diff(t) - sol.diff(x, 2)).simplify()

        self._setIC()


class PDEsol2(PDEsol):
    def __init__(self, K):
        sol = zero
        for k in range(1, K + 1):
            omegan = 5 * k * pi / 7
            fn = 2 * one / k
            _wave = fn / (4 * pi ** 2 * k ** 2 + I * omegan) * phi(k)
            sol += _wave * exp(I * omegan * t)
        sol += sy.conjugate(sol)
        sol += one
        self.sol = sol
        self.f = (sol.diff(t) - sol.diff(x, 2)).simplify()

        self._setIC()


class PDEsol3(PDEsol):
    def __init__(self):

        sol = zero

        sol = exp(I * 3 * t) * phi(1) / 8 + exp(I * t) * phi(3) / 4
        sol += sy.conjugate(sol)
        sol += one

        a = exp(2 * pi * I * t) * phi(2)
        a += sy.conjugate(a)

        f = (sol.diff(t) - (sol.diff(x) - a * sol).diff(x)).simplify()

        self.sol = sol
        self.a = a
        self.f = f

        self._setIC()

    def getNumA(self):
        # _numa = sy.lambdify((t,x), sy.re(self.a), 'numpy')
        _numa = sym2numfuc((t, x), self.a)
        return _numa
