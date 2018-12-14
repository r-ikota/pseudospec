import numpy as np
from numpy import testing
import sympy as sy
import pseudospec as ps
import expected_values as ev

NW = 15
xsym = sy.var('x', real=True)

def test_NtoJ():
    pow = 3
    J = ps.NWtoJ(NW, pow=3)
    assert type(J) == int and J > pow*NW + 1

def sym2num(sc,f):
        xnum = sc.get_x()
        _fnum = sy.lambdify(xsym, f, 'numpy')(xnum)
        _fnum = np.real(_fnum)
        return _fnum

def coeff2wave(sc,coeff):
        N2 = sc.N2
        J = sc.J
        w = sc.zeros()
        w[:N2] = np.asarray(coeff)*J
        return w

def check_each_funcs(sc,i):
    expected = sym2num(sc,ev.fs[i])
    w = coeff2wave(sc,ev.fft_coeffs[i])
    observed = sc.wave2phys(w)

    testing.assert_allclose(observed, expected)

def test_each_funcs():
    pow = 1
    sc = ps.SpecCalc(NW,pow)
    
    for i in range(len(ev.fs)):
        yield check_each_funcs, sc, i

def check_mult2A(sc,fs,fft_coeffs):

    fmult2 = ev.trunc(fs[0]*fs[1],NW)
    expected = sym2num(sc,fmult2)

    u1h = coeff2wave(sc, fft_coeffs[0])
    u2h = coeff2wave(sc, fft_coeffs[1])
    prod = sc.mult2(u1h, u2h)
    observed = sc.wave2phys(prod)

    testing.assert_allclose(observed,expected)

def test_mult2A():
        fs = ev.fs
        fft_coeffs = ev.fft_coeffs

        pow = 2
        sc = ps.SpecCalc(NW,pow)

        test_comb = [
          [0,1],
          [2,3],
          [1,3]
        ]

        for ij in test_comb:
                i = ij[0]; j = ij[1]
                yield check_mult2A, sc, [fs[i],fs[j]], [fft_coeffs[i], fft_coeffs[j]]

def check_mult2B(sc,fs,fft_coeffs):

        fmult2 = ev.trunc(fs[0]*fs[1],NW)
        expected = sym2num(sc,fmult2)

        u1h = coeff2wave(sc, fft_coeffs[0])
        u2h = coeff2wave(sc, fft_coeffs[1])
        outh = sc.zeros()
        sc.mult2(u1h, u2h, outh)
        observed = sc.wave2phys(outh)

        testing.assert_allclose(observed,expected)
       
def test_mult2B():
        fs = ev.fs
        fft_coeffs = ev.fft_coeffs

        pow = 2
        sc = ps.SpecCalc(NW,pow)

        test_comb = [
          [0,1],
          [2,3],
          [1,3]
        ]

        for ij in test_comb:
                i = ij[0]; j = ij[1]
                yield check_mult2B, sc, [fs[i],fs[j]], [fft_coeffs[i], fft_coeffs[j]]

def check_mult3A(sc,fs,fft_coeffs):

    fmult3 = ev.trunc(fs[0]*fs[1]*fs[2],NW)
    expected = sym2num(sc,fmult3)

    u1h = coeff2wave(sc, fft_coeffs[0])
    u2h = coeff2wave(sc, fft_coeffs[1])
    u3h = coeff2wave(sc, fft_coeffs[2])
    prod = sc.mult3(u1h, u2h, u3h)
    observed = sc.wave2phys(prod)

    testing.assert_allclose(observed,expected)

def test_mult3A():
        fs = ev.fs
        fft_coeffs = ev.fft_coeffs

        pow = 3
        sc = ps.SpecCalc(NW,pow)

        test_comb = [
          [0,1,2],
          [1,2,3],
          [0,1,3]
        ]

        for ijk in test_comb:
                i = ijk[0]; j = ijk[1]; k = ijk[2]
                yield check_mult3A, sc, [fs[i],fs[j],fs[k]], [fft_coeffs[i], fft_coeffs[j], fft_coeffs[k]]

def check_mult3B(sc,fs,fft_coeffs):

    fmult3 = ev.trunc(fs[0]*fs[1]*fs[2],NW)
    expected = sym2num(sc,fmult3)

    u1h = coeff2wave(sc, fft_coeffs[0])
    u2h = coeff2wave(sc, fft_coeffs[1])
    u3h = coeff2wave(sc, fft_coeffs[2])
    outh = sc.zeros()
    sc.mult3(u1h, u2h, u3h, outh)
    observed = sc.wave2phys(outh)

    testing.assert_allclose(observed,expected)


def test_mult3B():
        fs = ev.fs
        fft_coeffs = ev.fft_coeffs

        pow = 3
        sc = ps.SpecCalc(NW,pow)

        test_comb = [
          [0,1,2],
          [1,2,3],
          [0,1,3]
        ]

        for ijk in test_comb:
                i = ijk[0]; j = ijk[1]; k = ijk[2]
                yield check_mult3B, sc, [fs[i],fs[j],fs[k]], [fft_coeffs[i], fft_coeffs[j], fft_coeffs[k]]
