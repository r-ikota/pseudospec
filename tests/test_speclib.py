from pseudospec.speclib import SpecCalc
import numpy as np
from numpy import testing
import sympy as sy
import pseudospec as ps
import tests.expected_values as ev

NW = 15
xsym = sy.var("x", real=True)

fs = ev.fs
fft_coeffs_r = ev.fft_coeffs_r
fft_coeffs_c = ev.fft_coeffs_c


def sym2num(sc, f):
    xnum = sc.get_x()
    _fnum = sy.lambdify(xsym, f, "numpy")(xnum)
    _fnum = np.real(_fnum)
    return _fnum


def coeff2wave(sc, coeff):
    w = sc.get_zero_cwave()
    w[:] = np.asarray(coeff)
    return w


def test_get_zero_rwave():
    sc = SpecCalc(NW)
    shape = (3, 4)

    expected = np.zeros((3, 4, 2 * NW + 1), dtype=np.float64)
    observed = sc.get_zero_rwave(shape)

    testing.assert_allclose(observed, expected)


def test_wconvert_r2c():
    sc = SpecCalc(NW)
    shape = (2, 3)

    expected = sc.get_zero_cwave(shape)
    expected[0, 1][0] = 1.0 + 0.0j
    expected[1, 2][3] = -2.0 + 1.5j

    rwave = sc.get_zero_rwave(shape)
    rwave[0, 1][0] = 1.0
    rwave[1, 2][3] = -2.0
    rwave[1, 2][1 + NW + 2] = 1.5

    observed = sc.wconvert_r2c(rwave)

    testing.assert_allclose(observed, expected)


def test_wconvert_c2r():
    sc = SpecCalc(NW)
    shape = (2, 3)

    expected = sc.get_zero_rwave(shape)
    expected[0, 1][0] = 1.0
    expected[1, 2][3] = -2.0
    expected[1, 2][1 + NW + 2] = 1.5

    cwave = sc.get_zero_cwave(shape)
    cwave[0, 1][0] = 1.0 + 0.0j
    cwave[1, 2][3] = -2.0 + 1.5j

    observed = sc.wconvert_c2r(cwave)

    testing.assert_allclose(observed, expected)


def check_each_funcs(sc, i):
    expected = sym2num(sc, ev.fs[i])
    w = coeff2wave(sc, ev.fft_coeffs_c[i])
    observed = sc.transform_wc2wp(w)

    testing.assert_allclose(observed, expected)


def test_each_funcs():
    sc = ps.SpecCalc(NW)

    for i in range(len(fs)):
        yield check_each_funcs, sc, i


def check_mult2(sc, fs, fft_coeffs_c):

    fmult2 = ev.trunc(fs[0] * fs[1], NW)
    expected = sym2num(sc, fmult2)

    prod_c = sc.mult2(fft_coeffs_c[0], fft_coeffs_c[1])
    observed = sc.transform_wc2wp(prod_c)

    testing.assert_allclose(observed, expected, atol=1.0e-10)


def test_mult2():

    sc = ps.SpecCalc(NW)

    test_combination = [[0, 1], [2, 3], [1, 3]]

    for i, j in test_combination:
        yield check_mult2, sc, [fs[i], fs[j]], [fft_coeffs_c[i], fft_coeffs_c[j]]


def check_mult3A(sc, fs, fft_coeffs):

    fmult3 = ev.trunc(fs[0] * fs[1] * fs[2], NW)
    expected = sym2num(sc, fmult3)

    prod_c = sc.mult3(fft_coeffs[0], fft_coeffs[1], fft_coeffs[2])
    observed = sc.transform_wc2wp(prod_c)

    testing.assert_allclose(observed, expected)


def test_mult3A():

    sc = ps.SpecCalc(NW)

    test_comb = [[0, 1, 2], [1, 2, 3], [0, 1, 3]]

    for ijk in test_comb:
        i = ijk[0]
        j = ijk[1]
        k = ijk[2]
        yield check_mult3A, sc, [fs[i], fs[j], fs[k]], [
            fft_coeffs_c[i],
            fft_coeffs_c[j],
            fft_coeffs_c[k],
        ]


def test_mult3B():
    sc = ps.SpecCalc(NW)

    shape = (2, 3)
    wc1 = sc.get_zero_cwave(shape)
    wc2 = sc.get_zero_cwave(shape)
    wc3 = sc.get_zero_cwave(shape)
    expected = sc.get_zero_pwave(shape)
    for i, j in np.ndindex(shape):
        idx0 = i
        idx1 = i + 2
        idx2 = j + 1
        fmult3 = ev.trunc(fs[idx0] * fs[idx1] * fs[idx2], NW)
        expected[i, j, :] = sym2num(sc, fmult3)

        wc1[i, j, :] = fft_coeffs_c[idx0]
        wc2[i, j, :] = fft_coeffs_c[idx1]
        wc3[i, j, :] = fft_coeffs_c[idx2]
    prod_c = sc.mult3(wc1, wc2, wc3)
    observed = sc.transform_wc2wp(prod_c)

    testing.assert_allclose(observed, expected)


def check_sdiff1A(sc, f, fft_coeff):
    _df1 = f.diff(xsym)
    expected = sym2num(sc, _df1)

    diff_c = sc.sdiff1(fft_coeff)
    observed = sc.transform_wc2wp(diff_c)

    testing.assert_allclose(observed, expected)


def test_sdiff1A():
    sc = ps.SpecCalc(NW)

    for i in range(len(fs)):
        yield check_sdiff1A, sc, fs[i], fft_coeffs_c[i]


def check_sdiff2A(sc, f, fft_coeff):
    _df2 = f.diff(xsym, 2)
    expected = sym2num(sc, _df2)

    diff_c = sc.sdiff2(fft_coeff)
    observed = sc.transform_wc2wp(diff_c)

    testing.assert_allclose(observed, expected)


def test_sdiff2A():
    sc = ps.SpecCalc(NW)

    for i in range(len(fs)):
        yield check_sdiff2A, sc, fs[i], fft_coeffs_c[i]
