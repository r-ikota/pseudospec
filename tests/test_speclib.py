import pseudospec as ps
N = 20

def test_NtoJ():
    J = ps.NtoJ(N, pow=3)
    assert type(J) == int
