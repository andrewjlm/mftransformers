import numpy as np
from mftransformers.cx import CX

def test_cx():
    big_n, big_p = (10, 10)
    small_p = 5

    data = np.random.normal(size=(big_n, big_p))
    cx = CX(columns=small_p)
    cx.fit(data)
    C = cx.transform(data)

    assert C.shape == (big_n, small_p)
