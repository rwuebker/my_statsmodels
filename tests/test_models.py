import os
import sys
import numpy as np

p = os.getcwd()
if p not in sys.path:
    sys.path.insert(0, p)

from models import OLS


def test_ols():

    rng = np.random.default_rng(12345)
    assert True
