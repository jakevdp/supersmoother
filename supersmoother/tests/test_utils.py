from .. import utils

import numpy as np
from numpy.testing import assert_allclose


def test_validate_inputs_nosort(N=10, rseed=0):
    """Test input validation without sorting"""
    rng = np.random.RandomState(rseed)
    t, y = rng.rand(2, N)
    dy = 1

    t_, y_, dy_ = utils.validate_inputs(t, y, dy)
    assert_allclose(t, t_)
    assert_allclose(y, y_)
    assert_allclose(dy, dy_)


def test_validate_inputs_sort(N=10, rseed=0):
    """Test input validation with sorting"""
    rng = np.random.RandomState(rseed)
    t, y = rng.rand(2, N)
    dy = 1

    t_, y_, dy_ = utils.validate_inputs(t, y, dy, sort_by=t)
    isort = np.argsort(t)
    assert_allclose(t[isort], t_)
    assert_allclose(y[isort], y_)
    assert_allclose(dy, dy_)


def test_windowed_sum_fixed(N=10, span=5, rseed=0):
    """Test the windowed sum for a fixed-span"""
    rng = np.random.RandomState(rseed)
    data = rng.rand(3, N)

    for subtract_mid in [True, False]:
        assert_allclose(utils.windowed_sum(*data, span=span,
                                           subtract_mid=subtract_mid),
                        utils.windowed_sum(*data, span=span, slow=True,
                                           subtract_mid=subtract_mid))


def test_windowed_sum_variable(N=10, rseed=0):
    """Test the windowed sum for a variable span"""
    rng = np.random.RandomState(rseed)
    span = rng.randint(3, 6, N)
    data = np.random.random((3, N))

    for subtract_mid in [True, False]:
        assert_allclose(utils.windowed_sum(*data, span=span,
                                           subtract_mid=subtract_mid),
                        utils.windowed_sum(*data, span=span, slow=True,
                                           subtract_mid=subtract_mid))
