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


def make_linear(N=100, err=1E-6, rseed=None):
    rng = np.random.RandomState(rseed)
    t = 10 * rng.rand(N)
    y = t + err * rng.randn(N)
    return t, y, err


def test_constant_data(N=100, rseed=0):
    """Test linear & moving average smoothers on constant data"""
    rng = np.random.RandomState(rseed)
    t = 10 * rng.rand(N)
    y = 1.0
    dy = 1.0
    span = 5

    for method in [utils.moving_average_smooth,
                   utils.linear_smooth]:
        yfit = method(t, y, dy, span=5)
        assert_allclose(y, yfit)


def test_equal_spaced_linear_data(N=100, rseed=0):
    """Test linear & moving average smoothers on equally-spaced linear data"""
    t = np.linspace(0, 10, N)
    y = 1 + 2 * t
    dy = 1
    span = 5

    for method in [utils.moving_average_smooth,
                   utils.linear_smooth]:
        yfit = method(t, y, dy, span=5)
        assert_allclose(y[3:-3], yfit[3:-3])


def test_random_linear_data(N=100, rseed=0):
    """Test linear smoother on linear data"""
    rng = np.random.RandomState(rseed)
    t = 10 * rng.rand(N)
    y = 1 + 2 * t
    dy = 1.0
    span = 5

    yfit = utils.linear_smooth(t, y, dy, span=5)
    assert_allclose(y, yfit)
