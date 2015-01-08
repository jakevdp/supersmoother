import numpy as np
from numpy.testing import assert_allclose, assert_array_less, assert_equal

from .. import SuperSmoother


def make_sine(N=100, err=0.05, rseed=None):
    rng = np.random.RandomState(rseed)
    t = 2 * np.pi * rng.rand(N)
    y = np.sin(t) + err * rng.randn(N)
    return t, y, err


def make_linear(N=100, err=1E-6, rseed=None):
    rng = np.random.RandomState(rseed)
    t = 10 * rng.rand(N)
    y = t + err * rng.randn(N)
    return t, y, err


def test_sine():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    tfit = np.linspace(1, 5.3, 50)
    ytrue = np.sin(tfit)

    model = SuperSmoother().fit(t, y, dy)
    yfit = model.predict(tfit)
    obs_err = np.mean((yfit - ytrue) ** 2)
    assert_array_less(obs_err, 0.001)


def test_sine_cv():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)
    ytrue = np.sin(np.sort(t)[1:-1])

    model = SuperSmoother().fit(t, y, dy)
    yfit = model.cv_values()[1:-1]
    obs_err = np.mean((yfit - ytrue) ** 2)
    assert_array_less(obs_err, 0.001)


def test_line_linear():
    t, y, dy = make_linear(err=1E-6)
    tfit = np.linspace(0, 10, 40)

    model = SuperSmoother().fit(t, y, dy)
    yfit = model.predict(tfit)
    assert_allclose(tfit, yfit, atol=1E-5)
