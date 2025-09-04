import numpy as np
from numpy.testing import (assert_allclose, assert_array_less,
                           assert_equal, assert_raises)
import pytest

from .. import MovingAverageSmoother, LinearSmoother
from ..smoother import Smoother


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


@pytest.mark.parametrize("Model", [MovingAverageSmoother, LinearSmoother])
@pytest.mark.parametrize("span, err", [(0.05, 0.005), (0.2, 0.01), (0.5, 0.1)])
def test_sine(Model, span, err):
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    tfit = np.linspace(1, 5.3, 50)
    ytrue = np.sin(tfit)

    model = Model(span).fit(t, y, dy)
    yfit = model.predict(tfit)
    obs_err = np.mean((yfit - ytrue) ** 2)
    assert_array_less(obs_err, err)


@pytest.mark.parametrize("Model", [MovingAverageSmoother, LinearSmoother])
@pytest.mark.parametrize("span, err", [(0.05, 0.005), (0.2, 0.02), (0.5, 0.1)])
def test_sine_cv(Model, span, err):
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)
    ytrue = np.sin(np.sort(t)[1:-1])

    model = Model(span).fit(t, y, dy)
    yfit = model.cv_values()[1:-1]
    obs_err = np.mean((yfit - ytrue) ** 2)
    assert_array_less(obs_err, err)


@pytest.mark.parametrize("span", [0.05, 0.2, 0.5])
def test_line_linear(span):
    t, y, dy = make_linear(rseed=0, err=1E-6)
    tfit = np.linspace(0, 10, 40)

    model = LinearSmoother(span).fit(t, y, dy)
    yfit = model.predict(tfit)
    assert_allclose(tfit, yfit, atol=1E-5)


@pytest.mark.parametrize("Model", [MovingAverageSmoother, LinearSmoother])
@pytest.mark.parametrize("span", [0.05, 0.2, 0.5])
def test_variable_sine_predict(Model, span):
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    tfit = np.linspace(1, 5.3, 50)
    ytrue = np.sin(tfit)

    model1 = Model(span * np.ones_like(t))
    model2 = Model(span)
    assert_allclose(model1.fit(t, y, dy).predict(tfit),
                    model2.fit(t, y, dy).predict(tfit))


@pytest.mark.parametrize("Model", [MovingAverageSmoother, LinearSmoother])
@pytest.mark.parametrize("span", [0.05, 0.2, 0.5])
def test_variable_sine_crossval(Model, span):
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    tfit = np.linspace(1, 5.3, 50)
    ytrue = np.sin(tfit)

    model1 = Model(span * np.ones_like(t))
    model2 = Model(span)
    assert_allclose(model1.fit(t, y, dy).cv_values(),
                    model2.fit(t, y, dy).cv_values())


@pytest.mark.parametrize("Model", [MovingAverageSmoother, LinearSmoother])
@pytest.mark.parametrize("span", [0.05, 0.2, 0.5])
def test_func_span_const(Model, span):
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    spanfunc = lambda t, span=span: span * np.ones_like(t)

    model1 = Model(span)
    model2 = Model(spanfunc)
    assert_allclose(model1.fit(t, y, dy).predict(t),
                    model2.fit(t, y, dy).predict(t))


@pytest.mark.parametrize("Model", [MovingAverageSmoother, LinearSmoother])
def test_func_span_variable(Model):
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    spanfunc = lambda t, tmax=t.max(): 0.5 + 0.05 * t / tmax
    span = spanfunc(t)

    model1 = Model(span)
    model2 = Model(spanfunc)
    assert_allclose(model1.fit(t, y, dy).predict(t),
                    model2.fit(t, y, dy).predict(t))


@pytest.mark.parametrize("Model", [MovingAverageSmoother, LinearSmoother])
@pytest.mark.parametrize("cv", [True, False])
@pytest.mark.parametrize("skip_endpoints", [True, False])
def test_cv_interface(Model, cv, skip_endpoints):
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)
    span = 0.2

    model = Model(span).fit(t, y, dy)
    cv_values = model.cv_values(cv=cv)
    cv_residuals = (model.y - cv_values) / model.dy
    if skip_endpoints:
        cv_error = np.mean(abs(cv_residuals[1:-1]))
    else:
        cv_error = np.mean(abs(cv_residuals))

    assert_allclose(cv_residuals,
                    model.cv_residuals(cv=cv))
    assert_allclose(cv_error,
                    model.cv_error(cv=cv, skip_endpoints=skip_endpoints))


@pytest.mark.parametrize("Model", [MovingAverageSmoother, LinearSmoother])
@pytest.mark.parametrize("span", (-1, 0, 1, 2))
def test_span_extremes(Model, span):
    """Test models with extreme values of spans"""
    t, y, dy = make_sine(10, err=0.05, rseed=0)

    model1 = Model(span).fit(t, y, dy)
    model2 = Model(span + np.zeros_like(t)).fit(t, y, dy)
    assert_allclose(model1.cv_values(False), model2.cv_values(False))
    

@pytest.mark.parametrize("Model", [MovingAverageSmoother, LinearSmoother])
@pytest.mark.parametrize("span", [3, 4, 5])
def test_periodic(Model, span):
    rng = np.random.RandomState(0)
    N = 10
    period = 1
    t = np.linspace(0, period, N, endpoint=False)
    y = rng.rand(N)

    t_folded = np.concatenate([-period + t, t, t + period])
    y_folded = np.concatenate([y, y, y])

    model1 = Model(span / len(t_folded), period=None)
    model2 = Model(span / len(t), period=period)

    model1.fit(t_folded, y_folded)
    model2.fit(t, y)

    assert_allclose(model1.predict(t), model2.predict(t))


def test_baseclass():
    # silly tests to bring coverage to 100%
    assert_raises(NotImplementedError, Smoother)
    class Derived(Smoother):
        def __init__(self):
            pass

    d = Derived()
    assert_raises(NotImplementedError, d.fit, [1, 2, 3], [1, 1, 1])
    assert_raises(NotImplementedError, d.predict, [1, 2, 3])
    assert_raises(NotImplementedError, d.cv_values)


def test_duplicate_inputs():
    rng = np.random.RandomState(0)
    x = np.arange(100)
    x[:10] = 10
    y = rng.rand(100)
    sm = LinearSmoother(span=0.1).fit(x, y)
    assert_raises(ValueError, sm.predict, x)
