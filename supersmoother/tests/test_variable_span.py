import numpy as np
from numpy.testing import assert_allclose, assert_array_less, assert_equal

from .. import MovingAverageSmoother, LinearSmoother


def make_sine(N=100, err=0.05, rseed=None):
    rng = np.random.RandomState(rseed)
    t = 2 * np.pi * rng.rand(N)
    y = np.sin(t) + err * rng.randn(N)
    return t, y, err


def test_variable_sine_predict():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    tfit = np.linspace(1, 5.3, 50)
    ytrue = np.sin(tfit)

    def check_model(Model, span):
        model1 = Model(span * np.ones_like(t))
        model2 = Model(span)
        assert_allclose(model1.fit(t, y, dy).predict(tfit),
                        model2.fit(t, y, dy).predict(tfit))

    for Model in [MovingAverageSmoother, LinearSmoother]:
        for span in [0.05, 0.2, 0.5]:
            yield check_model, Model, span


def test_variable_sine_crossval():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    tfit = np.linspace(1, 5.3, 50)
    ytrue = np.sin(tfit)

    def check_model(Model, span):
        model1 = Model(span * np.ones_like(t))
        model2 = Model(span)
        assert_allclose(model1.fit(t, y, dy).cv_values(),
                        model2.fit(t, y, dy).cv_values())

    for Model in [MovingAverageSmoother, LinearSmoother]:
        for span in [0.05, 0.2, 0.5]:
            yield check_model, Model, span


def test_func_span_const():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    def check_model(Model, span, spanfunc):
        model1 = Model(span)
        model2 = Model(spanfunc)
        assert_allclose(model1.fit(t, y, dy).predict(t),
                        model2.fit(t, y, dy).predict(t))

    for Model in [MovingAverageSmoother, LinearSmoother]:
        for span in [0.05, 0.2, 0.5]:
            spanfunc = lambda t, span=span: span * np.ones_like(t)
            yield check_model, Model, span, spanfunc


def test_func_span_variable():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    spanfunc = lambda t, tmax=t.max(): 0.5 + 0.05 * t / tmax
    span = spanfunc(t)

    def check_model(Model, span, spanfunc):
        model1 = Model(span)
        model2 = Model(spanfunc)
        assert_allclose(model1.fit(t, y, dy).predict(t),
                        model2.fit(t, y, dy).predict(t))

    for Model in [MovingAverageSmoother, LinearSmoother]:
        yield check_model, Model, span, spanfunc
