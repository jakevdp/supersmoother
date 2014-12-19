import numpy as np
from numpy.testing import assert_allclose, assert_array_less, assert_equal

from .. import MovingAverageFixedSpan, LinearFixedSpan


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

    def check_model(Model, span, err):
        model = Model(span).fit(t, y, dy)
        yfit = model.predict(tfit)
        obs_err = np.mean((yfit - ytrue) ** 2)
        assert_array_less(obs_err, err)

    spans = [0.05, 0.2, 0.5]
    errs = [0.005, 0.01, 0.1]

    for Model in [MovingAverageFixedSpan, LinearFixedSpan]:
        for span, err in zip(spans, errs):
            yield check_model, Model, span, err


def test_sine_cv():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)
    ytrue = np.sin(np.sort(t)[1:-1])

    def check_model(Model, span, err):
        model = Model(span).fit(t, y, dy)
        yfit = model.cv_values(imin=1, imax=-1)
        obs_err = np.mean((yfit - ytrue) ** 2)
        assert_array_less(obs_err, err)

    spans = [0.05, 0.2, 0.5]
    errs = [0.005, 0.02, 0.1]

    for Model in [MovingAverageFixedSpan, LinearFixedSpan]:
        for span, err in zip(spans, errs):
            yield check_model, Model, span, err


def test_line_linear():
    t, y, dy = make_linear(err=1E-6)
    tfit = np.linspace(0, 10, 40)

    def check_model(span):
        model = LinearFixedSpan(span).fit(t, y, dy)
        yfit = model.predict(tfit)
        assert_allclose(tfit, yfit, atol=1E-5)

    for span in [0.05, 0.2, 0.5]:
        yield check_model, span


def test_sine_compare():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    tfit = np.linspace(1, 5.3, 50)
    ytrue = np.sin(tfit)

    def check_model(Model, span):
        model1 = Model(span)
        model2 = Model.slow(span)
        assert_allclose(model1.fit(t, y, dy).predict(tfit),
                        model2.fit(t, y, dy).predict(tfit))

    for Model in (MovingAverageFixedSpan, LinearFixedSpan):
        for span in [0.05, 0.2, 0.5]:
            yield check_model, Model, span


def test_sine_compare_cv():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    tfit = np.linspace(1, 5.3, 50)
    ytrue = np.sin(tfit)

    def check_model(Model, span):
        model1 = Model(span)
        model2 = Model.slow(span)
        assert_allclose(model1.fit(t, y, dy).cv_values(),
                        model2.fit(t, y, dy).cv_values())

    for Model in (MovingAverageFixedSpan, LinearFixedSpan):
        for span in [0.05, 0.2, 0.5]:
            yield check_model, Model, span
