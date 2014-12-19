import numpy as np
from numpy.testing import assert_allclose, assert_array_less, assert_equal

from .. import MovingAverageVariableSpan, LinearVariableSpan


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
        model1 = Model(span)
        model2 = Model.fixed(span)
        assert_allclose(model1.fit(t, y, dy).predict(tfit),
                        model2.fit(t, y, dy).predict(tfit))

    for Model in [MovingAverageVariableSpan, LinearVariableSpan]:
        for span in [0.05, 0.2, 0.5]:
            yield check_model, Model, span


def test_variable_sine_crossval():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    tfit = np.linspace(1, 5.3, 50)
    ytrue = np.sin(tfit)

    def check_model(Model, span):
        model1 = Model(span)
        model2 = Model.fixed(span)
        assert_allclose(model1.fit(t, y, dy).cv_values(),
                        model2.fit(t, y, dy).cv_values())

    for Model in [MovingAverageVariableSpan, LinearVariableSpan]:
        for span in [0.05, 0.2, 0.5]:
            yield check_model, Model, span
