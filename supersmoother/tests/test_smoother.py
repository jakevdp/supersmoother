import numpy as np
from numpy.testing import assert_allclose, assert_array_less, assert_equal

from .. import MovingAverageSmoother, LinearSmoother


def make_data(N=100, err_factor=0.5, rseed=None):
    rng = np.random.RandomState(rseed)
    x = 2 * rng.rand(N)
    dy = err_factor * x ** 2
    y = np.sin(np.pi * x ** 2) + dy * rng.randn(N)
    return x, y, dy


def make_sine(N=100, err=0.05, rseed=None):
    rng = np.random.RandomState(rseed)
    t = 2 * np.pi * rng.rand(N)
    y = np.sin(t) + err * rng.randn(N)
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

    for Model in [MovingAverageSmoother, LinearSmoother]:
        for span, err in zip(spans, errs):
            yield check_model, Model, span, err


def test_sine_cv():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)
    ytrue = np.sin(np.sort(t)[1:-1])

    def check_model(Model, span, err):
        model = Model(span).fit(t, y, dy)
        yfit = model.cross_validate(ret_y=True)
        obs_err = np.mean((yfit - ytrue) ** 2)
        assert_array_less(obs_err, err)

    spans = [0.05, 0.2, 0.5]
    errs = [0.005, 0.02, 0.1]

    for Model in [MovingAverageSmoother, LinearSmoother]:
        for span, err in zip(spans, errs):
            yield check_model, Model, span, err


def test_line_linear():
    # create data y=t with small amount of error to prevent singular matrices
    rng = np.random.RandomState(0)
    t = 10 * rng.rand(100)
    dy = 1E-6
    y = t + dy * rng.randn(len(t))

    tfit = np.linspace(0, 10, 40)

    def check_model(span):
        model = LinearSmoother(span).fit(t, y, dy)
        yfit = model.predict(tfit)
        print(np.max(abs(yfit - tfit)))
        assert_allclose(tfit, yfit, atol=1E-5)

    for span in [0.05, 0.2, 0.5]:
        yield check_model, span


def test_sine_compare():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    tfit = np.linspace(1, 5.3, 50)
    ytrue = np.sin(tfit)

    def check_model(Model, span):
        model = Model(span).fit(t, y, dy)
        yfit1 = model.predict(tfit, slow=False)
        assert_equal(model._predict_type, 'fast')
        yfit2 = model.predict(tfit, slow=True)
        assert_equal(model._predict_type, 'slow')
        assert_allclose(yfit1, yfit2)

    for Model in (MovingAverageSmoother, LinearSmoother):
        for span in [0.05, 0.2, 0.5]:
            yield check_model, Model, span


def test_sine_compare_cv():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    def check_model(Model, span, err):
        model = Model(span).fit(t, y, dy)
        yfit1 = model.cross_validate(slow=False, ret_y=True)
        assert_equal(model._cv_type, 'fast')
        yfit2 = model.cross_validate(slow=True, ret_y=True)
        assert_equal(model._cv_type, 'slow')
        obs_err = np.mean((yfit1 - yfit2) ** 2)
        assert_array_less(obs_err, err)

    spans = [0.05, 0.2, 0.5]
    errs = [0.005, 0.02, 0.1]

    for Model in [MovingAverageSmoother, LinearSmoother]:
        for span, err in zip(spans, errs):
            yield check_model, Model, span, err
