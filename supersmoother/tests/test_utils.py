from .. import utils

import numpy as np
from numpy.testing import assert_allclose, assert_, assert_equal, assert_raises


def test_iterable():
    x = [1, 2, 3]
    y = 1
    assert_(utils.iterable(x))
    assert_(not utils.iterable(y))


def test_multiinterp(rseed=0, N=100):
    """Test of the multinterp function"""
    rng = np.random.RandomState(rseed)

    def check_match(k):
        x = np.sort(rng.rand(k))
        y = rng.rand(k, N)
        xquery = rng.rand(N)

        res1 = utils.multinterp(x, y, xquery, slow=False)
        res2 = utils.multinterp(x, y, xquery, slow=True)
        print(abs(res1 - res2))
        assert_allclose(res1, res2)

    for k in 3, 4, 5:
        yield check_match, k


def test_setattr_context():
    """Test of the setattr_context() function"""
    class Foo(object):
        pass
    f = Foo()
    f.attr = "abc"
    with utils.setattr_context(f, attr="123"):
        assert_equal(f.attr, "123")
    assert_equal(f.attr, "abc")

    try:
        with utils.setattr_context(f, attr="123"):
            raise ValueError()
    except:
        pass
    assert_equal(f.attr, "abc")


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


def test_validate_inputs_fail():
    rng = np.random.RandomState(0)
    t, y = rng.rand(2, 10)
    dy = 1

    # bad keyword argument
    assert_raises(ValueError, utils.validate_inputs, t, y, dy,
                  blah='blah')

    # non-sortable array
    assert_raises(ValueError, utils.validate_inputs, 1, 1, 1)

    # bad sort array
    assert_raises(ValueError, utils.validate_inputs, t, y, dy, sort_by=[1])


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


def test_windowed_sum_combinations(N=10, rseed=0):
    """test passing combinations of parameters to the windowed_sum function"""
    rng = np.random.RandomState(rseed)
    data = np.random.random((3, N))

    def check_result(indices, span, subtract_mid):
        assert_allclose(utils.windowed_sum(*data, span=span, indices=indices,
                                           subtract_mid=subtract_mid),
                        utils.windowed_sum(*data, span=span, indices=indices,
                                           subtract_mid=subtract_mid,
                                           slow=True))

    for indices in [None, rng.randint(0, N, N)]:
        for span in [5, rng.randint(3, 6, N)]:
            for subtract_mid in [True, False]:
                yield check_result, indices, span, subtract_mid


def test_windowed_sum_indices(N=10, rseed=0):
    """Test case where indices and span correspond"""
    rng = np.random.RandomState(rseed)
    data = np.random.random((3, N))
    indices = rng.randint(0, N, N - 2)
    span = 5

    def check_result(subtract_mid):
        assert_allclose(utils.windowed_sum(*data, span=span, indices=indices,
                                           subtract_mid=subtract_mid),
                        utils.windowed_sum(*data, span=span, indices=indices,
                                           subtract_mid=subtract_mid,
                                           slow=True))

    for subtract_mid in [True, False]:
        yield check_result, subtract_mid


def test_windowed_sum_bad_kwargs():
    rng = np.random.RandomState(0)
    span = rng.randint(3, 6, 10)
    data = np.random.random((3, 10))

    # no span specified
    assert_raises(ValueError, utils.windowed_sum, *data)

    # non-positive span
    assert_raises(ValueError, utils.windowed_sum, *data, span=0)

    # nonsense keyword argument
    assert_raises(ValueError, utils.windowed_sum, *data, gobbledeygook='yay')

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


def test_perfect_sine(N=100, rseed=0):
    """Test approximate recovery of a sine wave with no noise"""
    t = np.linspace(0, 10, 200)
    y = np.sin(t)
    dy = 1

    for method in [utils.moving_average_smooth, utils.linear_smooth]:
        yfit = method(t, y, dy, span=5)
        assert_allclose(y[3:-3], yfit[3:-3], atol=0.005)


def test_constant_data(N=100, rseed=0):
    """Test linear & moving average smoothers on constant data"""
    rng = np.random.RandomState(rseed)
    t = 10 * rng.rand(N)
    y = 1.0
    dy = 1.0
    span = 5

    for method in [utils.moving_average_smooth, utils.linear_smooth]:
        yfit = method(t, y, dy, span=5)
        assert_allclose(y, yfit)


def test_equal_spaced_linear_data(N=100, rseed=0):
    """Test linear & moving average smoothers on equally-spaced linear data"""
    t = np.linspace(0, 10, N)
    y = 1 + 2 * t
    dy = 1
    span = 5

    for method in [utils.moving_average_smooth, utils.linear_smooth]:
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


def test_t_vs_t_out(N=100, rseed=0):
    """Test two means of arriving at the same answer"""
    rng = np.random.RandomState(rseed)
    t = np.sort(10 * rng.rand(N))
    y = np.sin(t)
    dy = 1.0
    span = 5

    for method in [utils.linear_smooth, utils.moving_average_smooth]:
        yfit1 = method(t, y, dy, span=5, cv=False)
        yfit2 = method(t, y, dy, span_out=5, t_out=t, cv=False)
        assert_allclose(yfit1, yfit2)


def test_smooth_assertions():
    """Test errors and assertions in linear_smooth and moving_average_smooth"""
    rng = np.random.RandomState(0)
    t = 10 * rng.rand(10)
    y = np.sin(t)
    dy = 1.0

    for method in [utils.linear_smooth, utils.moving_average_smooth]:
        # No span set
        assert_raises(ValueError, method, t, y, dy)

        # Both spans set
        assert_raises(ValueError, method, t, y, dy, span=1,
                      t_out=t, span_out=1)

        # span_out set without t_out
        assert_raises(ValueError, method, t, y, dy, span_out=1)
