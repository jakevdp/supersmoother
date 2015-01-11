from __future__ import division, print_function

from contextlib import contextmanager

import numpy as np


@contextmanager
def setattr_context(obj, **kwargs):
    """
    Context manager to temporarily change the values of object attributes
    while executing a function.

    Example
    -------
    >>> class Foo: pass
    >>> f = Foo(); f.attr = 'hello'
    >>> with setattr_context(f, attr='goodbye'):
    ...     print(f.attr)
    goodbye
    >>> print(f.attr)
    hello
    """
    old_kwargs = dict([(key, getattr(obj, key)) for key in kwargs])
    [setattr(obj, key, val) for key, val in kwargs.items()]
    try:
        yield
    finally:
        [setattr(obj, key, val) for key, val in old_kwargs.items()]


def iterable(obj):
    """Utility to check if object is iterable"""
    try:
        iter(obj)
    except:
        return False
    else:
        return True


def validate_inputs(*arrays, **kwargs):
    """Validate input arrays

    This checks that
    - Arrays are mutually broadcastable
    - Broadcasted arrays are one-dimensional

    Optionally, arrays are sorted according to the ``sort_by`` argument.

    Parameters
    ----------
    *args : ndarrays
        All non-keyword arguments are arrays which will be validated
    sort_by : array
        If specified, sort all inputs by the order given in this array.
    """
    arrays = np.broadcast_arrays(*arrays)
    sort_by = kwargs.pop('sort_by', None)

    if kwargs:
        raise ValueError("unrecognized arguments: {0}".format(kwargs.keys()))

    if arrays[0].ndim != 1:
        raise ValueError("Input arrays should be one-dimensional.")

    if sort_by is not None:
        isort = np.argsort(sort_by)
        if isort.shape != arrays[0].shape:
            raise ValueError("sort shape must equal array shape.")
        arrays = tuple([a[isort] for a in arrays])
    return arrays


def windowed_sum(*arrays, **kwargs):
    """Compute the windowed sum of the given arrays

    Parameters
    ----------
    *args : array_like
        Each additional argument is an array that will be windowed
    span : int or array of ints
        The span to use for the sum at each point
    subtract_mid : boolean
        If true, then subtract the middle value from each sum
    slow : boolean
        If true, use a slow method to compute the result. This is
        primarily useful for testing and validation.
    indices : array
        the indices of the center of the desired windows. If provided,
        it must be broadcastable with the span.

    Returns
    -------
    arrays : tuple of ndarrays
        arrays containing the windowed sum of each input array
    """
    span = kwargs.pop('span', None)
    subtract_mid = kwargs.pop('subtract_mid', False)
    slow = kwargs.pop('slow', False)
    indices = kwargs.pop('indices', None)

    if kwargs:
        raise ValueError("Unrecognized keywords: {0}".format(kwargs.keys()))

    if span is None:
        raise ValueError("Must provide a positive integer span")

    span = np.asarray(span, dtype=int)
    if not np.all(span > 0):
        raise ValueError("span values must be positive")

    if indices is not None:
        span, indices = np.broadcast_arrays(span, indices)

    if slow:
        # Slow fixed/variable span
        results = []
        for array in map(np.asarray, arrays):
            assert array.ndim == 1
            span, array = np.broadcast_arrays(span, array)
            if indices is None:
                ind_spans = enumerate(span)
            else:
                ind_spans = zip(indices, span)
            result = (array[max(0, i - s // 2): i - s // 2 + s].sum()
                      for i, s in ind_spans)
            results.append(np.fromiter(result,
                                       dtype=array.dtype,
                                       count=len(array)))
    elif span.ndim == 0:
        # Fast fixed-span
        window = np.ones(span)        
        def convolve_same(a, window):
            res = np.convolve(a, window, mode='same')
            if len(res) > len(a):
                res = res[len(res) // 2 - len(a) // 2:][:len(a)]
            return res
        results = [convolve_same(a, window) for a in arrays]

    else:
        # Fast variable-span
        if indices is None:
            indices = np.arange(len(span))
        mins = np.asarray(indices) - span // 2

        results = []
        for a in map(np.asarray, arrays):
            ranges = np.vstack([np.maximum(0, mins),
                                np.minimum(len(a), mins + span)]).ravel('F')
            results.append(np.add.reduceat(np.append(a, 0), ranges)[::2])

    if subtract_mid:
        if indices is None:
            indices = slice(None)
        results = (r - a[indices] for r, a in zip(results, arrays))

    return tuple(results)


def moving_average_smooth(t, y, dy, span=None, cv=True,
                          t_out=None, span_out=None):
    """Perform a moving-average smooth of the data

    Parameters
    ----------
    t, y, dy : array_like
        time, value, and error in value of the input data
    span : array_like
        the integer spans of the data
    cv : boolean (default=True)
        if True, treat the problem as a cross-validation, i.e. don't use
        each point in the evaluation of its own smoothing.
    t_out : array_like (optional)
        the output times for the moving averages
    span_out : array_like (optional)
        the spans associated with the output times t_out

    Returns
    -------
    y_smooth : array_like
        smoothed y values at each time t (or t_out)
    """
    t, y, dy = validate_inputs(t, y, dy, sort_by=t)

    if span_out is not None:
        if t_out is None:
            raise ValueError("Must specify t_out when span_out is given")
        if span is not None:
            raise ValueError("Must specify only one of span, span_out")
        span, t_out = np.broadcast_arrays(span_out, t_out)
        indices = np.searchsorted(t, t_out)
    elif span is None:
        raise ValueError("Must specify either span_out or span")
    else:
        indices = None

    w = dy ** -2
    w, yw = windowed_sum(w, y * w, span=span, subtract_mid=cv,
                         indices=indices)

    if t_out is None or span_out is not None:
        return yw / w
    else:
        i = np.minimum(len(t) - 1, np.searchsorted(t, t_out))
        return yw[i] / w[i]


def linear_smooth(t, y, dy, span=None, cv=True,
                  t_out=None, span_out=None):
    """Perform a linear smooth of the data

    Parameters
    ----------
    t, y, dy : array_like
        time, value, and error in value of the input data
    span : array_like
        the integer spans of the data
    cv : boolean (default=True)
        if True, treat the problem as a cross-validation, i.e. don't use
        each point in the evaluation of its own smoothing.
    t_out : array_like (optional)
        the output times for the moving averages
    span_out : array_like (optional)
        the spans associated with the output times t_out

    Returns
    -------
    y_smooth : array_like
        smoothed y values at each time t or t_out
    """
    t_input = t
    t, y, dy = validate_inputs(t, y, dy, sort_by=t)

    if span_out is not None:
        if t_out is None:
            raise ValueError("Must specify t_out when span_out is given")
        if span is not None:
            raise ValueError("Must specify only one of span, span_out")
        span, t_out = np.broadcast_arrays(span_out, t_out)
        indices = np.searchsorted(t, t_out)
    elif span is None:
        raise ValueError("Must specify either span_out or span")
    else:
        indices = None

    w = dy ** -2
    w, tw, yw, ttw, tyw = windowed_sum(w, t * w, y * w, t * t * w, t * y * w,
                                       span=span, indices=indices,
                                       subtract_mid=cv)

    denominator = (w * ttw - tw * tw)
    slope = (tyw * w - tw * yw)
    intercept = (ttw * yw - tyw * tw)

    if t_out is None:
        return (slope * t_input + intercept) / denominator
    elif span_out is not None:
        return (slope * t_out + intercept) / denominator
    else:
        i = np.minimum(len(t) - 1, np.searchsorted(t, t_out))
        return (slope[i] * t_out + intercept[i]) / denominator[i]


def multinterp(x, y, xquery, slow=False):
    """Multiple linear interpolations

    Parameters
    ----------
    x : array_like, shape=(N,)
        sorted array of x values
    y : array_like, shape=(N, M)
        array of y values corresponding to each x value
    xquery : array_like, shape=(M,)
        array of query values
    slow : boolean, default=False
        if True, use slow method (used mainly for unit testing)

    Returns
    -------
    yquery : ndarray, shape=(M,)
        The interpolated values corresponding to each x query.
    """
    x, y, xquery = map(np.asarray, (x, y, xquery))
    assert x.ndim == 1
    assert xquery.ndim == 1
    assert y.shape == x.shape + xquery.shape

    # make sure xmin < xquery < xmax in all cases
    xquery = np.clip(xquery, x.min(), x.max())

    if slow:
        from scipy.interpolate import interp1d
        return np.array([interp1d(x, y)(xq) for xq, y in zip(xquery, y.T)])
    elif len(x) == 3:
        # Most common case: use a faster approach
        yq_lower = y[0] + (xquery - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
        yq_upper = y[1] + (xquery - x[1]) * (y[2] - y[1]) / (x[2] - x[1])
        return np.where(xquery < x[1], yq_lower, yq_upper)
    else:
        i = np.clip(np.searchsorted(x, xquery, side='right') - 1,
                    0, len(x) - 2)
        j = np.arange(len(xquery))
        return y[i, j] + ((xquery - x[i]) *
                          (y[i + 1, j] - y[i, j]) / (x[i + 1] - x[i]))
