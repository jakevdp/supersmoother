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


def windowed_sum_slow(arrays, span, t=None, indices=None, tpowers=0,
                      period=None, subtract_mid=False):
    """Compute the windowed sum of the given arrays.

    This is a slow function, used primarily for testing and validation
    of the faster version of ``windowed_sum()``

    Parameters
    ----------
    arrays : tuple of arrays
        arrays to window
    span : int or array of ints
        The span to use for the sum at each point. If array is provided,
        it must be broadcastable with ``indices``
    indices : array
        the indices of the center of the desired windows. If ``None``,
        the indices are assumed to be ``range(len(arrays[0]))`` though
        these are not actually instantiated.
    t : array (optional)
        Times associated with the arrays
    tpowers : list (optional)
        Powers of t for each array sum
    period : float (optional)
        Period to use, if times are periodic. If supplied, input times
        must be arranged such that (t % period) is sorted!
    subtract_mid : boolean
        If true, then subtract the middle value from each sum

    Returns
    -------
    arrays : tuple of ndarrays
        arrays containing the windowed sum of each input array
    """
    span = np.asarray(span, dtype=int)
    if not np.all(span > 0):
        raise ValueError("span values must be positive")
    
    arrays = tuple(map(np.asarray, arrays))
    N = arrays[0].size
    if not all(a.shape == (N,) for a in arrays):
        raise ValueError("sizes of provided arrays must match")
    
    t_input = t
    if t is not None:
        t = np.asarray(t)
        if t.shape != (N,):
            raise ValueError("shape of t must match shape of arrays")
    else:
        t = np.ones(N)
    
    tpowers = tpowers + np.zeros(len(arrays))
    if len(tpowers) != len(arrays):
        raise ValueError("tpowers must be broadcastable with number of arrays")

    if period:
        if t_input is None:
            raise ValueError("periodic requires t to be provided")
        t = t % period
    
    if indices is None:
        indices = np.arange(N)
    spans, indices = np.broadcast_arrays(span, indices)
    
    results = []
    for tpower, array in zip(tpowers, arrays):
        if period:
            result = [sum(array[j % N]
                          * (t[j % N] + (j // N) * period) ** tpower
                          for j in range(i - s // 2,
                                         i - s // 2 + s)
                          if not (subtract_mid and j == i))
                      for i, s in np.broadcast(indices, spans)]
        else:
            result = [sum(array[j] * t[j] ** tpower
                          for j in range(max(0, i - s // 2),
                                         min(N, i - s // 2 + s))
                          if not (subtract_mid and j == i))
                      for i, s in np.broadcast(indices, spans)]
        results.append(np.asarray(result))
        
    return tuple(results)


def windowed_sum(arrays, span, t=None, indices=None, tpowers=0,
                 period=None, subtract_mid=False):
    """Compute the windowed sum of the given arrays.

    Parameters
    ----------
    arrays : tuple of arrays
        arrays to window
    span : int or array of ints
        The span to use for the sum at each point. If array is provided,
        it must be broadcastable with ``indices``
    indices : array
        the indices of the center of the desired windows. If ``None``,
        the indices are assumed to be ``range(len(arrays[0]))`` though
        these are not actually instantiated.
    t : array (optional)
        Times associated with the arrays
    tpowers : list (optional)
        Powers of t for each array sum
    period : float (optional)
        Period to use, if times are periodic. If supplied, input times
        must be arranged such that (t % period) is sorted!
    subtract_mid : boolean
        If true, then subtract the middle value from each sum

    Returns
    -------
    arrays : tuple of ndarrays
        arrays containing the windowed sum of each input array
    """
    span = np.asarray(span, dtype=int)
    if not np.all(span > 0):
        raise ValueError("span values must be positive")
    
    arrays = tuple(map(np.asarray, arrays))
    N = arrays[0].size
    if not all(a.shape == (N,) for a in arrays):
        raise ValueError("sizes of provided arrays must match")
    
    t_input = t
    if t is not None:
        t = np.asarray(t)
        if t.shape != (N,):
            raise ValueError("shape of t must match shape of arrays "
                             "t -> {0} arr -> {1}".format(t.shape,
                                                          arrays[0].shape))
    else:
        # XXX: special-case no t?
        t = np.ones(N)
    
    tpowers = tpowers + np.zeros(len(arrays))
    if len(tpowers) != len(arrays):
        raise ValueError("tpowers must be broadcastable with number of arrays")

    if indices is not None:
        span, indices = np.broadcast_arrays(span, indices)

    # For the periodic case, re-call the function with padded arrays
    if period:
        if t_input is None:
            raise ValueError("periodic requires t to be provided")
        t = t % period

        t, arrays, sl = _pad_arrays(t, arrays, indices, span, period)
        if len(t) > N:
            # arrays are padded. Recursively call windowed_sum() and return.
            if span.ndim == 0 and indices is None:
                # fixed-span/no index case is done faster this way
                arrs = windowed_sum(arrays, span, t=t, indices=indices,
                                    tpowers=tpowers, period=None,
                                    subtract_mid=subtract_mid)
                return tuple([a[sl] for a in arrs])
            else:
                # this works for variable span and general indices
                if indices is None:
                    indices = np.arange(N)
                indices = indices + sl.start
                return windowed_sum(arrays, span, t=t, indices=indices,
                                    tpowers=tpowers, period=None,
                                    subtract_mid=subtract_mid)
        else:
            # No padding needed! We can carry-on as if it's a non-periodic case
            period = None

    # The rest of the algorithm now proceeds without reference to the period
    # just as a sanity check...
    assert period is None

    if span.ndim == 0:
        # fixed-span case. Because of the checks & manipulations above
        # we know here that indices=None
        assert indices is None
        window = np.ones(span)

        def convolve_same(a, window):
            if len(window) <= len(a):
                res = np.convolve(a, window, mode='same')
            else:
                res = np.convolve(a, window, mode='full')
                start = (len(window) - 1) // 2
                res = res[start:start + len(a)]
            return res
        results = [convolve_same(a * t ** tp, window)
                   for a, tp in zip(arrays, tpowers)]
        indices = slice(None)

    else:
        # variable-span case. Use reduceat() in a clever way for speed.
        if indices is None:
            indices = np.arange(len(span))

        # we checked this above, but just as a sanity check assert it here...
        assert span.shape == indices.shape

        mins = np.asarray(indices) - span // 2
        results = []
        for a, tp in zip(arrays, tpowers):
            ranges = np.vstack([np.maximum(0, mins),
                                np.minimum(len(a), mins+span)]).ravel('F')
            results.append(np.add.reduceat(np.append(a * t ** tp, 0),
                                           ranges)[::2])

    # Subtract the midpoint if required: this is used in cross-validation
    if subtract_mid:
        results = [r - a[indices] * t[indices] ** tp
                   for r, a, tp in zip(results, arrays, tpowers)]

    return tuple(results)


def _pad_arrays(t, arrays, indices, span, period):
    """Internal routine to pad arrays for periodic models."""
    N = len(t)

    if indices is None:
        indices = np.arange(N)
    pad_left = max(0, 0 - np.min(indices - span // 2))
    pad_right = max(0, np.max(indices + span - span // 2) - (N - 1))

    if pad_left + pad_right > 0:
        Nright, pad_right = divmod(pad_right, N)
        Nleft, pad_left = divmod(pad_left, N)
        t = np.concatenate([t[N - pad_left:] - (Nleft + 1) * period]
                           + [t + i * period
                              for i in range(-Nleft, Nright + 1)]
                           + [t[:pad_right] + (Nright + 1) * period])
        arrays = [np.concatenate([a[N - pad_left:]]
                                 + (Nleft + Nright + 1) * [a]
                                 + [a[:pad_right]])
                  for a in arrays]
        pad_left = pad_left % N
        Nright = pad_right / N
        pad_right = pad_right % N

        return (t, arrays, slice(pad_left + Nleft * N,
                                 pad_left + (Nleft + 1) * N))
    else:
        return (t, arrays, slice(None))


def _prep_smooth(t, y, dy, span, t_out, span_out, period):
    """Private function to prepare & check variables for smooth utilities"""

    # If period is provided, sort by phases. Otherwise sort by t
    if period:
        t = t % period
        if t_out is not None:
            t_out = t_out % period

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

    return t, y, dy, span, t_out, span_out, indices


def moving_average_smooth(t, y, dy, span=None, cv=True,
                          t_out=None, span_out=None, period=None):
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
    period : float
        if provided, then consider the inputs periodic with the given period

    Returns
    -------
    y_smooth : array_like
        smoothed y values at each time t (or t_out)
    """
    prep = _prep_smooth(t, y, dy, span, t_out, span_out, period)
    t, y, dy, span, t_out, span_out, indices = prep

    w = dy ** -2
    w, yw = windowed_sum([w, y * w], span=span, subtract_mid=cv,
                         indices=indices)

    if t_out is None or span_out is not None:
        return yw / w
    else:
        i = np.minimum(len(t) - 1, np.searchsorted(t, t_out))
        return yw[i] / w[i]


def linear_smooth(t, y, dy, span=None, cv=True,
                  t_out=None, span_out=None, period=None):
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
    period : float
        if provided, then consider the inputs periodic with the given period

    Returns
    -------
    y_smooth : array_like
        smoothed y values at each time t or t_out
    """
    t_input = t
    prep = _prep_smooth(t, y, dy, span, t_out, span_out, period)
    t, y, dy, span, t_out, span_out, indices = prep
    if period:
        t_input = np.asarray(t_input) % period

    w = dy ** -2
    w, yw, tw, tyw, ttw = windowed_sum([w, y * w, w, y * w, w], t=t,
                                       tpowers=[0, 0, 1, 1, 2],
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


