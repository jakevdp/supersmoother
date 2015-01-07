import numpy as np


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
    for key in kwargs:
        if key != 'sort_by':
            raise ValueError("unrecognized argument: {0}".format(key))

    if arrays[0].ndim != 1:
        raise ValueError("Input arrays should be one-dimensional.")

    
    if 'sort_by' in kwargs:
        isort = np.argsort(kwargs['sort_by'])
        if isort.shape != arrays[0].shape:
            raise ValueError("sort shape must equal array shape.")
        arrays = tuple([a[isort] for a in arrays])
    return arrays


def windowed_sum(span, *arrays):
    """Compute the windowed sum of the given arrays

    Parameters
    ----------
    span : int or array of ints
        The span to use for the sum at each point
    *args : array_like
        Each additional argument is an array that will be windowed
    
    Returns
    -------
    arrays : tuple of ndarrays
        arrays containing the windowed sum of each input array
    """
    span = np.asarray(span, dtype=int)

    if span.ndim == 0:
        # Fixed-span
        window = np.ones(span)
        results = [np.convolve(a, window, mode='same') for a in arrays]
    else:
        # Variable-span
        window = np.asarray(span)
        mins = np.arange(len(window)) - window // 2
        results = []
        for a in map(np.asarray, arrays):
            assert a.shape == window.shape
            ranges = np.vstack([np.maximum(0, mins),
                                np.minimum(len(a), mins + window)]).ravel('F')
            results.append(np.add.reduceat(np.append(a, 0), ranges)[::2])

    return tuple(results)


def windowed_sum_slow(span, *arrays):
    """Slow version of the windowed sum of the given arrays

    This function is used mainly for testing; call signature and return
    value matches that of the windowed_sum() function.
    """
    results = []
    span = np.asarray(span, dtype=int)

    for array in map(np.asarray, arrays):
        span, array = np.broadcast_arrays(span, array)
        assert array.ndim == 1

        results.append(np.fromiter((array[max(0, i - s // 2):
                                          i - s // 2 + s].sum()
                                    for i, s, in enumerate(span)),
                                   dtype=array.dtype,
                                   count=len(array)))
    return tuple(results)


def build_windowed_arrays(arrays, span, cv=True):
    windowed_arrays = windowed_sum(span, *arrays)
    if cv:
        windowed_arrays = [(asum - a) for (asum, a)
                           in zip(windowed_arrays, arrays)]
    return windowed_arrays


def moving_average_smooth(t, y, dy, span, cv=True):
    t, y, dy = validate_inputs(t, y, dy, sort_by=t)
    w = dy ** -2
    w, yw = build_windowed_arrays((w, y * w), span, cv)
    return yw / w


def linear_smooth(t, y, dy, span, cv=True):
    t, y, dy = validate_inputs(t, y, dy, sort_by=t)
    w = dy ** -2
    w, tw, yw, ttw, tyw = build_windowed_arrays((w, t * w, y * w,
                                                 t * t * w, t * y * w),
                                                span, cv)
    denominator = (w * ttw - tw * tw)
    slope = (tyw * w - tw * yw)
    intercept = (ttw * yw - ty * tw)

    return (slope * t + intercept) / denominator
