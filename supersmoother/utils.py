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
    
    Returns
    -------
    arrays : tuple of ndarrays
        arrays containing the windowed sum of each input array
    """
    span = kwargs.pop('span', None)
    subtract_mid = kwargs.pop('subtract_mid', False)
    slow = kwargs.pop('slow', False)

    if kwargs:
        raise ValueError("Unrecognized keywords: {0}".format(kwargs.keys()))

    if span is None:
        raise ValueError("Must provide a positive integer span")

    span = np.asarray(span, dtype=int)
    if not np.all(span > 0):
        raise ValueError("span values must be positive")

    if slow:
        # Slow fixed/variable span
        results = []
        for array in map(np.asarray, arrays):
            assert array.ndim == 1
            span, array = np.broadcast_arrays(span, array)
            result = (array[max(0, i - s // 2): i - s // 2 + s].sum()
                      for i, s, in enumerate(span))
            results.append(np.fromiter(result,
                                       dtype=array.dtype,
                                       count=len(array)))

    elif span.ndim == 0:
        # Fast fixed-span
        window = np.ones(span)
        results = [np.convolve(a, window, mode='same') for a in arrays]

    else:
        # Fast variable-span
        window = np.asarray(span)
        mins = np.arange(len(window)) - window // 2
        results = []
        for a in map(np.asarray, arrays):
            assert a.shape == window.shape
            ranges = np.vstack([np.maximum(0, mins),
                                np.minimum(len(a), mins + window)]).ravel('F')
            results.append(np.add.reduceat(np.append(a, 0), ranges)[::2])

    if subtract_mid:
        results = (r - a for r, a in zip(results, arrays))

    return tuple(results)


def moving_average_smooth(t, y, dy, span, cv=True):
    t, y, dy = validate_inputs(t, y, dy, sort_by=t)
    w = dy ** -2
    w, yw = windowed_sum(w, y * w, span=span, subtract_mid=cv)
    return yw / w


def linear_smooth(t, y, dy, span, cv=True):
    t, y, dy = validate_inputs(t, y, dy, sort_by=t)
    w = dy ** -2
    w, tw, yw, ttw, tyw = windowed_sum(w, t * w, y * w, t * t * w, t * y * w,
                                       span=span, subtract_mid=cv)
    denominator = (w * ttw - tw * tw)
    slope = (tyw * w - tw * yw)
    intercept = (ttw * yw - ty * tw)

    return (slope * t + intercept) / denominator
