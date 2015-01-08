from __future__ import division, print_function
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
            ranges = np.vstack([np.maximum(0, mins),
                                np.minimum(len(a), mins + window)]).ravel('F')
            results.append(np.add.reduceat(np.append(a, 0), ranges)[::2])

    if subtract_mid:
        results = (r - a for r, a in zip(results, arrays))

    return tuple(results)


def windowed_sum_varspan(*arrays, **kwargs):
    """TODO: document this
    TODO: combine with windowed_sum??
    """
    span = kwargs.pop('span', None)
    indices = kwargs.pop('indices', None)
    slow = kwargs.pop('slow', False)

    if kwargs:
        raise ValueError("Unrecognized keywords: {0}".format(kwargs.keys()))

    if span is None:
        raise ValueError("Must provide a positive spans")

    if indices is None:
        raise ValueError("Must provide an array of indices")

    span = np.asarray(span, dtype=int)
    if not np.all(span > 0):
        raise ValueError("span values must be positive")

    span, indices = np.broadcast_arrays(span, indices)

    if slow:
        raise ValueError("slow version not implemented")
    else:
        window = np.asarray(span)
        mins = indices - window // 2

        results = []
        for a in map(np.asarray, arrays):
            ranges = np.vstack([np.maximum(0, mins),
                                np.minimum(len(a), mins + window)]).ravel('F')
            results.append(np.add.reduceat(np.append(a, 0), ranges)[::2])

    return results


def moving_average_smooth(t, y, dy, span, cv=True, t_out=None):
    """Perform a moving-average smooth of the data

    Parameters
    ----------
    t, y, dy : array_like
        time, value, and error in value of the input data
    span : int
        the integer span of the data
    cv : boolean (default=True)
        if True, treat the problem as a cross-validation, i.e. don't use
        each point in the evaluation of its own smoothing.

    Returns
    -------
    y_smooth : array_like
        smoothed y values at each time t
    """
    t, y, dy = validate_inputs(t, y, dy, sort_by=t)
    w = dy ** -2
    w, yw = windowed_sum(w, y * w, span=span, subtract_mid=cv)

    if t_out is None:
        return yw / w
    else:
        i = np.minimum(len(t) - 1, np.searchsorted(t, t_out))
        return yw[i] / w[i]
        

def moving_average_smooth_varspan(t, y, dy, span, t_out):
    """
    TODO: doc
    TODO: combine with standard linear smooth?
    span matches t_out
    """
    t, y, dy = validate_inputs(t, y, dy, sort_by=t)
    indices = np.searchsorted(t, t_out)
    
    w = dy ** -2
    w, yw = windowed_sum_varspan(w, y * w, span=span, indices=indices)

    return yw / w


def linear_smooth(t, y, dy, span, t_out=None, cv=True):
    """Perform a linear smooth of the data

    Parameters
    ----------
    t, y, dy : array_like
        time, value, and error in value of the input data
    span : int
        the integer span of the data
    cv : boolean (default=True)
        if True, treat the problem as a cross-validation, i.e. don't use
        each point in the evaluation of its own smoothing.

    Returns
    -------
    y_smooth : array_like
        smoothed y values at each time t
    """
    t_input = t

    t, y, dy = validate_inputs(t, y, dy, sort_by=t)
    w = dy ** -2
    w, tw, yw, ttw, tyw = windowed_sum(w, t * w, y * w, t * t * w, t * y * w,
                                       span=span, subtract_mid=cv)
    denominator = (w * ttw - tw * tw)
    slope = (tyw * w - tw * yw)
    intercept = (ttw * yw - tyw * tw)

    if t_out is None:
        return (slope * t_input + intercept) / denominator
    else:
        i = np.minimum(len(t) - 1, np.searchsorted(t, t_out))
        return (slope[i] * t_out + intercept[i]) / denominator[i]
        

def linear_smooth_varspan(t, y, dy, span, t_out):
    """
    TODO: doc
    TODO: combine with standard linear smooth?
    span matches t_out
    """
    t, y, dy = validate_inputs(t, y, dy, sort_by=t)
    indices = np.searchsorted(t, t_out)
    
    w = dy ** -2
    w, tw, yw, ttw, tyw = windowed_sum_varspan(w, t * w, y * w,
                                               t * t * w, t * y * w,
                                               span=span, indices=indices)
    denominator = (w * ttw - tw * tw)
    slope = (tyw * w - tw * yw)
    intercept = (ttw * yw - tyw * tw)

    return (slope * t_out + intercept) / denominator
