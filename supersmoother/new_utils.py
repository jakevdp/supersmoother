import numpy as np


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
        must be sorted according to t % period!
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
                                         i - s // 2 + s))
                      for i, s in np.broadcast(indices, spans)]
        else:
            result = [sum(array[j]
                          * t[j] ** tpower
                          for j in range(max(0, i - s // 2),
                                         min(N, i - s // 2 + s)))
                      for i, s in np.broadcast(indices, spans)]
        results.append(np.asarray(result))

    if subtract_mid:
        results = [r - a[indices] * t[indices] ** tp
                   for r, a, tp in zip(results, arrays, tpowers)]
        
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
        must be sorted according to t % period!
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

    if period:
        if t_input is None:
            raise ValueError("periodic requires t to be provided")
        t = t % period
    
    if indices is not None:
        span, indices = np.broadcast_arrays(span, indices)

    # For the periodic case, re-call the function with padded arrays
    if period:
        t, arrays, sl = _pad_arrays(t, arrays, indices, span, period)
        if len(t) == N:
            # No padding needed! We'll carry-on with the algorithm here.
            pass
        else:
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

    # Now we should be safely in the non-periodic case.
    if span.ndim == 0:
        # fixed-span case
        window = np.ones(span)
        if period:
            # This should have been taken care of above
            raise ValueError("Something went terribly wrong...")
        else:
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
            indices = slice(None) # for below

    else:
        # variable-span case
        if indices is None:
            indices = np.arange(len(span))
        if period:
            # This should have been taken care of above
            raise ValueError("Something went terribly wrong...")
        else:
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


#----------------------------------------------------------------------
# Testing routines

from numpy.testing import assert_allclose
from nose import SkipTest


def _fixed_span_sum(a, span, subtract_mid=False):
    """fixed-span sum; used for testing windowed_span"""
    window = np.ones(span)
    if span <= len(a):
        res = np.convolve(a, window, mode='same')
    else:
        start = (span - 1) // 2
        res = np.convolve(a, window, mode='full')[start:start + len(a)]
    if subtract_mid:
        res -= a
    return res


def _variable_span_sum(a, span, subtract_mid=False):
    """variable-span sum, using multiple runs of _fixed_span_sum"""
    a, spans = np.broadcast_arrays(a, span)
    unique_spans, inv = np.unique(spans, return_inverse=True)
    results = [_fixed_span_sum(a, s, subtract_mid) for s in unique_spans]
    return np.asarray([results[j][i] for i, j in enumerate(inv)])


def test_windowed_sum_simple():
    #"""windowed sum in the simplest case"""
    rng = np.random.RandomState(0)
        
    def check_results(wsum, N, span):
        a = rng.rand(N)
        assert_allclose([_fixed_span_sum(a, span),
                         _fixed_span_sum(a ** 2, span)],
                        wsum([a, a ** 2], span))

    for wsum in [windowed_sum, windowed_sum_slow]:
        for N in range(2, 6):
            for span in range(1, 7):
                yield check_results, wsum, N, span


def test_windowed_sum_with_t():
    #"""windowed sum with powers of t"""
    rng = np.random.RandomState(0)
        
    def check_results(wsum, N, span):
        a = rng.rand(N)
        t = np.sort(rng.rand(N))
        assert_allclose([_fixed_span_sum(a, span),
                         _fixed_span_sum(a * t, span)],
                        wsum([a, a], span, t=t, tpowers=[0, 1]))

    for wsum in [windowed_sum, windowed_sum_slow]:
        for N in range(2, 6):
            for span in range(1, 7):
                yield check_results, wsum, N, span


def test_windowed_sum_varspan():
    #"""windowed sum with a variable span"""
    rng = np.random.RandomState(0)
        
    def check_results(wsum, N):
        a = rng.rand(N)
        span = rng.randint(1, 5, N)
        assert_allclose([_variable_span_sum(a, span),
                         _variable_span_sum(a ** 2, span)],
                        wsum([a, a ** 2], span))

    for wsum in [windowed_sum, windowed_sum_slow]:
        for N in range(2, 6):
            yield check_results, wsum, N


def test_windowed_sum_with_period():
    #"""windowed sum with period"""
    rng = np.random.RandomState(0)
        
    def check_results(wsum, N, span, tpowers, period=0.6):
        a = rng.rand(N)
        t = np.sort(period * rng.rand(N))

        # find the periodic result straightforwardly by concatenating arrays
        a_all = np.concatenate([a for i in range(6)])
        t_all = np.concatenate([t + i * period for i in range(-3, 3)])

        res1 = wsum(len(tpowers) * [a_all], span, t=t_all,
                    tpowers=tpowers, period=None)
        res1 = [a[3 * N: 4 * N] for a in res1]
        res2 = wsum(len(tpowers) * [a], span, t=t,
                    tpowers=tpowers, period=period)
        assert_allclose(res1, res2)

    for wsum in [windowed_sum, windowed_sum_slow]:
        for N in range(4, 7):
            for span in range(1, 7):
                for tpowers in [(0, 1), (0, 1, 2)]:
                    yield check_results, wsum, N, span, tpowers
    

def test_windowed_sum_with_indices():
    #"""windowed sum with indices"""
    rng = np.random.RandomState(0)
        
    def check_results(wsum, N, span):
        ind = rng.randint(0, N, 4)
        a = rng.rand(N)
        t = np.sort(rng.rand(N))

        res1 = wsum([a, a], span, t=t, tpowers=[0, 1])
        res1 = [a[ind] for a in res1]
        res2 = wsum([a, a], span, t=t, tpowers=[0, 1],
                                 indices=ind)
        
        assert_allclose(res1, res2)

    for wsum in [windowed_sum, windowed_sum_slow]:
        for N in range(4, 6):
            for span in range(1, 7):
                yield check_results, wsum, N, span


def test_windowed_sum_subtract_mid():
    #"""windowed sum, subtracting the middle item"""
    rng = np.random.RandomState(0)
        
    def check_results(wsum, N, span):
        a = rng.rand(N)
        t = np.sort(rng.rand(N))
        assert_allclose([_fixed_span_sum(a, span, subtract_mid=True),
                         _fixed_span_sum(a * t, span, subtract_mid=True)],
                        wsum([a, a], span, t=t, tpowers=[0, 1],
                             subtract_mid=True))

    for wsum in [windowed_sum, windowed_sum_slow]:
        for N in range(2, 6):
            for span in range(1, 7):
                yield check_results, wsum, N, span


def test_windowed_sum_fast_vs_slow():
    #"""Test fast vs slow windowed sum for many combinations of parameters"""
    rng = np.random.RandomState(0)
        
    def check_results(N, span, indices, subtract_mid,
                      tpowers, period, use_t):
        a = rng.rand(N)
        t = np.sort(rng.rand(N))
        if period:
            t *= period

        if np.asarray(tpowers).size == 1:
            arrs = [a]
        else:
            arrs = len(tpowers) * [a]

        if not use_t:
            t = None

        res1 = windowed_sum_slow(arrs, span, t=t, subtract_mid=subtract_mid,
                                 indices=indices, tpowers=tpowers,
                                 period=period)
        res2 = windowed_sum(arrs, span, t=t, subtract_mid=subtract_mid,
                            indices=indices, tpowers=tpowers, period=period)
        assert_allclose(res1, res2)

    for N in [3, 5, 7]:
        for span in [3, 5, rng.randint(2, 5, N)]:
            for indices in [None, rng.randint(0, N, N)]:
                for subtract_mid in [True, False]:
                    for tpowers in [0, (0, 1)]:
                        for period in (0.6, None):
                            for use_t in [True, False]:
                                if period and not use_t:
                                    continue
                                yield (check_results, N, span, indices,
                                       subtract_mid, tpowers, period, use_t)
