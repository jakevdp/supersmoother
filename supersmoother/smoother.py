from __future__ import division, print_function

import numpy as np
from scipy.signal import fftconvolve


class Smoother(object):
    """Smoother base class"""
    def __init__(self):
        raise NotImplementedError()

    def fit(self, t, y, dy, sort_inputs=True):
        self.t, self.y, self.dy = np.broadcast_arrays(t, y, dy)
        
        if sort_inputs:
            self._sort_inputs()

        self._set_span()
        self._fit()
        return self

    def _sort_inputs(self):
        i_sort = np.argsort(self.t)
        self.t = self.t[i_sort]
        self.y = self.y[i_sort]
        self.d = self.dy[i_sort]

    def _set_span(self):
        # Full span needs to be at least 3, or cross-validation will fail.
        self.halfspan = int(max(1, (self.span * len(self.t)) // 2))
        self.lowerspan = self.halfspan
        self.upperspan = self.halfspan + 1
        self.fullspan = 2 * self.halfspan + 1

    def _fit(self):
        pass

    def _imin_imax(self, ind):
        imin = np.maximum(0, ind - self.lowerspan)
        imax = np.minimum(len(self.t), ind + self.upperspan)
        return imin, imax

    def predict(self, t, slow=False):
        t = np.asarray(t)
        outshape = t.shape
        t = t.ravel()

        if not slow:
            try:
                out = self._predict_batch(t)
                self._predict_type = 'fast'
            except:
                slow = True

        if slow:
            out = np.fromiter(map(self._predict_single, t),
                              dtype=float, count=len(t))
            self._predict_type = 'slow'

        return out.reshape(outshape)

    def _predict_single(self, t, imin, imax):
        raise NotImplementedError()

    def _predict_batch(self, t):
        raise NotImplementedError()

    def cross_validate(self, ret_y=False):
        # Don't use first or last entry (extrapolation is bad...)
        y_cv = np.zeros_like(self.t)

        for ind in range(1, len(self.t) - 1):
            imin, imax = self._imin_imax(ind)
            y_cv[ind] = self._cross_validate_single(ind, imin, imax)

        if ret_y:
            return y_cv
        else:
            return np.mean((self.y[1:-1] - y_cv[1:-1]) ** 2
                           / self.dy[1:-1] ** 2)

    def _cross_validate_single(self, ind, imin, imax):
        raise NotImplementedError()


class MovingAverageSmoother(Smoother):
    """Fixed-span moving average smoother"""
    def __init__(self, span=0.05):
        self.span = span

    def _fit(self):
        window = np.ones(self.fullspan)
        w = self.dy ** -2
        yw = self.y * w
        self._fit_params = {'w': w,
                            'yw': yw,
                            'wsum': fftconvolve(w, window, 'same'),
                            'ywsum': fftconvolve(yw, window, 'same')}

    def _predict_single(self, t):
        ind = np.searchsorted(self.t, t)
        imin, imax = self._imin_imax(ind)
        sl = slice(imin, imax)
        return (np.dot(self.y[sl], self.dy[sl] ** -2) / 
                np.sum(self.dy[sl] ** -2))

    def _predict_batch(self, t):
        ind = np.searchsorted(self.t, t)
        return self._fit_params['ywsum'][ind] / self._fit_params['wsum'][ind]

    def _cross_validate_single(self, ind, imin, imax):
        ys, dys = [np.concatenate([a[imin: ind], a[ind + 1: imax]])
                   for a in [self.y, self.dy]]
        return np.dot(ys, dys ** -2) / np.sum(dys ** -2)


class LinearSmoother(Smoother):
    """Fixed-span linear smoother"""
    def __init__(self, span=0.05):
        self.span = span

    def _predict_single(self, t):
        ind = np.searchsorted(self.t, t)
        imin, imax = self._imin_imax(ind)
        sl = slice(imin, imax)

        ts = self.t[sl]
        ys = self.y[sl]
        dys = self.dy[sl]

        X = np.transpose(np.vstack([np.ones_like(ts), ts]) / dys)
        y = ys / dys

        theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        return theta[0] + theta[1] * t

    def _cross_validate_single(self, ind, imin, imax):
        ts, ys, dys = [np.concatenate([a[imin: ind], a[ind + 1: imax]])
                       for a in [self.t, self.y, self.dy]]

        X = np.transpose(np.vstack([np.ones_like(ts), ts]) / dys)
        y = ys / dys

        theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        return theta[0] + theta[1] * self.t[ind]
        
