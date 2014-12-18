from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt


class Smoother(object):
    """Smoother base class"""
    def __init__(self):
        raise NotImplementedError()

    def fit(self, t, y, dy):
        t, y, dy = np.broadcast_arrays(t, y, dy)
        self.t = t
        self.y = y
        self.dy = dy
    
        self._sort_inputs()
        self._set_span()
        self._fit()
        return self

    def _sort_inputs(self):
        i_sort = np.argsort(self.t)
        self.tsorted_ = self.t[i_sort]
        self.ysorted_ = self.y[i_sort]
        self.dysorted_ = self.dy[i_sort]

    def _set_span(self):
        # Full span needs to be at least 3, or cross-validation will fail.
        halfspan = int(max(1, (self.span * len(self.t)) // 2))
        self.lowerspan = halfspan
        self.upperspan = halfspan + 1

    def _fit(self):
        raise NotImplementedError()

    def predict(self, t):
        t = np.asarray(t)
        outshape = t.shape
        ind = np.searchsorted(self.tsorted_, t.ravel())
        imin = np.maximum(0, ind - self.lowerspan)
        imax = np.minimum(len(self.tsorted_), ind + self.upperspan)
        return self._predict_batch(t.ravel(), imin, imax).reshape(outshape)

    def _predict_batch(self, t, imin, imax):
        raise NotImplementedError()

    def predict_slow(self, t):
        t = np.asarray(t)
        outshape = t.shape
        out = np.zeros(t.size)
        for i, ti in enumerate(t.ravel()):
            ind = np.searchsorted(self.tsorted_, ti)
            imin = max(0, ind - self.lowerspan)
            imax = ind + self.upperspan
            out[i] = self._predict_single(ti, imin, imax)
        return out.reshape(outshape)

    def _predict_single(self, t, imin, imax):
        raise NotImplementedError()

    def cross_validate_slow(self, t, y, dy):
        self.fit(t, y, dy)
        t, y, dy = self.tsorted_, self.ysorted_, self.dysorted_
        remove_i = lambda a, i: np.concatenate([a[:i], a[i + 1:]])
        y_cv = np.zeros(t.size)
        for i in range(1, len(t) - 1):
            self.fit(remove_i(t, i), remove_i(y, i), remove_i(dy, i))
            self.upperspan -= 1
            y_cv[i] = self.predict(t[i])
        return np.mean(((y - y_cv) / dy) ** 2)


class MovingAverageSmoother(Smoother):
    """Fixed-span moving average smoother"""
    def __init__(self, span=0.05):
        self.span = span

    def _fit(self):
        w = 1. / self.dysorted_
        y = self.ysorted_ / self.dysorted_
        self.fit_data_ = [np.concatenate([[0], np.cumsum(y * w)]),
                          np.concatenate([[0], np.cumsum(w * w)])]

    def _predict_batch(self, t, imin, imax):
        numerator, denominator = [a[imax] - a[imin]
                                  for a in self.fit_data_]
        return numerator / denominator

    def _predict_single(self, t, imin, imax):
        sl = slice(imin, imax)
        return (np.dot(self.ysorted_[sl], self.dysorted_[sl] ** -2) / 
                np.sum(self.dysorted_[sl] ** -2))


class LinearSmoother(Smoother):
    """Fixed-span linear smoother"""
    def __init__(self, span=0.05):
        self.span = span
    
    def _fit(self):
        w = 1. / self.dysorted_
        x = self.tsorted_ / self.dysorted_
        y = self.ysorted_ / self.dysorted_

        self.fit_data_ = [np.concatenate([[0.], np.cumsum(w * w)]),
                          np.concatenate([[0.], np.cumsum(w * x)]),
                          np.concatenate([[0.], np.cumsum(w * y)]),
                          np.concatenate([[0.], np.cumsum(x * x)]),
                          np.concatenate([[0.], np.cumsum(x * y)])]

    def _predict_batch(self, t, imin, imax):
        n, x, y, xx, xy = ((a[imax] - a[imin])
                           for a in self.fit_data_)

        intercept = (xx * y - xy * x) / (n * xx - x * x)
        slope = (n * xy - x * y) / (n * xx - x * x)

        return slope * t + intercept        

    def _predict_single(self, t, imin, imax):
        sl = slice(imin, imax)

        ts = self.tsorted_[sl]
        ys = self.ysorted_[sl]
        dys = self.dysorted_[sl]

        X = np.transpose(np.vstack([np.ones_like(ts), ts]) / dys)
        y = ys / dys

        theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        return theta[0] + theta[1] * t
