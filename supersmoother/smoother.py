from __future__ import division, print_function
import numpy as np

__all__ = ['MovingAverageFixedSpan', 'MovingAverageVariableSpan',
           'LinearFixedSpan', 'LinearVariableSpan']


def _simultaneous_sort(*args):
    isort = np.argsort(args[0])
    return [arg[isort] for arg in args]


class BaseSmoother(object):
    def __init__(self):
        raise NotImplementedError()

    def fit(self, t, y, dy=1, presorted=False):
        self.t, self.y, self.dy = self._process_inputs(t, y, dy, presorted)
        self._fit()
        return self

    def _fit(self):
        raise NotImplementedError()

    def _process_inputs(self, t, y, dy, presorted=False):
        t, y, dy = np.broadcast_arrays(t, y, dy)
        isort = slice(None) if presorted else np.argsort(t)
        return (a[isort] for a in (t, y, dy))

    def predict(self, t):
        raise NotImplementedError()

    def cv_values(self, imin=0, imax=None):
        raise NotImplementedError()

    def cv_residuals(self, imin=0, imax=None):
        vals = self.cv_values(imin, imax)
        return (self.y[imin:imax] - vals) / self.dy[imin:imax]

    def cv_error(self, imin=0, imax=None):
        resids = self.cv_residuals(imin, imax)
        return np.mean(resids ** 2)


class BaseFixedSpan(BaseSmoother):
    def __init__(self, span):
        self.span = span

    def _fit(self, presorted=False):
        self._set_span(self.span)
        
    def _set_span(self, span):
        self.span = span
        self.halfspan = np.maximum(1, (span * len(self.t)) // 2).astype(int)
        self.fullspan = (2 * self.halfspan + 1).astype(int)

    def _find_indices(self, a, vals):
        return np.minimum(len(a) - 1, np.searchsorted(a, vals))

    def _imin_imax(self, i):
        imin = int(i - self.halfspan)
        imax = int(imin + 2 * self.halfspan + 1)
        return max(0, imin), min(len(self.t), imax)


class SlowFixedSpan(BaseFixedSpan):
    def cv_values(self, imin=0, imax=None):
        start, stop, step = slice(imin, imax).indices(len(self.t))
        return np.fromiter(map(self._cv_at_index, range(start, stop, step)),
                           dtype=float, count=len(self.t[imin:imax]))

    def predict(self, t):
        t = np.asarray(t)
        return np.fromiter(map(self._predict_at_val, t.ravel()),
                           dtype=float, count=t.size).reshape(t.shape)
        
    def _cv_at_index(self, i):
        imin, imax = self._imin_imax(i)
        args = [np.concatenate([a[imin: i], a[i + 1: imax]])
                for a in [self.t, self.y, self.dy]]
        return self._make_prediction(self.t[i], *args)

    def _predict_at_val(self, t):
        imin, imax = self._imin_imax(self._find_indices(self.t, t))
        args = [a[imin:imax] for a in (self.t, self.y, self.dy)]
        return self._make_prediction(t, *args)


class FastFixedSpan(BaseFixedSpan):
    def cv_values(self, imin=0, imax=None):
        return self._predict_on('cv', sl=slice(imin, imax))

    def predict(self, t):
        return self._predict_on('sum', t=t)

    def _fit(self, presorted=False):
        self._set_span(self.span)
        self._prepare_calcs()

    def _windowed_sum(self, a, window):
        # TODO: switch to fftconvolve when it will make a difference
        return np.convolve(a, np.ones(window), mode='same')

    def _set_fit_params(self, **kwargs):
        fit_params = {} 
        fit_params.update(kwargs)
        fit_params.update(
            dict([(key + 'sum', self._windowed_sum(val, self.fullspan))
                  for key, val in kwargs.items()]))
        fit_params.update(
            dict([(key + 'cv', fit_params[key + 'sum'] - fit_params[key])
                  for key in kwargs]))

        self._fit_params = fit_params


class MovingAverageMixin(object):
    def _make_prediction(self, t, tfit, yfit, dyfit):
        w = dyfit ** -2
        return np.dot(yfit, w) / w.sum()

    def _prepare_calcs(self):
        w = self.dy ** -2
        self._set_fit_params(w=w, yw=self.y * w)

    def _predict_on(self, suffix, t=None, sl=slice(None)):
        vals = (self._fit_params['yw' + suffix][sl] /
                self._fit_params['w' + suffix][sl])
        if t is not None:
            vals = vals[self._find_indices(self.t[sl], t)]
        return vals


class LinearMixin(object):
    def _make_prediction(self, t, tfit, yfit, dyfit):
        X = np.transpose(np.vstack([np.ones_like(tfit), tfit]) / dyfit)
        y = yfit / dyfit
        theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        return theta[0] + theta[1] * t

    def _prepare_calcs(self):
        w = self.dy ** -2
        self._set_fit_params(w=w, tw=self.t * w, yw=self.y * w,
                             tt=self.t * self.t * w,
                             ty=self.t * self.y * w)

    def _predict_on(self, suffix, t=None, sl=slice(None)):
        w, tw, yw, tt, ty = (self._fit_params[key + suffix][sl]
                             for key in ['w', 'tw', 'yw', 'tt', 'ty'])
        denominator = (w * tt - tw * tw)
        slope = (ty * w - tw * yw) / denominator
        intercept = (tt * yw - ty * tw) / denominator

        if t is None:
            return slope * self.t[sl] + intercept
        else:
            i = self._find_indices(self.t[sl], t)
            return slope[i] * t + intercept[i]


class MovingAverageFixedSpanSlow(SlowFixedSpan, MovingAverageMixin):
    pass


class LinearFixedSpanSlow(SlowFixedSpan, LinearMixin):
    pass


class MovingAverageFixedSpan(FastFixedSpan, MovingAverageMixin):
    slow = MovingAverageFixedSpanSlow


class LinearFixedSpan(FastFixedSpan, LinearMixin):
    slow = LinearFixedSpanSlow


class VariableSpanMixin(object):
    def __init__(self, span):
        self._input_span = span

    def _process_inputs(self, t, y, dy, presorted=False):
        t, y, dy, span = np.broadcast_arrays(t, y, dy, self._input_span)
        isort = slice(None) if presorted else np.argsort(t)
        self.span = span[isort]
        return (a[isort] for a in (t, y, dy))

    def _windowed_sum(self, a, window):
        a = np.asarray(a)
        window = np.asarray(window)
        assert a.shape == window.shape
        N = len(a)
        mins = np.arange(len(a)) - window // 2
        ranges = np.vstack([np.maximum(0, mins),
                            np.minimum(len(a), mins + window)]).ravel('F')
        return np.add.reduceat(np.append(a, 0), ranges)[::2]


class MovingAverageVariableSpan(FastFixedSpan, VariableSpanMixin, MovingAverageMixin):
    fixed = MovingAverageFixedSpan


class LinearVariableSpan(FastFixedSpan, VariableSpanMixin, LinearMixin):
    fixed = LinearFixedSpan
