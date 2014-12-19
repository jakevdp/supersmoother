from __future__ import division, print_function
import numpy as np

__all__ = ['FloatingMeanSmoother', 'LinearSmoother']


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


class BaseFixedSpanSlow(BaseSmoother):
    def __init__(self, span):
        self._input_span = span

    def _fit(self, presorted=False):
        self._set_span(self._input_span)
        
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

    def cv_values(self, imin=0, imax=None):
        start, stop, step = slice(imin, imax).indices(len(self.t))
        return np.fromiter(map(self._cv_at_index, range(start, stop, step)),
                           dtype=float, count=len(self.t[imin:imax]))

    def _make_prediction(self, t, tsamp, ysamp, dysamp):
        raise NotImplementedError()
        
    def _cv_at_index(self, i):
        imin, imax = self._imin_imax(i)
        args = [np.concatenate([a[imin: i], a[i + 1: imax]])
                for a in [self.t, self.y, self.dy]]
        return self._make_prediction(self.t[i], *args)

    def predict(self, t):
        t = np.asarray(t)
        return np.fromiter(map(self._predict_at_val, t.ravel()),
                           dtype=float, count=t.size).reshape(t.shape)

    def _predict_at_val(self, t):
        imin, imax = self._imin_imax(self._find_indices(self.t, t))
        args = [a[imin:imax] for a in (self.t, self.y, self.dy)]
        return self._make_prediction(t, *args)


class BaseFixedSpan(BaseFixedSpanSlow):
    def _fit(self, presorted=False):
        self._set_span(self._input_span)
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

    def cv_values(self, imin=0, imax=None):
        return self._predict_on('cv', sl=slice(imin, imax))

    def predict(self, t):
        return self._predict_on('sum', t=t)

    def _predict_on(self, suffix, t=None, sl=slice(None)):
        raise NotImplementedError()


class MovingAverageFixedSpanSlow(BaseFixedSpanSlow):
    def _make_prediction(self, t, tfit, yfit, dyfit):
        w = dyfit ** -2
        return np.dot(yfit, w) / w.sum()


class MovingAverageFixedSpan(BaseFixedSpan):
    slow = MovingAverageFixedSpanSlow

    def _prepare_calcs(self):
        w = self.dy ** -2
        self._set_fit_params(w=w, yw=self.y * w)

    def _predict_on(self, suffix, t=None, sl=slice(None)):
        vals = (self._fit_params['yw' + suffix][sl] /
                self._fit_params['w' + suffix][sl])
        if t is not None:
            vals = vals[self._find_indices(self.t[sl], t)]
        return vals


class LinearFixedSpanSlow(BaseFixedSpanSlow):
    def _make_prediction(self, t, tfit, yfit, dyfit):
        X = np.transpose(np.vstack([np.ones_like(tfit), tfit]) / dyfit)
        y = yfit / dyfit
        theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        return theta[0] + theta[1] * t


class LinearFixedSpan(BaseFixedSpan):
    slow = LinearFixedSpanSlow

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


class MovingAverageVariableSpan(MovingAverageFixedSpan):
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
        


class Smoother(object):
    """Smoother base class"""
    @staticmethod
    def _window(a, window, mode='same'):
        if hasattr(window, '__len__'):
            assert len(window) == len(a)
            N = len(a)
            window = np.asarray(window)
            mins = np.arange(len(a)) - window // 2
            ranges = np.vstack([np.maximum(0, mins),
                                np.minimum(len(a), mins + window)]).ravel('F')
            return np.add.reduceat(np.append(a, 0), ranges)[::2]
        else:
            # TODO: switch to fftconvolve when it will make a difference
            return np.convolve(a, np.ones(window), mode=mode)

    def __init__(self):
        raise NotImplementedError()

    def fit(self, t, y, dy, sort_inputs=True):
        self._fit_data = np.broadcast_arrays(t, y, dy)
        t, y, dy = self._fit_data

        if sort_inputs:
            self.i_sort = np.argsort(t)
            self.t = t[self.i_sort]
            self.y = y[self.i_sort]  
            self.dy = dy[self.i_sort]
        else:
            self.i_sort = np.arange(len(t))
            self.t = t
            self.y = y
            self.dy = dy

        self._set_span()
        self._fit()
        return self

    def _set_span(self, span=None, sort=True):
        # Full span needs to be at least 3, or cross-validation will fail.
        if span is not None:
            self.span = span
        self.processed_span = self.span
        if sort and hasattr(self.span, '__len__'):
            self.processed_span = self.span[self.i_sort]
        self.halfspan = np.maximum(
            (self.processed_span * len(self.t)) // 2, 1).astype(int)
        self.fullspan = 2 * self.halfspan + 1

    def _fit(self):
        pass

    def _imin_imax(self, ind):
        # In case halfspan is an array, try this:
        if hasattr(self.halfspan, '__len__'):
            halfspan = self.halfspan[ind]
        else:
            halfspan = self.halfspan

        imin = np.maximum(0, ind - halfspan)
        imax = np.minimum(len(self.t), ind + halfspan + 1)
        return imin, imax

    def predict(self, t, slow=None):
        t = np.asarray(t)
        outshape = t.shape
        t = t.ravel()

        if not slow:
            try:
                out = self._predict_batch(t)
                self._predict_type = 'fast'
            except NotImplementedError:
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

    def cross_validate(self, ret_y=False, ret_resids=False,
                       slow=False, imin=1, imax=-1):
        sl = slice(imin, imax)

        if not slow:
            try:
                y_cv = self._cross_validate_batch(sl)
                self._cv_type = 'fast'
            except NotImplementedError:
                slow = True
        
        if slow:
            y_cv = np.fromiter(map(self._cross_validate_single,
                                   range(*sl.indices(len(self.t)))),
                               dtype=float, count=len(self.t[sl]))
            self._cv_type = 'slow'

        if ret_y:
            return y_cv
        elif ret_resids:
            return self.y[sl] - y_cv
        else:
            return np.mean((self.y[sl] - y_cv) ** 2 / self.dy[sl] ** 2)

    def _cross_validate_single(self, ind):
        raise NotImplementedError()

    def crossval_residuals(self, slow=False, imin=0, imax=None):
        return self.cross_validate(ret_resids=True,
                                   slow=slow, imin=imin, imax=imax)


class MovingAverageSmoother(Smoother):
    """Fixed-span moving average smoother"""
    def __init__(self, span=0.05):
        self.span = span

    def _fit(self):
        w = self.dy ** -2
        self._fit_params = {'w': w, 'yw': self.y * w}
        self._fit_params.update(
            dict([(key + 'sum', self._window(val, self.fullspan))
                  for key, val in self._fit_params.items()]))

    def _predict_single(self, t):
        ind = np.searchsorted(self.t, t)
        imin, imax = self._imin_imax(ind)
        sl = slice(imin, imax)
        return (np.dot(self.y[sl], self.dy[sl] ** -2) / 
                np.sum(self.dy[sl] ** -2))

    def _predict_batch(self, t):
        ind = np.minimum(np.searchsorted(self.t, t), len(self.t) - 1)
        return self._fit_params['ywsum'][ind] / self._fit_params['wsum'][ind]

    def _cross_validate_single(self, ind):
        imin, imax = self._imin_imax(ind)
        ys, dys = [np.concatenate([a[imin: ind], a[ind + 1: imax]])
                   for a in [self.y, self.dy]]
        return np.dot(ys, dys ** -2) / np.sum(dys ** -2)

    def _cross_validate_batch(self, sl):
        ywsum = self._fit_params['ywsum'][sl] - self._fit_params['yw'][sl]
        wsum = self._fit_params['wsum'][sl] - self._fit_params['w'][sl]
        return ywsum / wsum


class LinearSmoother(Smoother):
    """Fixed-span linear smoother"""
    def __init__(self, span=0.05):
        self.span = span

    def _fit(self):
        w = self.dy ** -2
        self._fit_params = {'w': w,
                            'tw': self.t * w, 'yw': self.y * w,
                            'tt': self.t * self.t * w,
                            'ty': self.t * self.y * w}
        self._fit_params.update(
            dict([(key + 'sum', self._window(val, self.fullspan))
                  for key, val in self._fit_params.items()]))

    def _predict_single(self, t):
        ind = np.searchsorted(self.t, t)
        imin, imax = self._imin_imax(ind)
        sl = slice(imin, imax)
        ts, ys, dys = self.t[sl], self.y[sl], self.dy[sl]

        X = np.transpose(np.vstack([np.ones_like(ts), ts]) / dys)
        y = ys / dys

        theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        return theta[0] + theta[1] * t

    def _predict_batch(self, t):
        ind = np.minimum(np.searchsorted(self.t, t), len(self.t) - 1)
        vals = (self._fit_params[key + 'sum']
                for key in ['w', 'tw', 'yw', 'tt', 'ty'])

        w, tw, yw, tt, ty = (v[ind] for v in vals)
        denominator = (w * tt - tw * tw)
        slope = (ty * w - tw * yw)
        intercept = (tt * yw - ty * tw)

        return (slope * t + intercept) / denominator
        
    def _cross_validate_single(self, ind):
        imin, imax = self._imin_imax(ind)
        ts, ys, dys = [np.concatenate([a[imin: ind], a[ind + 1: imax]])
                       for a in [self.t, self.y, self.dy]]

        X = np.transpose(np.vstack([np.ones_like(ts), ts]) / dys)
        y = ys / dys

        theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        return theta[0] + theta[1] * self.t[ind]

    def _cross_validate_batch(self, sl):
        vals = (self._fit_params[key + 'sum'] - self._fit_params[key]
                for key in ['w', 'tw', 'yw', 'tt', 'ty'])

        w, tw, yw, tt, ty = (v[sl] for v in vals)
        denominator = (w * tt - tw * tw)
        slope = (ty * w - tw * yw)
        intercept = (tt * yw - ty * tw)
        
        return (slope * self.t[sl] + intercept) / denominator
