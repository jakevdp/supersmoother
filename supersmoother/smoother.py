from __future__ import division, print_function
import numpy as np

__all__ = ['MovingAverageFixedSpan', 'MovingAverageVariableSpan',
           'LinearFixedSpan', 'LinearVariableSpan']


class _BaseSmoother(object):
    """Base Class for Smoothers"""
    def __init__(self):
        raise NotImplementedError()

    def fit(self, t, y, dy=1, presorted=False):
        """Fit the smoother

        Parameters
        ----------
        t : array_like
            time locations of the points to smooth
        y : array_like
            y locations of the points to smooth
        dy : array_like or float (default = 1)
            Errors in the y values
        presorted : bool (default = False)
            If True, then t is assumed to be sorted.

        Returns
        -------
        self : Smoother instance
        """
        self.t, self.y, self.dy = self._process_inputs(t, y, dy, presorted)
        self._fit()
        return self

    def predict(self, t):
        """Predict the smoothed function value at time t
        
        Parameters
        ----------
        t : array_like
            Times at which to predict the result

        Returns
        -------
        y : ndarray
            Smoothed values at time t
        """
        t = np.asarray(t)
        return self._predict(t).reshape(t.shape)

    def cv_values(self, imin=0, imax=None):
        """Return the values of the cross-validation for the fit data

        Parameters
        ----------
        imin, imax : integers
            Indices of the minimum and maximum input time at which to compute
            the cross-validation. Default is (imin, imax) = (0, None)
        Returns
        -------
        vals : array
            The cross-validation smoothed values for self.t[imin:imax]
        """
        return self._cv_values(imin, imax)

    def cv_residuals(self, imin=0, imax=None):
        """Return the residuals of the cross-validation for the fit data

        Parameters
        ----------
        imin, imax : integers
            Indices of the minimum and maximum input time at which to compute
            the cross-validation. Default is (imin, imax) = (0, None)
        Returns
        -------
        resids : array
            The cross-validation residuals for self.t[imin:imax]
        """
        vals = self.cv_values(imin, imax)
        return (self.y[imin:imax] - vals) / self.dy[imin:imax]

    def cv_error(self, imin=0, imax=None):
        """Return the sum of cross-validation residuals for the input data

        Parameters
        ----------
        imin, imax : integers
            Indices of the minimum and maximum input time at which to compute
            the cross-validation. Default is (imin, imax) = (0, None)
        Returns
        -------
        resid : float
            The cross-validation residual for the fit data
        """
        resids = self.cv_residuals(imin, imax)
        return np.mean(resids ** 2)

    def _process_inputs(self, t, y, dy, presorted=False):
        """Private function to process inputs to self.fit()"""
        t, y, dy = np.broadcast_arrays(t, y, dy)
        isort = slice(None) if presorted else np.argsort(t)
        return (a[isort] for a in (t, y, dy))

    def _fit(self):
        """Private function to perform fit() on input data"""
        raise NotImplementedError()

    def _predict(self, t):
        """Private function implementing prediction for new data"""
        raise NotImplementedError()

    def _cv_values(self, imin=0, imax=None):
        """Private function implementing cross-validation on fit data"""
        raise NotImplementedError()


class _BaseFixedSpan(_BaseSmoother):
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


class _SlowFixedSpan(_BaseFixedSpan):
    def _cv_values(self, imin=0, imax=None):
        start, stop, step = slice(imin, imax).indices(len(self.t))
        return np.fromiter(map(self._cv_at_index, range(start, stop, step)),
                           dtype=float, count=len(self.t[imin:imax]))

    def _predict(self, t):
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


class _FastFixedSpan(_BaseFixedSpan):
    def _cv_values(self, imin=0, imax=None):
        return self._predict_on('cv', sl=slice(imin, imax))

    def _predict(self, t):
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


class _MovingAverageMixin(object):
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


class _LinearMixin(object):
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


class MovingAverageFixedSpanSlow(_SlowFixedSpan, _MovingAverageMixin):
    """Slow version of MovingAverageFixedSpan. Primarily used for testing.

    Refer to documentation of MovingAverageFixedSpan
    """
    pass


class LinearFixedSpanSlow(_SlowFixedSpan, _LinearMixin):
    """Slow version of LinearFixedSpan. Primarily used for testing.

    Refer to documentation of LinearFixedSpan
    """
    pass


class MovingAverageFixedSpan(_FastFixedSpan, _MovingAverageMixin):
    """Fixed-span smoother based on a moving average

    Parameters
    ----------
    span : int
        The size of the span; i.e. the number of adjacent points to use in
        the smoother.
    """
    slow = MovingAverageFixedSpanSlow


class LinearFixedSpan(_FastFixedSpan, _LinearMixin):
    """Fixed-span smoother based on a local linear model

    Parameters
    ----------
    span : int
        The size of the span; i.e. the number of adjacent points to use in
        the smoother.
    """
    slow = LinearFixedSpanSlow


class _VariableSpanMixin(object):
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


class MovingAverageVariableSpan(_FastFixedSpan, _VariableSpanMixin,
                                _MovingAverageMixin):
    """Variable span smoother based on a moving average model.

    Parameters
    ----------
    span : int or array of ints
        The size of the span to use at each point. If span is an array, then
        the size of the array must match the size of any data passed to fit().
    """
    fixed = MovingAverageFixedSpan


class LinearVariableSpan(_FastFixedSpan, _VariableSpanMixin, _LinearMixin):
    """Variable span smoother based on a local linear regression at each point.

    Parameters
    ----------
    span : int or array of ints
        The size of the span to use at each point. If span is an array, then
        the size of the array must match the size of any data passed to fit().
    """
    fixed = LinearFixedSpan
    
