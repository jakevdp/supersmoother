from __future__ import division, print_function
import numpy as np
from .utils import linear_smooth, moving_average_smooth, linear_smooth_varspan

__all__ = ['MovingAverageSmoother', 'LocalLinearSmoother',
           'FunctionSpanLinearSmoother']


class Smoother(object):
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
        self.t, self.y, self.dy = self._validate_inputs(t, y, dy, presorted)
        self._fit(self.t, self.y, self.dy)
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
        return self._predict(np.ravel(t)).reshape(t.shape)

    def cv_values(self, cv=True):
        """Return the values of the cross-validation for the fit data"""
        return self._cv_values(cv)

    def cv_residuals(self, cv=True):
        """Return the residuals of the cross-validation for the fit data"""
        vals = self.cv_values(cv)
        return (self.y- vals) / self.dy

    def cv_error(self, cv=True, skip_endpoints=True):
        """Return the sum of cross-validation residuals for the input data"""
        resids = self.cv_residuals(cv)
        if skip_endpoints:
            resids = resids[1:-1]
        return np.mean(resids ** 2)

    def _validate_inputs(self, t, y, dy, presorted=False):
        t, y, dy = np.broadcast_arrays(t, y, dy)
        if not presorted:
            isort = np.argsort(t)
            t, y, dy = t[isort], y[isort], dy[isort]
        return t, y, dy

    def _fit(self):
        """Private function to perform fit() on input data"""
        raise NotImplementedError()

    def _predict(self, t):
        """Private function implementing prediction for new data"""
        raise NotImplementedError()

    def _cv_values(self, cv=True):
        """Private function implementing cross-validation on fit data"""
        raise NotImplementedError()


class MovingAverageSmoother(Smoother):
    """Local smoother based on a moving average of adjacent points

    Parameters
    ----------
    span : float or array
        The fraction of the data to use at each point of the smooth
    """
    def __init__(self, span):
        self.span = span

    def _fit(self, t, y, dy):
        self.span_int = self.span * len(t)

    def _predict(self, t):
        return moving_average_smooth(self.t, self.y, self.dy,
                                      self.span_int, cv=False, t_out=t)

    def _cv_values(self, cv=True):
        return moving_average_smooth(self.t, self.y, self.dy,
                                     self.span_int, cv=cv)


class LocalLinearSmoother(Smoother):
    """Local smoother based on a locally linear fit of adjacent points

    Parameters
    ----------
    span : float or array
        The fraction of the data to use at each point of the smooth
    """
    def __init__(self, span):
        self.span = span

    def _fit(self, t, y, dy):
        self.span_int = (self.span * len(t)).astype(int)

    def _predict(self, t):
        return linear_smooth(self.t, self.y, self.dy,
                             self.span_int, cv=False, t_out=t)

    def _cv_values(self, cv=True):
        return linear_smooth(self.t, self.y, self.dy,
                             self.span_int, cv=cv)


class FunctionSpanLinearSmoother(LocalLinearSmoother):
    """TODO: document this"""
    def __init__(self, spanfunc):
        self.spanfunc = spanfunc

    def _fit(self, t, y, dy):
        self.span_int = len(t) * self.spanfunc(t)

    def _predict(self, t):
        span_int = len(self.t) * self.spanfunc(t)
        return linear_smooth_varspan(self.t, self.y, self.dy,
                                     span=span_int, t_out=t)
