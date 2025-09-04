from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from .utils import linear_smooth, moving_average_smooth, iterable

__all__ = ['MovingAverageSmoother', 'LinearSmoother']


class Smoother:
    """Base Class for Smoothers"""
    def __init__(self):
        raise NotImplementedError()

    def fit(self, t: ArrayLike, y: ArrayLike, dy: ArrayLike = 1,
            presorted: bool = False) -> Smoother:
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

    def predict(self, t: ArrayLike) -> np.ndarray:
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

    def cv_values(self, cv: bool = True) -> np.ndarray:
        """Return the values of the cross-validation for the fit data"""
        return self._cv_values(cv)

    def cv_residuals(self, cv: bool = True) -> np.ndarray:
        """Return the residuals of the cross-validation for the fit data"""
        vals = self.cv_values(cv)
        return (self.y - vals) / self.dy

    def cv_error(self, cv: bool = True, skip_endpoints: bool = True) -> np.ndarray:
        """Return the sum of cross-validation residuals for the input data"""
        resids = self.cv_residuals(cv)
        if skip_endpoints:
            resids = resids[1:-1]
        return np.mean(abs(resids))

    def _validate_inputs(self, t: ArrayLike, y: ArrayLike, dy: ArrayLike,
                         presorted: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t, y, dy = np.broadcast_arrays(t, y, dy)
        if presorted:
            self.isort: slice | np.ndarray = slice(None)
        elif hasattr(self, 'period') and self.period:
            self.isort = np.argsort(t % self.period)
        else:
            self.isort = np.argsort(t)
        return t[self.isort], y[self.isort], dy[self.isort]

    def _fit(self, t: np.ndarray, y: np.ndarray, dy: np.ndarray) -> None:
        """Private function to perform fit() on input data"""
        raise NotImplementedError()

    def _predict(self, t: ArrayLike) -> np.ndarray:
        """Private function implementing prediction for new data"""
        raise NotImplementedError()

    def _cv_values(self, cv: bool = True) -> np.ndarray:
        """Private function implementing cross-validation on fit data"""
        raise NotImplementedError()


class SpannedSmoother(Smoother):
    """Base class for smoothers based on local spans of sorted data"""
    def __init__(self, span: ArrayLike | Callable[[np.ndarray], np.ndarray],
                 period: float | None = None):
        self.span = span
        self.period = period

    @staticmethod
    def _smoothfunc(t: ArrayLike, y: ArrayLike, dy: ArrayLike,
                    span: ArrayLike | None = None, cv: bool = True,
                    t_out: ArrayLike | None = None,
                    span_out: ArrayLike | None = None,
                    period: float | None = None) -> np.ndarray:
        raise NotImplementedError()

    def _fit(self, t: np.ndarray, y: np.ndarray, dy: np.ndarray) -> None:
        pass

    def span_int(self, t: ArrayLike | None = None) -> np.ndarray:
        if t is None:
            t = self.t

        if callable(self.span):
            spanint = self.span(t) * len(self.t)  # type: ignore[arg-type]
        elif iterable(self.span):
            spanint = np.asarray(self.span)[self.isort] * len(self.t)
        else:
            spanint = np.asarray(self.span) * len(self.t)

        return np.clip(spanint, 3, None)

    def _predict(self, t: ArrayLike) -> np.ndarray:
        if callable(self.span):
            return self._smoothfunc(self.t, self.y, self.dy, cv=False,
                                    span_out=self.span_int(t), t_out=t,
                                    period=self.period)
        else:
            return self._smoothfunc(self.t, self.y, self.dy, cv=False,
                                    span=self.span_int(), t_out=t,
                                    period=self.period)

    def _cv_values(self, cv: bool = True) -> np.ndarray:
        return self._smoothfunc(self.t, self.y, self.dy, cv=cv,
                                span=self.span_int(), period=self.period)


class MovingAverageSmoother(SpannedSmoother):
    """Local smoother based on a moving average of adjacent points

    Parameters
    ----------
    span : float, array, or function
        The fraction of the data to use at each point of the smooth.
        If a function is passed, then this will be evaluated at each input
        time to determine the smooth.
    period : float (optional)
        If specified, then use a periodic smoother with the given period.
        Default is to assume no periodicity.
    """
    _smoothfunc = staticmethod(moving_average_smooth)


class LinearSmoother(SpannedSmoother):
    """Local smoother based on a locally linear fit of adjacent points

    Parameters
    ----------
    span : float, array, or function
        The fraction of the data to use at each point of the smooth.
        If a function is passed, then this will be evaluated at each input
        time to determine the smooth.
    period : float (optional)
        If specified, then use a periodic smoother with the given period.
        Default is to assume no periodicity.
    """
    _smoothfunc = staticmethod(linear_smooth)
