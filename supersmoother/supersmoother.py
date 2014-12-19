import numpy as np
from .smoother import LinearSmoother


class SuperSmoother(LinearSmoother):
    """
    SuperSmoother, as described in [1]

    [1] Friedman 1984, "A Variable Span Smoother"
        http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-3477.pdf

    There are 3 smoothing levels:
    - tweeter (span=0.05)
    - midrange (span=0.2)
    - woofer (span=0.5)

    9 passes over the data:

    1. Primary smooths for 3 spans & compute residuals
    2. Smooth these residuals with the midrange span
    3. Select span yielding best smoothed residual for each point
    4. Smooth best span estimates with midrange span
    5. Final model: use smoothed span estimates at each point
    """
    default_spans = [0.05, 0.2, 0.5]

    def __init__(self, primary_spans=None):
        if primary_spans is None:
            primary_spans = self.default_spans
        self.primary_spans = np.sort(primary_spans)
        self.middle_span = primary_spans[len(primary_spans) // 2]
        self.primary_smooths = [LinearSmoother(span) for span in primary_spans]
        self.span = self.middle_span

    def _fit(self):
        t = self.t
        y = self.y
        dy = self.dy
        resids = [smoother.fit(t, y, dy,
                               sort_inputs=False).crossval_residuals()
                  for smoother in self.primary_smooths]
        smoothed_resids = np.array([LinearSmoother(self.middle_span)
                                    .fit(t, abs(resid), 1, sort_inputs=False)
                                    .predict(t) for resid in resids])
        best_spans = self.primary_spans[np.argmin(smoothed_resids, 0)]
        smoothed_spans = LinearSmoother(self.middle_span)\
            .fit(t, best_spans, 1, sort_inputs=False).predict(t)

        self.resids = resids
        self.smoothed_resids = smoothed_resids
        self.best_spans = best_spans
        self.span = smoothed_spans
        self._set_span(smoothed_spans, sort=False)
        LinearSmoother._fit(self)            
