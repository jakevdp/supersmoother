import numpy as np
from .smoother import LocalLinearSmoother


class SuperSmoother(LocalLinearSmoother):
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
        self.middle_span = self.primary_spans[len(primary_spans) // 2]
        self.primary_smooths = [LocalLinearSmoother(span)
                                for span in primary_spans]
        self.middle_smooth = self.primary_smooths[len(primary_spans) // 2]

    def _fit(self, t, y, dy):
        # 1. Get residuals for each of the primary smooths
        [smoother.fit(t, y, dy, presorted=True)
         for smoother in self.primary_smooths]
        resids = [smoother.cv_residuals()
                  for smoother in self.primary_smooths]

        # 2. Smooth each set of residuals with the midrange
        smoothed_resids = np.array([self.middle_smooth
                                    .fit(t, abs(resid), 1, False)
                                    .cv_values(False)
                                    for resid in resids])

        # 3. Select span yielding best residual at each point
        best_spans = self.primary_spans[np.argmin(smoothed_resids, 0)]

        # 4. Smooth best span estimates with midrange span
        smoothed_spans = (self.middle_smooth
                          .fit(t, best_spans, 1, False)
                          .cv_values(False))

        # 5. Use these smoothed span estimates at each point
        self.span = smoothed_spans
        LocalLinearSmoother._fit(self, t, y, dy)
