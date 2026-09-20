"""Sliding-window construction and training-only scaling."""
import numpy as np


class MinMaxScaler:
    """Fitted on the training split alone; validation and test may exceed [0,1]."""

    def fit(self, values):
        v = np.asarray(values, float)
        v = v[~np.isnan(v)]
        self.lo, self.hi = float(v.min()), float(v.max())
        if self.hi == self.lo:
            raise ValueError("constant series cannot be scaled")
        return self

    def transform(self, values):
        return (np.asarray(values, float) - self.lo) / (self.hi - self.lo)

    def inverse(self, values):
        return np.asarray(values, float) * (self.hi - self.lo) + self.lo


def make_sequences(values, months, years, lookback, lo_year, hi_year):
    """Windows whose TARGET falls within [lo_year, hi_year].

    A window is admitted only if neither it nor its target contains a missing
    value; windows spanning an excluded month are discarded, never imputed.
    """
    v = np.asarray(values, float)
    m = np.asarray(months, int)
    y = np.asarray(years, int)
    X, t, mm = [], [], []
    for i in range(lookback, len(v)):
        if not (lo_year <= y[i] <= hi_year):
            continue
        win, tgt = v[i - lookback:i], v[i]
        if np.isnan(win).any() or np.isnan(tgt):
            continue
        X.append(win); t.append(tgt); mm.append(m[i])
    if not X:
        return np.empty((0, lookback, 1)), np.empty(0), np.empty(0, int)
    return np.asarray(X)[..., None], np.asarray(t), np.asarray(mm, int)
