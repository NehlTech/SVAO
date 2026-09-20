"""Seasonal variance profiles and the SVAO weight rule."""
import numpy as np

DRY_MONTHS = (11, 12, 1, 2, 3)


def profile_season(residuals, months):
    """Variance ratio per calendar month, estimated over two seasonal groups.

    Selected in notebook 04 by validation predictive log-likelihood, in
    preference to a month-indexed profile which overfits at this sample size.
    """
    v = np.asarray(residuals, float)
    m = np.asarray(months, int)
    gv = np.var(v, ddof=1)
    dry = np.isin(m, DRY_MONTHS)
    dv = np.var(v[dry], ddof=1) / gv
    wv = np.var(v[~dry], ddof=1) / gv
    return np.array([dv if k in DRY_MONTHS else wv for k in range(1, 13)])


def svao_weights(profile, lam, normalise=True):
    """w_m = 1 / (1 + lam * sigma2_m), rescaled to unit mean.

    Normalisation holds the average effective learning rate fixed so the
    comparison isolates seasonal redistribution rather than also changing the
    global step size. lam = 0 recovers uniform weighting exactly.
    """
    w = 1.0 / (1.0 + lam * np.asarray(profile, float))
    return w / w.mean() if normalise else w
