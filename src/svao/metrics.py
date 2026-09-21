"""Forecast accuracy metrics and the Diebold-Mariano test."""
import numpy as np
from scipy import stats

DRY_MONTHS = (11, 12, 1, 2, 3)


def _align(y_true, y_pred):
    a = np.asarray(y_true, float)
    b = np.asarray(y_pred, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return a[ok], b[ok]


def rmse(y_true, y_pred):
    a, b = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def mae(y_true, y_pred):
    a, b = _align(y_true, y_pred)
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


def mape(y_true, y_pred):
    a, b = _align(y_true, y_pred)
    nz = np.abs(a) > 1e-9
    return float(np.mean(np.abs((a[nz] - b[nz]) / a[nz])) * 100) if nz.any() else float("nan")


def r2(y_true, y_pred):
    """Against the mean of the EVALUATION period. Negative values are possible
    and mean the model does worse than predicting that mean."""
    a, b = _align(y_true, y_pred)
    if len(a) < 2:
        return float("nan")
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def all_metrics(y_true, y_pred):
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred),
            "MAPE": mape(y_true, y_pred), "R2": r2(y_true, y_pred)}


def diebold_mariano(y_true, pred_a, pred_b, h=1, power=2):
    """Equal predictive accuracy test with the Harvey-Leybourne-Newbold
    small-sample correction, referred to a t distribution.

    The correction matters here: the test period is 48 months, and the
    asymptotic normal reference is unreliable at that length.

    Negative statistic favours A. p < alpha means the difference is unlikely
    to be sampling noise.
    """
    a = np.asarray(y_true, float)
    pa = np.asarray(pred_a, float)
    pb = np.asarray(pred_b, float)
    ok = ~(np.isnan(a) | np.isnan(pa) | np.isnan(pb))
    a, pa, pb = a[ok], pa[ok], pb[ok]
    n = len(a)
    if n < 8:
        return None

    d = np.abs(a - pa) ** power - np.abs(a - pb) ** power
    d_bar = float(np.mean(d))
    gamma0 = float(np.mean((d - d_bar) ** 2))
    v = gamma0
    for lag in range(1, h):
        v += 2 * float(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
    if v <= 0:
        return None

    dm = d_bar / np.sqrt(v / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_star = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_star), df=n - 1))
    return {"dm": float(dm_star), "p": float(p), "n": int(n),
            "favours": "A" if d_bar < 0 else "B"}
