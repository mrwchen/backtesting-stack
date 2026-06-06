"""Layer 2 — Price process: Kalman Filter | State-Space Model (switchable).

Both implementations expose the same interface and produce *causal* filtered
estimates: the level/slope at bar t use only observations up to and including t.
Parameters are (re)estimated on a past-only training slice; the forward filter is
then run over the full series and sliced — which is identical to running it bar by
bar because Kalman filtering is a forward recursion.

Outputs consumed by the decision layer:
    level   filtered fair price
    slope   filtered local trend (level change per bar)
The decision layer uses price_deviation = close - level, and slope.
"""

import logging
import warnings

import numpy as np

from . import config

log = logging.getLogger(__name__)


def make_price_filter(model: str):
    if model == "kalman":
        return KalmanPriceFilter()
    if model == "state_space":
        return StateSpacePriceFilter()
    raise ValueError(f"Unknown PRICE_MODEL {model!r}")


def _llt_forward(close: np.ndarray, r: float, q_level: float, q_trend: float) -> tuple[np.ndarray, np.ndarray]:
    """Local-linear-trend Kalman forward filter.

    state = [level, trend];  F = [[1,1],[0,1]];  H = [1,0];
    Q = diag(q_level, q_trend);  R = r.  Returns (levels, slopes) per bar.
    """
    n = close.shape[0]
    levels = np.empty(n, dtype=np.float64)
    slopes = np.empty(n, dtype=np.float64)

    # Init at first observation, large initial covariance.
    x = np.array([close[0], 0.0], dtype=np.float64)
    P = np.array([[r * 10.0, 0.0], [0.0, q_trend * 10.0 + 1e-9]], dtype=np.float64)
    F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    Q = np.array([[q_level, 0.0], [0.0, q_trend]], dtype=np.float64)
    levels[0], slopes[0] = x[0], x[1]

    for t in range(1, n):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        # Update with scalar observation y = level
        y = close[t]
        s = P[0, 0] + r
        k0 = P[0, 0] / s
        k1 = P[1, 0] / s
        resid = y - x[0]
        x[0] += k0 * resid
        x[1] += k1 * resid
        # Joseph-free covariance update for scalar obs (H = [1,0])
        P00, P01, P10, P11 = P[0, 0], P[0, 1], P[1, 0], P[1, 1]
        P[0, 0] = P00 - k0 * P00
        P[0, 1] = P01 - k0 * P01
        P[1, 0] = P10 - k1 * P00
        P[1, 1] = P11 - k1 * P01
        levels[t], slopes[t] = x[0], x[1]

    return levels, slopes


class KalmanPriceFilter:
    """Hand-rolled local-linear-trend filter; noise self-scaled from training data."""

    def __init__(self) -> None:
        self._r = 1.0
        self._q_level = 0.1
        self._q_trend = 0.001

    def update_params(self, close_train: np.ndarray) -> None:
        if close_train.shape[0] < 3:
            base_var = 1.0
        else:
            base_var = float(np.var(np.diff(close_train)))
            if not np.isfinite(base_var) or base_var <= 0.0:
                base_var = 1.0
        self._r = config.KF_OBS_NOISE_MULT * base_var
        self._q_level = config.KF_LEVEL_NOISE_MULT * base_var
        self._q_trend = config.KF_TREND_NOISE_MULT * base_var

    def filtered(self, close_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _llt_forward(close_full, self._r, self._q_level, self._q_trend)


class StateSpacePriceFilter:
    """statsmodels UnobservedComponents (local linear trend); MLE-estimated variances.

    Variances are estimated by MLE on the training slice; the resulting parameters
    are then applied with a forward Kalman filter (filtered_state, causal) over the
    full series.
    """

    def __init__(self) -> None:
        self._params = None  # estimated [sigma2_irregular, sigma2_level, sigma2_trend]

    def update_params(self, close_train: np.ndarray) -> None:
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        if close_train.shape[0] < 30:
            self._params = None
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = UnobservedComponents(close_train, level="local linear trend")
                res = model.fit(disp=False, maxiter=50)
            self._params = np.asarray(res.params, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            log.warning("State-space param fit failed (%s); falling back to Kalman defaults", exc)
            self._params = None

    def filtered(self, close_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        if self._params is None:
            # Fallback: behave like the Kalman filter with self-scaled noise.
            base_var = float(np.var(np.diff(close_full))) if close_full.shape[0] > 3 else 1.0
            base_var = base_var if (np.isfinite(base_var) and base_var > 0) else 1.0
            return _llt_forward(close_full, base_var, 0.1 * base_var, 0.001 * base_var)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = UnobservedComponents(close_full, level="local linear trend")
                res = model.filter(self._params)
            fs = np.asarray(res.filtered_state)  # shape (2, n): [level, trend]
            return fs[0].astype(np.float64), fs[1].astype(np.float64)
        except Exception as exc:  # noqa: BLE001
            log.warning("State-space filter failed (%s); falling back to Kalman", exc)
            base_var = float(np.var(np.diff(close_full))) if close_full.shape[0] > 3 else 1.0
            base_var = base_var if (np.isfinite(base_var) and base_var > 0) else 1.0
            return _llt_forward(close_full, base_var, 0.1 * base_var, 0.001 * base_var)
