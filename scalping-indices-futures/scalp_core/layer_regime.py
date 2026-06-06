"""Layer 1 — Market regime via Hidden Markov Model.

A Gaussian HMM (diagonal covariance) is fit on standardized [log_return,
|log_return|]. The scaler is fit from the past-only training window at each refit.
Hidden states are ranked by their emission variance so the highest-variance state
is flagged as the "turbulent" regime.

For per-bar state assignment we use a **causal forward filter**: the filtered
posterior P(state_t | observations up to t) is computed from the fitted parameters
(startprob, transmat, Gaussian emissions). This is forward-only, so state_t never
depends on future bars — and it is computed for every bar in a single O(n·k²) pass,
giving identical features for both training and live inference.
"""

import logging
import warnings

import numpy as np

from . import config

log = logging.getLogger(__name__)

REGIME_FEATURES = ("log_ret", "abs_ret")


class RegimeModel:
    def __init__(self, n_states: int, n_iter: int | None = None, covariance_type: str | None = None):
        self.n_states = n_states
        self.n_iter = n_iter if n_iter is not None else config.HMM_N_ITER
        self.covariance_type = covariance_type if covariance_type is not None else config.HMM_COVARIANCE_TYPE
        self.min_covar = config.HMM_MIN_COVAR
        self._fitted = False
        self._feature_mean = None
        self._feature_scale = None
        self._startprob = None
        self._transmat = None
        self._means = None      # (k, d)
        self._inv_cov = None    # (k, d, d) full inverse covariance
        self._log_norm = None   # (k,) constant part of the multivariate-Gaussian log pdf
        self._high_vol_state = -1

    def fit(self, features: np.ndarray) -> None:
        """features: (n_obs, d) array, e.g. [log_ret, abs_ret]."""
        from hmmlearn.hmm import GaussianHMM

        self._fitted = False
        if features.shape[0] < self.n_states * 10:
            return
        features_scaled = self._standardize_for_fit(features)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    min_covar=self.min_covar,
                    tol=1e-3,
                    random_state=42,
                )
                model.fit(features_scaled)
        except Exception as exc:  # noqa: BLE001 - HMM fits can fail to converge
            log.warning("HMM fit failed (%s); regime layer disabled this window", exc)
            return

        means = np.asarray(model.means_, dtype=np.float64)            # (k, d)
        k, d = means.shape
        # hmmlearn's covars_ shape depends on covariance_type; normalise to (k, d, d).
        cov = self._to_full_cov(np.asarray(model.covars_, dtype=np.float64), k, d)
        cov = cov + np.eye(d)[None, :, :] * self.min_covar            # regularise

        self._startprob = np.clip(np.asarray(model.startprob_, dtype=np.float64), 1e-12, None)
        self._transmat = np.clip(np.asarray(model.transmat_, dtype=np.float64), 1e-12, None)
        self._means = means
        self._inv_cov = np.linalg.inv(cov)                           # (k, d, d)
        sign, logdet = np.linalg.slogdet(cov)
        self._log_norm = -0.5 * (d * np.log(2.0 * np.pi) + logdet)   # (k,)
        # Rank states by log_ret emission variance -> highest is "turbulent".
        self._high_vol_state = int(np.argmax(cov[:, 0, 0]))
        self._fitted = True

    def _standardize_for_fit(self, features: np.ndarray) -> np.ndarray:
        """Fit a past-only scaler and return standardized HMM features."""
        x = np.asarray(features, dtype=np.float64)
        finite = np.where(np.isfinite(x), x, np.nan)
        mean = np.nanmean(finite, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.nanstd(finite, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        self._feature_mean = mean
        self._feature_scale = scale
        return self._standardize(features)

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        mean = self._feature_mean
        scale = self._feature_scale
        if mean is None or scale is None:
            return np.nan_to_num(np.asarray(features, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        x = (np.asarray(features, dtype=np.float64) - mean) / scale
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _to_full_cov(cov: np.ndarray, k: int, d: int) -> np.ndarray:
        """Normalise hmmlearn covars_ (shape varies by covariance_type) to (k, d, d)."""
        if cov.ndim == 3 and cov.shape == (k, d, d):          # full
            return cov
        if cov.ndim == 2 and cov.shape == (d, d):             # tied
            return np.repeat(cov[None, :, :], k, axis=0)
        if cov.ndim == 2 and cov.shape == (k, d):             # diag
            return np.stack([np.diag(cov[i]) for i in range(k)])
        if cov.ndim == 1 and cov.shape == (k,):               # spherical
            return np.stack([np.eye(d) * cov[i] for i in range(k)])
        # Fallback: broadcast a diagonal from whatever is given.
        flat = np.broadcast_to(cov.reshape(-1)[:1], (k, d))
        return np.stack([np.diag(flat[i]) for i in range(k)])

    @property
    def fitted(self) -> bool:
        return self._fitted

    def _emission_logprob(self, X: np.ndarray) -> np.ndarray:
        """Multivariate-Gaussian log pdf per state. Returns (n, k)."""
        diff = X[:, None, :] - self._means[None, :, :]               # (n, k, d)
        tmp = np.einsum("nkd,kde->nke", diff, self._inv_cov)         # (n, k, d)
        quad = np.einsum("nke,nke->nk", tmp, diff)                   # (n, k)
        return self._log_norm[None, :] - 0.5 * quad

    def filtered_states(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Causal forward filter over all bars.

        Returns (states, high_vol_mask), both length n. When unfitted, returns
        all-zero states and an all-False mask.
        """
        n = features.shape[0]
        if not self._fitted or n == 0:
            return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)

        features_scaled = self._standardize(features)
        logB = self._emission_logprob(features_scaled)     # (n, k)
        k = self.n_states
        states = np.empty(n, dtype=np.int64)

        # t = 0
        logb = logB[0] - logB[0].max()
        alpha = self._startprob * np.exp(logb)
        alpha /= alpha.sum() if alpha.sum() > 0 else 1.0
        states[0] = int(np.argmax(alpha))

        for t in range(1, n):
            pred = alpha @ self._transmat                  # (k,)
            logb = logB[t] - logB[t].max()
            alpha = pred * np.exp(logb)
            s = alpha.sum()
            alpha = alpha / s if s > 0 else np.full(k, 1.0 / k)
            states[t] = int(np.argmax(alpha))

        high_vol_mask = states == self._high_vol_state
        return states, high_vol_mask
