"""Layer 1 — Market regime via Hidden Markov Model.

A Gaussian HMM (diagonal covariance) is fit on [log_return, |log_return|]. Hidden
states are ranked by their emission variance so the highest-variance state is flagged
as the "turbulent" regime.

For per-bar state assignment we use a **causal forward filter**: the filtered
posterior P(state_t | observations up to t) is computed from the fitted parameters
(startprob, transmat, Gaussian emissions). This is forward-only, so state_t never
depends on future bars — and it is computed for every bar in a single O(n·k²) pass,
giving identical features for both training and live inference.
"""

import logging
import warnings

import numpy as np

log = logging.getLogger(__name__)

REGIME_FEATURES = ("log_ret", "abs_ret")


class RegimeModel:
    def __init__(self, n_states: int):
        self.n_states = n_states
        self._fitted = False
        self._startprob = None
        self._transmat = None
        self._means = None      # (k, d)
        self._inv_var = None    # (k, d)
        self._log_norm = None   # (k,) constant part of the diag-Gaussian log pdf
        self._high_vol_state = -1

    def fit(self, features: np.ndarray) -> None:
        """features: (n_obs, d) array, e.g. [log_ret, abs_ret]."""
        from hmmlearn.hmm import GaussianHMM

        self._fitted = False
        if features.shape[0] < self.n_states * 10:
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="diag",
                    n_iter=100,
                    tol=1e-3,
                    random_state=42,
                )
                model.fit(features)
        except Exception as exc:  # noqa: BLE001 - HMM fits can fail to converge
            log.warning("HMM fit failed (%s); regime layer disabled this window", exc)
            return

        means = np.asarray(model.means_, dtype=np.float64)            # (k, d)
        covars = np.asarray(model.covars_, dtype=np.float64)
        var = covars[:, np.arange(means.shape[1]), np.arange(means.shape[1])] if covars.ndim == 3 else covars
        var = np.clip(var, 1e-12, None)                               # (k, d)

        self._startprob = np.clip(np.asarray(model.startprob_, dtype=np.float64), 1e-12, None)
        self._transmat = np.clip(np.asarray(model.transmat_, dtype=np.float64), 1e-12, None)
        self._means = means
        self._inv_var = 1.0 / var
        self._log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * var), axis=1)  # (k,)
        # Rank states by log_ret emission variance -> highest is "turbulent".
        self._high_vol_state = int(np.argmax(var[:, 0]))
        self._fitted = True

    @property
    def fitted(self) -> bool:
        return self._fitted

    def _emission_logprob(self, X: np.ndarray) -> np.ndarray:
        """Diagonal-Gaussian log pdf per state. Returns (n, k)."""
        # (n, 1, d) - (1, k, d) -> (n, k, d)
        diff = X[:, None, :] - self._means[None, :, :]
        quad = np.sum((diff ** 2) * self._inv_var[None, :, :], axis=2)  # (n, k)
        return self._log_norm[None, :] - 0.5 * quad

    def filtered_states(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Causal forward filter over all bars.

        Returns (states, high_vol_mask), both length n. When unfitted, returns
        all-zero states and an all-False mask.
        """
        n = features.shape[0]
        if not self._fitted or n == 0:
            return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)

        logB = self._emission_logprob(features)            # (n, k)
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
