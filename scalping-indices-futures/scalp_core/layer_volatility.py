"""Layer 3 — Volatility: GARCH | EGARCH (switchable, via the `arch` package).

Conditional variance h_t in a GARCH/EGARCH model depends only on past returns and
past variance, so the one-step-ahead conditional volatility for bar t is known at
the close of bar t-1 — i.e. it is a causal forecast usable for sizing the trade
opened on bar t.

Workflow (mirrors the price layer):
  * update_params(returns_train)  — MLE-fit on a past-only slice, store params+scale.
  * conditional_vol(returns_full) — re-apply fixed params over the full series and
    read the (causal) conditional volatility, returned as a *fractional* return std.
"""

import logging
import warnings

import numpy as np

from . import config

log = logging.getLogger(__name__)


def make_vol_model(model: str):
    if model in ("garch", "egarch"):
        return VolModel(model)
    raise ValueError(f"Unknown VOL_MODEL {model!r}")


class VolModel:
    def __init__(self, kind: str) -> None:
        self.kind = kind  # "garch" | "egarch"
        self._params = None
        self._scale = 1.0
        self._fallback_sigma = 0.0  # fractional return std fallback

    def _spec(self, scaled_returns: np.ndarray):
        from arch import arch_model

        vol = "EGARCH" if self.kind == "egarch" else "GARCH"
        o = 1 if self.kind == "egarch" else 0
        return arch_model(
            scaled_returns, mean="Zero", vol=vol,
            p=config.GARCH_P, o=o, q=config.GARCH_Q, dist=config.GARCH_DIST,
        )

    def update_params(self, returns_train: np.ndarray) -> None:
        r = returns_train[np.isfinite(returns_train)]
        std = float(np.std(r)) if r.size else 0.0
        self._fallback_sigma = std if (np.isfinite(std) and std > 0) else 1e-6
        # Scale returns so the optimiser sees O(1) magnitudes.
        self._scale = 1.0 / self._fallback_sigma if self._fallback_sigma > 0 else 1.0

        if r.size < 100:
            self._params = None
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = self._spec(r * self._scale).fit(disp="off", show_warning=False)
            self._params = res.params
        except Exception as exc:  # noqa: BLE001 - GARCH MLE can fail to converge
            log.warning("%s fit failed (%s); using rolling-std fallback", self.kind, exc)
            self._params = None

    def conditional_vol(self, returns_full: np.ndarray) -> np.ndarray:
        """Return causal fractional-return volatility per bar (same length as input)."""
        n = returns_full.shape[0]
        r = np.nan_to_num(returns_full, nan=0.0)

        if self._params is None:
            # EWMA fallback (RiskMetrics-style), causal.
            return self._ewma_sigma(r)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = self._spec(r * self._scale).fix(self._params)
            cond_vol_scaled = np.asarray(res.conditional_volatility, dtype=np.float64)
            sigma = cond_vol_scaled / self._scale  # back to fractional return units
            sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, self._fallback_sigma)
            if sigma.shape[0] != n:  # safety: align length
                sigma = np.resize(sigma, n)
            return sigma
        except Exception as exc:  # noqa: BLE001
            log.warning("%s fixed-param filter failed (%s); using rolling-std fallback", self.kind, exc)
            return self._ewma_sigma(r)

    @staticmethod
    def _ewma_sigma(r: np.ndarray, lam: float = 0.94) -> np.ndarray:
        n = r.shape[0]
        sigma = np.empty(n, dtype=np.float64)
        var = float(np.var(r[: min(n, 50)])) if n else 1e-12
        var = var if (np.isfinite(var) and var > 0) else 1e-12
        for t in range(n):
            sigma[t] = np.sqrt(var)
            var = lam * var + (1.0 - lam) * (r[t] ** 2)  # update for next bar (causal)
        return sigma
