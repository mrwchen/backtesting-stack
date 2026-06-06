"""Layer 4 -- Trade decision from calibrated trade-outcome probabilities.

The classifier predicts whether a full candidate trade wins after costs. Long and
short are modelled separately because their feature/payoff distributions can be
very different in index scalping.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config

log = logging.getLogger(__name__)

FEATURE_COLUMNS = (
    "regime_state",
    "high_vol_state",
    "price_deviation_basis",
    "slope_basis",
    "sigma_ret",
    "momentum",
    "rsi_centered",
    "session_progress",
)


@dataclass(frozen=True)
class PayoffStats:
    avg_win_r: float = 1.0
    avg_loss_r: float = -1.0
    fitted: bool = False


@dataclass(frozen=True)
class DecisionScores:
    prob_long_win: float
    prob_short_win: float
    expected_long_r: float
    expected_short_r: float
    long_fitted: bool
    short_fitted: bool


def make_decision_model(model: str):
    if model in ("bayes", "logistic"):
        return DecisionModel(model)
    raise ValueError(f"Unknown DECISION_MODEL {model!r}")


class DecisionModel:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._long_clf = None
        self._short_clf = None
        self._long_stats = PayoffStats()
        self._short_stats = PayoffStats()

    def _base_estimator(self):
        if self.kind == "bayes":
            from sklearn.naive_bayes import GaussianNB

            return GaussianNB()
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, C=config.LOGISTIC_C),
        )

    def _new_estimator(self, y: np.ndarray):
        base = self._base_estimator()
        if not config.CALIBRATE_PROBABILITIES:
            return base

        classes, counts = np.unique(y, return_counts=True)
        if classes.shape[0] < 2:
            return base
        cv = int(min(3, counts.min()))
        if cv < 2 or y.shape[0] < max(config.MIN_TRAIN_ROWS * 2, 100):
            return base

        from sklearn.calibration import CalibratedClassifierCV

        try:
            return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv)
        except TypeError:
            return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=cv)

    @staticmethod
    def _payoff_stats(net_r: np.ndarray) -> PayoffStats:
        r = np.asarray(net_r, dtype=np.float64)
        r = r[np.isfinite(r)]
        if r.size == 0:
            return PayoffStats()
        wins = r > 0.0
        losses = ~wins
        avg_win = float(r[wins].mean()) if wins.any() else 1.0
        avg_loss = float(r[losses].mean()) if losses.any() else -1.0
        return PayoffStats(avg_win_r=avg_win, avg_loss_r=avg_loss, fitted=True)

    def _fit_side(self, side: str, X: np.ndarray, y: np.ndarray, net_r: np.ndarray):
        if X.shape[0] < config.MIN_TRAIN_ROWS:
            log.info("%s decision side %s skipped train rows %d below minimum", self.kind, side, X.shape[0])
            return None, self._payoff_stats(net_r)

        mask = np.isfinite(y) & np.isfinite(net_r) & np.all(np.isfinite(X), axis=1)
        X_fit = X[mask]
        y_fit = y[mask].astype(float)
        r_fit = net_r[mask].astype(float)
        stats = self._payoff_stats(r_fit)
        if X_fit.shape[0] < config.MIN_TRAIN_ROWS:
            return None, stats

        classes = np.unique(y_fit)
        if classes.shape[0] < 2:
            log.info("%s decision side %s has one class only; side disabled", self.kind, side)
            return None, stats

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = self._new_estimator(y_fit)
                clf.fit(X_fit, y_fit)
            return clf, stats
        except Exception as exc:  # noqa: BLE001
            log.warning("%s decision side %s fit failed (%s); side disabled", self.kind, side, exc)
            return None, stats

    def fit(
        self,
        X_long: np.ndarray,
        y_long: np.ndarray,
        net_r_long: np.ndarray,
        X_short: np.ndarray,
        y_short: np.ndarray,
        net_r_short: np.ndarray,
    ) -> None:
        self._long_clf, self._long_stats = self._fit_side("LONG", X_long, y_long, net_r_long)
        self._short_clf, self._short_stats = self._fit_side("SHORT", X_short, y_short, net_r_short)

    @property
    def fitted(self) -> bool:
        return self._long_clf is not None or self._short_clf is not None

    @staticmethod
    def _positive_proba(clf, X: np.ndarray) -> np.ndarray:
        if clf is None:
            return np.full(X.shape[0], 0.5, dtype=np.float64)
        proba = clf.predict_proba(X)
        classes = list(clf.classes_)
        idx = classes.index(1.0) if 1.0 in classes else int(np.argmax(classes))
        return proba[:, idx].astype(np.float64)

    @staticmethod
    def _expected_r(p: np.ndarray, stats: PayoffStats) -> np.ndarray:
        return p * stats.avg_win_r + (1.0 - p) * stats.avg_loss_r

    def score(self, X: np.ndarray) -> DecisionScores:
        p_long = float(self._positive_proba(self._long_clf, X)[0])
        p_short = float(self._positive_proba(self._short_clf, X)[0])
        exp_long = float(self._expected_r(np.array([p_long]), self._long_stats)[0])
        exp_short = float(self._expected_r(np.array([p_short]), self._short_stats)[0])
        return DecisionScores(
            prob_long_win=p_long,
            prob_short_win=p_short,
            expected_long_r=exp_long,
            expected_short_r=exp_short,
            long_fitted=self._long_clf is not None,
            short_fitted=self._short_clf is not None,
        )
