"""Layer 4 — Trade decision: Bayesian classifier | probabilistic (logistic) model.

Both predict P(next bar up) from the feature vector
    [regime_state, price_deviation, slope, sigma, momentum, rsi].
The classifier is refit walk-forward on past-only (feature, label) pairs, where the
label is the sign of the *next* bar's return (already shifted in data.build_features).

  bayes    -> sklearn GaussianNB                (generative Bayesian classifier)
  logistic -> sklearn LogisticRegression        (discriminative probabilistic model)
"""

import logging
import warnings

import numpy as np

log = logging.getLogger(__name__)

FEATURE_COLUMNS = ("regime_state", "price_deviation", "slope", "sigma", "momentum", "rsi")


def make_decision_model(model: str):
    if model in ("bayes", "logistic"):
        return DecisionModel(model)
    raise ValueError(f"Unknown DECISION_MODEL {model!r}")


class DecisionModel:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._clf = None
        self._single_class = None  # set if training data had only one label

    def _new_estimator(self):
        if self.kind == "bayes":
            from sklearn.naive_bayes import GaussianNB

            return GaussianNB()
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=200, C=1.0),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._clf = None
        self._single_class = None
        if X.shape[0] < 50:
            return
        classes = np.unique(y)
        if classes.shape[0] < 2:
            # Degenerate: only ups or only downs in the window -> no usable edge.
            self._single_class = float(classes[0]) if classes.shape[0] else None
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = self._new_estimator()
                clf.fit(X, y)
            self._clf = clf
        except Exception as exc:  # noqa: BLE001
            log.warning("%s decision fit failed (%s); decisions disabled this window", self.kind, exc)
            self._clf = None

    @property
    def fitted(self) -> bool:
        return self._clf is not None

    def proba_up(self, X: np.ndarray) -> np.ndarray:
        """P(up) for each row; returns 0.5 (no edge) when unfitted/degenerate."""
        if self._clf is None:
            if self._single_class is not None:
                return np.full(X.shape[0], self._single_class, dtype=np.float64)
            return np.full(X.shape[0], 0.5, dtype=np.float64)
        proba = self._clf.predict_proba(X)
        # Locate the column for the positive ("up", label==1.0) class.
        classes = list(self._clf.classes_)
        up_idx = classes.index(1.0) if 1.0 in classes else int(np.argmax(classes))
        return proba[:, up_idx].astype(np.float64)
