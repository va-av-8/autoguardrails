"""
TF-IDF + Logistic Regression baseline.

Два режима:
- argmax: OOS как дополнительный класс (воспроизводит AutoIntent Table 3)
- threshold: с калиброванным порогом по validation set (честный бейзлайн)
"""

from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfLogreg:
    """
    TF-IDF + Logistic Regression classifier.

    Интерфейс:
        model = TfidfLogreg()
        model.fit(train_texts, train_labels)
        model.calibrate_threshold(val_texts, val_labels)  # опционально
        y_pred = model.predict(test_texts)
        y_scores = model.predict_proba(test_texts)
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        C: float = 1.0,
        class_weight: str = "balanced",
    ):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                C=C,
                max_iter=1000,
                class_weight=class_weight,
                n_jobs=-1,
            )),
        ])
        self._oos_label = -1
        self._oos_class_idx = None
        self.threshold_: float | None = None

    def fit(self, texts: list[str], labels: list[int]) -> "TfidfLogreg":
        """Fit the model."""
        self.pipeline.fit(texts, labels)
        classes = self.pipeline.named_steps["clf"].classes_
        if self._oos_label in classes:
            self._oos_class_idx = np.where(classes == self._oos_label)[0][0]
        else:
            self._oos_class_idx = None
        return self

    def calibrate_threshold(
        self,
        val_texts: list[str],
        val_labels: list[int],
        n_thresholds: int = 50,
    ) -> float:
        """
        Подбирает оптимальный порог на validation set по F1 OOS.
        Сохраняет в self.threshold_.
        """
        from shared.metrics import f1_oos as _f1_oos

        oos_scores = self.predict_proba(val_texts)
        proba = self.pipeline.predict_proba(val_texts)
        classes = self.pipeline.named_steps["clf"].classes_
        val_labels = np.array(val_labels)

        thresholds = np.linspace(oos_scores.min(), oos_scores.max(), n_thresholds)
        best_f1, best_t = 0.0, thresholds[0]

        for t in thresholds:
            y_pred = self._predict_with_threshold(oos_scores, proba, classes, t)
            f1 = _f1_oos(val_labels, y_pred)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        self.threshold_ = best_t
        return best_t

    def _predict_with_threshold(
        self,
        oos_scores: np.ndarray,
        proba: np.ndarray,
        classes: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Применяет порог: если oos_score >= threshold -> -1, иначе argmax по in-scope."""
        predictions = []
        for i, score in enumerate(oos_scores):
            if score >= threshold:
                predictions.append(-1)
            else:
                p = proba[i].copy()
                if self._oos_class_idx is not None:
                    p[self._oos_class_idx] = -np.inf
                predictions.append(classes[np.argmax(p)])
        return np.array(predictions)

    def predict(self, texts: list[str], use_threshold: bool = True) -> np.ndarray:
        """
        Predict class labels.

        Args:
            texts: list of text samples
            use_threshold: если True и threshold_ установлен, использует порог

        Returns:
            predicted labels (-1 for OOS)
        """
        if use_threshold and self.threshold_ is not None:
            oos_scores = self.predict_proba(texts)
            proba = self.pipeline.predict_proba(texts)
            classes = self.pipeline.named_steps["clf"].classes_
            return self._predict_with_threshold(oos_scores, proba, classes, self.threshold_)
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Get OOS probability scores.

        Returns:
            OOS probability for each sample (higher = more OOS)
        """
        proba = self.pipeline.predict_proba(texts)
        if self._oos_class_idx is not None:
            return proba[:, self._oos_class_idx]
        return np.zeros(len(texts))
