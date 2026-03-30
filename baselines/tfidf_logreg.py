"""
TF-IDF + Logistic Regression baseline.

OOS обрабатывается как дополнительный класс — намеренно слабый подход,
служит нижней границей качества.
"""

from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TfidfLogreg:
    """
    TF-IDF + Logistic Regression classifier.

    OOS добавляется как отдельный класс (label = n_intents).
    Это baseline без специальной поддержки OOS.

    Attributes:
        vectorizer: TF-IDF vectorizer
        classifier: Logistic Regression classifier
        n_classes: number of classes (including OOS)
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
    ):
        """
        Initialize TF-IDF + LogReg model.

        Args:
            max_features: max number of TF-IDF features
            ngram_range: n-gram range for TF-IDF
            sublinear_tf: use sublinear TF scaling
            C: regularization strength for LogReg
            max_iter: max iterations for LogReg
            class_weight: class weighting strategy
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
        )
        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
        )
        self.n_classes = None

    def fit(self, texts: list[str], labels: list[int]) -> "TfidfLogreg":
        """
        Fit the model.

        Args:
            texts: list of text samples
            labels: list of integer labels (OOS = max label)

        Returns:
            self
        """
        # TODO: реализовать
        raise NotImplementedError

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Predict class labels.

        Args:
            texts: list of text samples

        Returns:
            predicted labels
        """
        # TODO: реализовать
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            texts: list of text samples

        Returns:
            probability matrix (n_samples, n_classes)
        """
        # TODO: реализовать
        raise NotImplementedError

    def get_oos_scores(self, texts: list[str]) -> np.ndarray:
        """
        Get OOS probability scores.

        Args:
            texts: list of text samples

        Returns:
            OOS probability for each sample
        """
        # TODO: реализовать
        raise NotImplementedError
