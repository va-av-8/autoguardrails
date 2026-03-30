"""
Zero-shot LLM baseline for OOS detection.

Uses LLM to classify whether a query belongs to known intents
or is out-of-scope.
"""

from __future__ import annotations
import numpy as np


class ZeroShotLLM:
    """
    Zero-shot OOS detection using LLM.

    Prompt template:
    "Given the following intents: {intent_list}
     Classify this query: {query}
     Is this query in-scope (belongs to one of the intents)
     or out-of-scope? Answer: in-scope/out-of-scope"

    Attributes:
        intent_descriptions: dict mapping intent_id -> description
        model: LLM model name
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
    ):
        """
        Initialize zero-shot LLM classifier.

        Args:
            model: LLM model to use
            temperature: sampling temperature
        """
        self.model = model
        self.temperature = temperature
        self.intent_descriptions = None

    def fit(self, intent_descriptions: dict[int, str]) -> "ZeroShotLLM":
        """
        Set intent descriptions for classification.

        Args:
            intent_descriptions: mapping from intent_id to description

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
            predicted labels (-1 for OOS)
        """
        # TODO: реализовать
        raise NotImplementedError

    def get_oos_scores(self, texts: list[str]) -> np.ndarray:
        """
        Get OOS confidence scores.

        Args:
            texts: list of text samples

        Returns:
            OOS confidence for each sample
        """
        # TODO: реализовать
        raise NotImplementedError
