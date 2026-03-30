"""
SOTA: DETER — Dual Encoder for Threshold-Based Re-Classification.

Ссылка: arxiv.org/abs/2405.19967 (2024)
GitHub: github.com/Hossam-Mohammed-tech/Intent_Classification_OOS

Метод:
- Dual encoder: USE (Universal Sentence Encoder) + TSDAE
  (Transformer-based Denoising AutoEncoder)
- Генерирует синтетические выбросы через self-supervision
- Threshold-based re-classification для уточнения предсказаний
- Превосходит ADB (AAAI 2021) на CLINC150, Banking77, StackOverflow:
  +13% и +5% F1 на CLINC150 и StackOverflow,
  +16% known и +24% unknown F1 на Banking77

Роль в эксперименте: SOTA (верхняя граница качества).
Интеграция: запускаем оригинальный код через subprocess или
адаптируем dataloader под наш pipeline.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


class DETERWrapper:
    """
    Обёртка над оригинальной реализацией DETER.

    Два варианта интеграции (выбрать при реализации):
    A) subprocess: запуск оригинального main.py с нашими данными
    B) adapter: импортируем модули DETER напрямую, подменяем dataloader

    Рекомендуется вариант B для единообразного pipeline.
    """

    def __init__(
        self,
        deter_repo_path: Path | None = None,
        device: str = "cpu",
    ):
        """
        Args:
            deter_repo_path: путь к склонированному репозиторию DETER.
                             Если None — ищем в ./external/deter/
            device: "cpu" или "cuda"
        """
        self.deter_repo_path = deter_repo_path or Path("external/deter")
        self.device = device
        self.model = None

    def fit(self, train_texts: list[str], train_labels: list[int]) -> None:
        """
        Обучает DETER на обучающей выборке.

        Args:
            train_texts: тексты (in-scope)
            train_labels: intent-метки
        """
        # TODO: реализовать
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        OOS-скор для каждого текста.

        Returns:
            np.ndarray shape (len(texts),)
        """
        # TODO: реализовать
        raise NotImplementedError

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Бинарное предсказание: 1 = OOS, 0 = in-scope.

        Returns:
            np.ndarray shape (len(texts),)
        """
        # TODO: реализовать
        raise NotImplementedError
