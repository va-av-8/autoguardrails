"""
Бейзлайн B: Cosine Similarity + Threshold для OOS-детекции.

Логика: если максимальная косинусная близость запроса к любому
обучающему примеру ниже порога — запрос считается OOS.

Поддерживает два варианта модели через параметр model_name:
- "bert-base-uncased": замороженный BERT без дообучения.
  Используется в литературе как нижняя граница embedding-методов
  (ADB AAAI 2021, DCLOOS ACL 2021). Слабее, но сравним с опубл. числами.
- "sentence-transformers/all-MiniLM-L6-v2": дообучен на semantic
  similarity, даёт лучше zero-shot качество.

Оба варианта — заморожены, дообучения нет.
"""

from __future__ import annotations
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


SUPPORTED_MODELS = [
    "bert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling — усредняем токены с учётом attention mask.
    Стандартный способ получить sentence embedding из токен-эмбеддингов.
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)


def get_embeddings(
    texts: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """
    Вычисляет эмбеддинги для списка текстов батчами.

    Args:
        texts: список входных текстов
        model: замороженная HuggingFace модель
        tokenizer: соответствующий токенизатор
        batch_size: размер батча
        device: "cpu" или "cuda"

    Returns:
        np.ndarray shape (len(texts), hidden_dim)
    """
    # TODO: реализовать
    raise NotImplementedError


class EmbeddingThreshold:
    """
    Cosine similarity baseline для OOS-детекции.

    Параметр model_name позволяет запускать два варианта
    одной командой без дублирования кода:

        # вариант 1 — для связи с литературой
        model = EmbeddingThreshold(model_name="bert-base-uncased")

        # вариант 2 — сильнее zero-shot
        model = EmbeddingThreshold(model_name="sentence-transformers/all-MiniLM-L6-v2")

        model.fit(train_texts, train_labels)
        oos_scores = model.predict_proba(test_texts)

    OOS-скор = 1 - max_cosine_similarity к обучающим примерам.
    Чем выше скор — тем более OOS запрос.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        threshold: float = 0.85,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """
        Args:
            model_name: название модели из SUPPORTED_MODELS
            threshold: порог cosine similarity (ниже = OOS)
            batch_size: батч для инференса
            device: "cpu" или "cuda"
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"model_name должен быть одним из {SUPPORTED_MODELS}")
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = None
        self.model = None
        self.train_embeddings = None  # np.ndarray (n_train, hidden_dim)
        self.train_labels = None      # np.ndarray (n_train,)

    def load_model(self) -> None:
        """
        Загружает токенизатор и модель с HuggingFace.
        Вызывается лениво при первом fit() или predict_proba().
        """
        # TODO: реализовать
        raise NotImplementedError

    def fit(self, texts: list[str], labels: list[int]) -> None:
        """
        Запоминает эмбеддинги обучающих примеров.
        Дообучения нет — модель заморожена.

        Args:
            texts: обучающие тексты (in-scope)
            labels: intent-метки (OOS-примеры не передаются)
        """
        # TODO: реализовать
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        OOS-скор для каждого текста.
        OOS-скор = 1 - max_cosine_similarity к train.
        Чем выше — тем более OOS.

        Returns:
            np.ndarray shape (len(texts),), значения в [0, 1]
        """
        # TODO: реализовать
        raise NotImplementedError

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Бинарное предсказание: 1 = OOS, 0 = in-scope.
        Порог применяется к OOS-скору.

        Returns:
            np.ndarray shape (len(texts),), значения в {0, 1}
        """
        # TODO: реализовать
        raise NotImplementedError
