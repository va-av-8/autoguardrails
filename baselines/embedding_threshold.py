"""
Cosine Similarity + Threshold baseline для OOS-детекции.

Два режима:
- argmax: nearest neighbor с дефолтным порогом (без калибровки)
- threshold: с калиброванным порогом по validation set

Поддерживает:
- bert-base-uncased: замороженный BERT (нижняя граница)
- sentence-transformers/all-MiniLM-L6-v2: дообучен на semantic similarity
"""

from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


SUPPORTED_MODELS = [
    "bert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
]


class EmbeddingThreshold:
    """
    Cosine similarity baseline для OOS-детекции.
    OOS-скор = 1 - max_cosine_similarity к train.
    """

    SUPPORTED_MODELS = SUPPORTED_MODELS

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        threshold: float = 0.85,
        batch_size: int = 32,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ):
        """
        Args:
            model_name: название модели из SUPPORTED_MODELS
            threshold: дефолтный порог cosine similarity (используется без калибровки)
            batch_size: батч для инференса
            device: "cpu" или "cuda"
            cache_dir: директория для кеширования эмбеддингов (None = без кеша)
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"model_name должен быть одним из {SUPPORTED_MODELS}")
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.tokenizer = None
        self.model = None
        self.train_embeddings = None
        self.train_labels = None
        self.threshold_: float | None = None

    def _load_model(self) -> None:
        """Загружает токенизатор и модель с HuggingFace."""
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _cache_key(self, texts: list[str]) -> str:
        """
        Уникальный ключ кеша для данного набора текстов и модели.
        Использует первые 10 текстов + длину списка как fingerprint.
        """
        fingerprint = " ".join(texts[:10]) + str(len(texts))
        texts_hash = hashlib.md5(fingerprint.encode()).hexdigest()[:8]
        model_safe = self.model_name.replace("/", "_")
        return f"{model_safe}_{texts_hash}_{len(texts)}"

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Mean pooling поверх last_hidden_state.
        Если cache_dir задан — читает/пишет кеш на диск.
        """
        # Проверить кеш
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{self._cache_key(texts)}.npy"
            if cache_file.exists():
                return np.load(cache_file)

        # Считать эмбеддинги
        self._load_model()
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"]
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = (token_embeddings * mask_expanded).sum(1)
                sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
                embeddings = sum_embeddings / sum_mask
                all_embeddings.append(embeddings.cpu().numpy())

        result = np.vstack(all_embeddings)

        # Сохранить в кеш
        if self.cache_dir is not None:
            np.save(cache_file, result)

        return result

    def fit(self, texts: list[str], labels: list[int]) -> "EmbeddingThreshold":
        """Запоминает эмбеддинги только in-scope примеров."""
        inscope_texts = []
        inscope_labels = []
        for text, label in zip(texts, labels):
            if label != -1:
                inscope_texts.append(text)
                inscope_labels.append(label)

        self.train_embeddings = self._get_embeddings(inscope_texts)
        self.train_labels = np.array(inscope_labels)
        return self

    def calibrate_threshold(
        self,
        val_texts: list[str],
        val_labels: list[int],
        n_thresholds: int = 50,
    ) -> float:
        """
        Подбирает оптимальный порог на validation set по F1 OOS.
        Эмбеддинги для val считаются ровно один раз.
        """
        from shared.metrics import f1_oos as _f1_oos

        # Считаем эмбеддинги ОДИН раз
        query_embeddings = self._get_embeddings(val_texts)
        sim_matrix = cosine_similarity(query_embeddings, self.train_embeddings)
        max_sim = sim_matrix.max(axis=1)
        oos_scores = 1.0 - max_sim
        nearest_idx = sim_matrix.argmax(axis=1)

        val_labels_arr = np.array(val_labels)
        thresholds = np.linspace(oos_scores.min(), oos_scores.max(), n_thresholds)
        best_f1, best_t = 0.0, thresholds[0]

        for t in thresholds:
            y_pred = np.where(
                oos_scores >= t,
                -1,
                self.train_labels[nearest_idx],
            )
            f1 = _f1_oos(val_labels_arr, y_pred)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        self.threshold_ = best_t
        return best_t

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        OOS-скор для каждого текста.
        OOS-скор = 1 - max_cosine_similarity к train.
        """
        query_embeddings = self._get_embeddings(texts)
        sim_matrix = cosine_similarity(query_embeddings, self.train_embeddings)
        max_sim = sim_matrix.max(axis=1)
        return 1.0 - max_sim

    def predict(self, texts: list[str], use_threshold: bool = True) -> np.ndarray:
        """
        Предсказание: OOS (-1) или ближайший in-scope intent.

        Args:
            texts: list of text samples
            use_threshold: если True и threshold_ установлен, использует калиброванный порог

        Returns:
            predicted labels (-1 for OOS)
        """
        query_embeddings = self._get_embeddings(texts)
        sim_matrix = cosine_similarity(query_embeddings, self.train_embeddings)
        max_sim = sim_matrix.max(axis=1)
        nearest_idx = sim_matrix.argmax(axis=1)

        if use_threshold and self.threshold_ is not None:
            oos_scores = 1.0 - max_sim
            return np.where(
                oos_scores >= self.threshold_,
                -1,
                self.train_labels[nearest_idx]
            )
        else:
            return np.where(
                max_sim < self.threshold,
                -1,
                self.train_labels[nearest_idx]
            )
