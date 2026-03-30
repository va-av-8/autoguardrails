"""
Guardrail Reference: govtech/stsb-roberta-base-off-topic.

Ссылка: arxiv.org/abs/2411.12946 (Chua et al., 2024)
HuggingFace: huggingface.co/govtech/stsb-roberta-base-off-topic

ВАЖНО: это НЕ SOTA по метрикам на стандартных бенчмарках.
Роль в эксперименте — reference по постановке задачи:
авторы решают ту же задачу (off-topic guardrail для LLM),
используют ту же framing (релевантен ли промпт системному промпту).

Почему включаем:
- Единственная открытая модель, специально обученная под off-topic guardrail
- Методологически: показывает что бывает при domain-specific fine-tuning
- Связь с будущими задачами: авторы показывают обобщение на jailbreak

Ограничения:
- Обучена на синтетических данных, не на CLINC150
- Нет peer-reviewed числа на наших метриках (Recall@FPR)
- Результаты на нашем тест-сплите могут быть хуже специализированных методов
"""

from __future__ import annotations
import numpy as np


MODEL_NAME = "govtech/stsb-roberta-base-off-topic"


class GuardrailReference:
    """
    Off-topic guardrail модель от GovTech Singapore.

    Модель классифицирует: релевантен ли запрос пользователя
    заданному системному промпту. Это отличается от классической
    OOS-детекции — здесь нет intent-классов, только бинарное решение.

    Адаптация к нашему pipeline:
    - fit() принимает тексты, но модель не обучается (zero-shot)
    - system_prompt задаётся при инициализации или через set_system_prompt()
    - predict_proba() возвращает OOS-скор (вероятность off-topic)
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        device: str = "cpu",
    ):
        """
        Args:
            system_prompt: описание мандата системы.
                           Если None — нужно задать через set_system_prompt().
            device: "cpu" или "cuda"
        """
        self.system_prompt = system_prompt
        self.device = device
        self.model = None
        self.tokenizer = None

    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Задаёт системный промпт (описание мандата агента).
        Для CLINC150: краткое описание из 150 intent-категорий.
        """
        self.system_prompt = system_prompt

    def load_model(self) -> None:
        """Загружает модель с HuggingFace."""
        # TODO: реализовать
        raise NotImplementedError

    def fit(self, texts: list[str], labels: list[int]) -> None:
        """
        Нет обучения — модель zero-shot.
        Метод существует для единообразия интерфейса с другими бейзлайнами.
        """
        # просто загружаем модель если ещё не загружена
        # TODO: вызвать load_model()
        pass

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        OOS-скор = вероятность off-topic для каждого текста.

        Returns:
            np.ndarray shape (len(texts),)
        """
        # TODO: реализовать
        raise NotImplementedError

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Бинарное предсказание: 1 = OOS/off-topic, 0 = in-scope.

        Returns:
            np.ndarray shape (len(texts),)
        """
        # TODO: реализовать
        raise NotImplementedError
