# Jailbreak Detection

## Постановка задачи

Детекция попыток jailbreak на input, до передачи промпта в LLM.
OOS-переформулировка: safe = in-scope интент, jailbreak = OOS-класс.

Задача структурно аналогична OOS-детекции, но данные adversarial
по природе — jailbreak-промпты специально сконструированы, чтобы
обходить защиты LLM.

## Датасет

**WildJailbreak** (Jiang et al., NeurIPS 2024)
- HuggingFace: `allenai/wildjailbreak`
- Train: 261 559 примеров
- Eval: 2 210 примеров (используется как test)
- Колонки: `vanilla` (str), `adversarial` (str), `completion` (str),
  `data_type` (str)
- Prompt-колонка: `adversarial` если непустой, иначе `vanilla`
- Бинарная метка: `*_harmful` → jailbreak (1), `*_benign` → safe (0)

**Распределение eval:**
| data_type | count | binary_label |
|-----------|-------|--------------|
| adversarial_harmful | 2000 | jailbreak |
| adversarial_benign | 210 | safe |

Vanilla-примеров в eval нет.

**Распределение train (приблизительно):**
| data_type | count |
|-----------|-------|
| vanilla_harmful | 50 050 |
| vanilla_benign | 50 050 |
| adversarial_harmful | 82 728 |
| adversarial_benign | 78 706 |

**Предупреждение:** датасет содержит примеры тяжёлого вреда,
включая CSAM-категорию. При отображении примеров избегать
воспроизведения контента из этих категорий.

## Few-shot протокол

- Balanced sampling: N примеров на класс (safe / jailbreak)
- n_shots: [10, 20, 50], seeds: [42, 123, 456], итого 9 runs
- Естественное распределение (~10:1 jailbreak/safe) сохранено
  только в test-сплите
- Соответствует практике WildGuard (NeurIPS 2024):
  uniform mixture при обучении и оценке

## Метрики

Все функции в `shared/metrics.py`, параметр `oos_label=1`.

### Основные

| Метрика | Описание | Функция |
|---------|----------|---------|
| **F1** | F1-score на jailbreak-классе | `f1_oos(y_true, y_pred, oos_label=1)` |
| **Precision** | TP / (TP + FP) на jailbreak | `precision_oos(y_true, y_pred, oos_label=1)` |
| **Recall** | TPR на jailbreak-классе | `oos_recall(y_true, y_pred, oos_label=1)` |
| **Over-refusal Rate** | FPR на safe-промптах | `over_refusal_rate(y_true, y_pred, oos_label=1)` |

### По подмножествам

| Метрика | Описание |
|---------|----------|
| **recall_vanilla_harmful** | Recall на vanilla_harmful (недоступен в текущем eval) |
| **recall_adversarial_harmful** | Recall на adversarial_harmful |

Агрегатор: `evaluate_jailbreak(y_true, y_pred, data_types, oos_label=1)`.

## SOTA-модели

| Модель | Параметры | Тип | Источник |
|--------|-----------|-----|----------|
| PromptGuard 2 | 86M | BERT-style classifier | LlamaFirewall, arxiv:2505.03574 |
| Qwen3Guard-Gen | 4B / 8B | Generative LLM | arxiv:2510.14276 |

**Примечания по выбору:**
- WildGuard (NeurIPS 2024) исключён: обучен на WildGuardMix,
  включающем данные из WildJailbreak — data leakage на нашем тесте.
- Бейзлайны из OOS-спринта (TF-IDF, cosine) не запускаются:
  adversarial промпты специально маскируют вредоносный intent,
  bag-of-words методы предсказуемо деградируют.
- SOTA запускается в zero-shot режиме только на тестовом сплите.
- PromptGuard 2: SOTA-статус заявлен авторами (Meta),
  независимых рецензируемых оценок нет.
- Qwen3Guard: независимо оценён в arxiv:2511.22047,
  показывает generalization gap −57 п.п. на novel prompts.

## Гипотезы

| # | Гипотеза | Статус |
|---|----------|--------|
| HYP-JB-001 | Обобщение OOS-пайплайна на adversarial данные | запланирована |
| HYP-JB-002 | Vanilla vs adversarial generalization gap | запланирована (ограничение: eval содержит только adversarial) |

## Ноутбуки

| Ноутбук | Содержание |
|---------|------------|
| `01_eda.ipynb` | EDA датасета WildJailbreak |
| `02_sota_eval.ipynb` | Запуск SOTA-моделей (Colab/Kaggle) |
| `03_autointent_fewshot.ipynb` | AutoIntent few-shot (10/20/50-shot, 3 seeds) |

## Ограничения

- **Eval только adversarial:** vanilla_harmful и vanilla_benign
  отсутствуют в eval-сплите, что ограничивает тестирование HYP-JB-002.
- **Дисбаланс классов:** eval содержит ~10:1 jailbreak/safe,
  что влияет на интерпретацию precision.

## Ссылки

- [WildJailbreak](https://arxiv.org/abs/2406.18510) — Jiang et al., NeurIPS 2024
- [WildGuard](https://arxiv.org/abs/2406.18495) — Han et al., NeurIPS 2024
- [XSTest](https://arxiv.org/abs/2308.01263) — Röttger et al., NAACL 2024
- [AutoIntent](https://arxiv.org/abs/2509.21138) — Golubev et al., EMNLP 2025
- [LlamaFirewall / PromptGuard 2](https://arxiv.org/abs/2505.03574) — Meta, 2025
- [Qwen3Guard](https://arxiv.org/abs/2510.14276) — Alibaba, 2025

## Результаты AutoIntent few-shot

**Модель:** AutoIntent classic-light
**Embedder:** intfloat/multilingual-e5-large-instruct

| n_shots | F1 | Recall | CV(F1) |
|---------|----|--------|--------|
| 10-shot | 0.757 ± 0.026 | 0.642 ± 0.039 | 3.4% |
| 20-shot | 0.796 ± 0.062 | 0.701 ± 0.108 | 7.8% |
| 50-shot | 0.731 ± 0.148 | 0.623 ± 0.196 | 20.2% |

Вердикт: **UNSTABLE** (overall CV = 10.3%). Оптимальная точка —
20-shot. Высокий Over-refusal Rate (до 74%) — открытая проблема.
