# Архитектура скрипта Framework Benchmarks

> Документация для `scripts/run_framework_benchmarks.py` и `src/experiment_runner.py`

## 1. ОБЩАЯ СТРУКТУРА

```
┌─────────────────────────────────────────────────────────────────┐
│              run_framework_benchmarks.py (CLI)                  │
│  parse_args() → load embedder → run_grid()                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              experiment_runner.py (orchestrator)                │
│  run_grid() → цикл по (frameworks × modes × seeds)              │
│                        │                                         │
│                        ▼                                         │
│                  run_single()                                    │
│  1. load_train() + load_test()                                   │
│  2. get_or_compute_embeddings() (кэш!)                          │
│  3. create_wrapper(framework) → fit()                           │
│  4. predict_proba_from_embeddings() → (scores >= threshold)     │
│  5. evaluate_jailbreak() → metrics                              │
│  6. append_to_metrics_json()                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              framework_wrappers/ (модульные обёртки)            │
├─────────────────────────────────────────────────────────────────┤
│  BinaryFrameworkWrapper (ABC)                                    │
│    ├── AutoGluonBinaryWrapper                                    │
│    ├── H2OBinaryWrapper                                          │
│    └── LAMABinaryWrapper                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. КЛЮЧЕВОЕ ОТЛИЧИЕ ОТ run_automl_baselines.py

| Аспект | run_automl_baselines.py | run_framework_benchmarks.py |
|--------|------------------------|----------------------------|
| **Текст→признаки** | TF-IDF+SVD (128D) / AG text | E5 embeddings (1024D) |
| **Эмбеддинг** | Внутри каждого фреймворка | **Общий**, один раз загружается |
| **Кэширование** | Нет | ✅ `embedding_cache.py` (.npy файлы) |
| **Фреймворки** | H2O, AutoGluon, LightAutoML | AutoGluon, H2O, LAMA (binary wrappers) |
| **Порог** | 0.5 (упрощение/сознательное) | 0.5 (default_threshold) + опционально `calibrate_threshold()` |

**Главная идея:** все фреймворки используют **одни и те же E5 эмбеддинги**, что делает сравнение честным.

---

## 3. РЕЖИМЫ РАБОТЫ

```python
# CLI
--frameworks autogluon h2o lama   # по умолчанию все три
--n-shots 10 20 50                # few-shot значения
--seeds 42 123 456                # random seeds
--run-full                        # full-train эксперименты
--run-fewshot                     # few-shot эксперименты
```

| Режим | Данные | Формат mode |
|-------|--------|-------------|
| `full` | `wildjailbreak_full100k_seed{S}.json` | `"full"` |
| `fewshot` | `train_shot{N}_seed{S}.json` | `"{N}shot"` |

---

## 4. КЭШИРОВАНИЕ ЭМБЕДДИНГОВ

```python
# embedding_cache.py
CACHE_DIR = "tasks/jailbreak_detection/data/processed/embeddings_cache/"

def get_or_compute_embeddings(embedder, embedder_hf_model, split_id, texts):
    """
    Cache path: {safe_model_name}_{split_id}.npy

    Примеры:
    - intfloat_multilingual-e5-large-instruct_10shot_seed42.npy
    - intfloat_multilingual-e5-large-instruct_test.npy
    - intfloat_multilingual-e5-large-instruct_full100k_seed42.npy
    """
    path = CACHE_DIR / f"{safe_model_name}_{split_id}.npy"

    if path.exists():
        return np.load(path)  # cache hit

    embeddings = embedder.encode(texts, normalize_embeddings=True)
    np.save(path, embeddings)
    return embeddings
```

**Преимущества:**
- Эмбеддинги вычисляются **один раз** на split
- Ускоряет повторные эксперименты в 10-100x
- Гарантирует идентичные признаки для всех фреймворков

---

## 5. FRAMEWORK WRAPPERS (модульная архитектура)

### Базовый класс

```python
class BinaryFrameworkWrapper(ABC):
    """Binary classification wrapper (не OOS detection!)"""

    positive_label = 1  # jailbreak
    default_threshold = 0.5

    @abstractmethod
    def fit(self, train_texts, train_labels, precomputed_embeddings=None): ...

    @abstractmethod
    def _predict_proba_raw(self, texts) -> np.ndarray: ...

    @abstractmethod
    def _predict_proba_raw_from_embeddings(self, embeddings) -> np.ndarray: ...

    def predict_proba(self, texts) -> np.ndarray:
        """P(jailbreak) с защитой от дегенеративных скоров (std < 1e-6)"""

    def predict(self, texts) -> np.ndarray:
        """(scores >= threshold).astype(int)"""

    def calibrate_threshold(self, val_texts, val_labels, metric="f1") -> float:
        """Опционально: подбор порога на валидации (grid search по F1)"""
```

### Реестр обёрток

```python
# framework_wrappers/__init__.py
WRAPPER_REGISTRY = {
    "autogluon": AutoGluonBinaryWrapper,
    "h2o": H2OBinaryWrapper,
    "lama": LAMABinaryWrapper,
}

def create_wrapper(name: str, **kwargs) -> BinaryFrameworkWrapper:
    return WRAPPER_REGISTRY[name.lower()](**kwargs)
```

---

## 6. AUTOGLUON BINARY WRAPPER (пример)

```python
class AutoGluonBinaryWrapper(BinaryFrameworkWrapper):
    def __init__(
        self,
        default_threshold=0.5,
        embedder_name="intfloat/multilingual-e5-large-instruct",
        embedder=None,           # можно передать загруженный
        time_limit=600,          # секунд на AutoML
        num_cpus=1,
        seed=42,
        verbosity=0,
    ): ...

    def fit(self, train_texts, train_labels, precomputed_embeddings=None):
        # 1. Используем precomputed embeddings или вычисляем
        # 2. DataFrame с признаками f_0, f_1, ..., f_1023
        # 3. TabularPredictor(problem_type="binary", learner_kwargs={"random_state": seed})
        # 4. predictor.fit(time_limit=..., ag_args_fit={"num_cpus": ...})

    def _predict_proba_raw_from_embeddings(self, embeddings):
        proba_df = self._predictor.predict_proba(test_df)
        return proba_df[self.positive_label].to_numpy()  # P(jailbreak)
```

---

## 7. PIPELINE ВЫПОЛНЕНИЯ (run_single)

```python
def run_single(framework, mode, seed, n_shots, embedder, embedder_hf_model, ...):
    # 1. Load data
    train_texts, train_labels = load_train(mode, seed, n_shots)
    test_texts, test_labels = load_test()
    data_types = load_eval_data_types()

    # 2. Get/compute embeddings (CACHED!)
    train_embeddings = get_or_compute_embeddings(embedder, embedder_hf_model, split_id, train_texts)
    test_embeddings = get_or_compute_embeddings(embedder, embedder_hf_model, "test", test_texts)

    # 3. Create wrapper
    wrapper = create_wrapper(framework, embedder=embedder, seed=seed, ...)

    # 4. Fit with precomputed embeddings
    wrapper.fit(train_texts, train_labels, precomputed_embeddings=train_embeddings)

    # 5. Predict
    scores = wrapper.predict_proba_from_embeddings(test_embeddings)
    threshold = wrapper._effective_threshold()  # 0.5 или калиброванный
    preds = (scores >= threshold).astype(int)

    # 6. Evaluate
    metrics = evaluate_jailbreak(test_labels, preds, data_types, oos_label=1)

    # 7. Build record & save
    record = {...}
    append_to_metrics_json(record, results_file)

    return record
```

---

## 8. ФОРМАТ ВЫХОДНЫХ МЕТРИК

```python
record = {
    "model_name": "autogluon",  # framework name
    "mode": "10shot",           # или "full"
    "n_shots": 10,              # или None для full
    "seed": 42,
    "f1": 0.85,
    "precision": 0.82,
    "recall": 0.88,
    "over_refusal_rate": 0.15,
    "recall_adversarial_harmful": 0.75,
    "timestamp": "2024-...",
    "extra": {
        "embedder": "fixed (e5-large-instruct)",
        "embedder_hf_model": "intfloat/multilingual-e5-large-instruct",
        "embedder_fixed": True,
        "threshold_used": 0.5,
        "eval_counts": {"tp": ..., "fp": ..., "fn": ..., "tn": ...},
        "scores_eval_summary": {"margin_mean": ..., "scores_mean": ...},
        "scores": [...],  # полные P(jailbreak) для анализа
        "training_time_sec": 120.5,
    },
}
```

---

## 9. CLI ИНТЕРФЕЙС

```bash
# Pilot: один фреймворк, один seed
python scripts/run_framework_benchmarks.py \
    --frameworks autogluon --n-shots 10 --seeds 42 --run-fewshot

# Full grid: все фреймворки × все режимы × все seeds
python scripts/run_framework_benchmarks.py --run-full --run-fewshot

# H2O и LAMA только на full-train
python scripts/run_framework_benchmarks.py \
    --frameworks h2o lama --run-full

# Продолжить после ошибки + пропустить уже выполненные
python scripts/run_framework_benchmarks.py \
    --run-full --continue-on-error --skip-existing

# Другой эмбеддер
python scripts/run_framework_benchmarks.py \
    --embedder "intfloat/multilingual-e5-small" --run-fewshot
```

**Флаги:**

| Флаг | Описание |
|------|----------|
| `--frameworks` | autogluon h2o lama (по умолчанию все) |
| `--n-shots` | 10 20 50 (few-shot значения) |
| `--seeds` | 42 123 456 (random seeds) |
| `--run-full` | запустить full-train эксперименты |
| `--run-fewshot` | запустить few-shot эксперименты |
| `--results-file` | путь к metrics.json |
| `--continue-on-error` | продолжить grid при ошибке |
| `--skip-existing` | пропустить уже выполненные эксперименты |
| `--embedder` | HuggingFace модель эмбеддера |

---

## 10. СРАВНЕНИЕ С ДРУГИМИ СКРИПТАМИ

| Аспект | run_automl_baselines | run_autointent | run_framework_benchmarks |
|--------|---------------------|----------------|--------------------------|
| **Эмбеддинг** | TF-IDF+SVD (128D) | E5 (внутри AutoIntent) | E5 (общий, кэшированный) |
| **Кэш эмбеддингов** | ❌ | ❌ | ✅ `.npy` файлы |
| **Фреймворки** | H2O, AG, LAMA (raw) | AutoIntent preset | AG, H2O, LAMA (wrappers) |
| **HPO** | Внутри фреймворков | Optuna (scoring+decision) | Внутри фреймворков |
| **Подбор порога** | 0.5 (упрощение) | ✅ Decision Node | `calibrate_threshold()` опционально |
| **Grid runner** | Вручную (--all-seeds) | --all-seeds | ✅ `run_grid()` |
| **Skip existing** | ❌ | ❌ | ✅ --skip-existing |

---

## 11. ПОДБОР ПОРОГА (calibrate_threshold)

```python
# base_binary.py
def calibrate_threshold(self, val_texts, val_labels, n_thresholds=50, metric="f1"):
    """
    Grid search по порогу для максимизации F1.

    НЕ вызывается автоматически — это ОПЦИЯ.
    По умолчанию используется default_threshold=0.5
    """
    scores = self.predict_proba(val_texts)
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

    best_threshold = thresholds[0]
    best_score = -1.0

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        score = f1_oos(y_true, y_pred, oos_label=self.positive_label)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    self.threshold_ = best_threshold
    return best_threshold
```

**Статус:** Реализовано в `BinaryFrameworkWrapper`, но **не используется** в `run_single()`. Порог фиксирован на 0.5.

---

## 12. СТРУКТУРА ФАЙЛОВ

```
tasks/jailbreak_detection/
├── scripts/
│   └── run_framework_benchmarks.py   # CLI entry point
│
├── src/
│   ├── experiment_runner.py          # run_single(), run_grid()
│   ├── embedding_cache.py            # get_or_compute_embeddings()
│   ├── data_utils.py                 # load_train(), load_test()
│   ├── metrics.py                    # evaluate_jailbreak()
│   └── framework_wrappers/
│       ├── __init__.py               # WRAPPER_REGISTRY, create_wrapper()
│       ├── base_binary.py            # BinaryFrameworkWrapper (ABC)
│       ├── autogluon_wrapper.py      # AutoGluonBinaryWrapper
│       ├── h2o_wrapper.py            # H2OBinaryWrapper
│       └── lama_wrapper.py           # LAMABinaryWrapper
│
├── data/processed/
│   └── embeddings_cache/             # .npy файлы кэша
│
└── results/
    └── metrics.json                   # все результаты
```
