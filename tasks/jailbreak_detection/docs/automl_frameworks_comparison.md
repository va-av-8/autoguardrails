# Сравнение AutoML фреймворков: возможности vs реализация в проекте

> Документация сравнивает возможности фреймворков (AutoGluon, H2O, LightAutoML) с двумя реализациями в проекте:
> - **run_automl_baselines.py** — TF-IDF+SVD (128D) признаки
> - **run_framework_benchmarks.py** — E5 эмбеддинги (1024D) признаки

---

## 1. АРХИТЕКТУРА ДВУХ РЕАЛИЗАЦИЙ

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     run_automl_baselines.py                                 │
│                     (TF-IDF + SVD, 128D)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  text → TfidfVectorizer(4000) → TruncatedSVD(128) → f0...f127              │
│                                                                             │
│  AutoGluon: text колонка → внутренний TextPredictor (для full mode)        │
│  H2O:       TF-IDF+SVD → numeric features                                   │
│  LAMA:      TF-IDF+SVD → numeric features                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                     run_framework_benchmarks.py                             │
│                     (E5 embeddings, 1024D)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  text → SentenceTransformer(E5-large) → f_0...f_1023 (L2 normalized)       │
│                                                                             │
│  AutoGluon: E5 embeddings → numeric features (NO text preprocessing)       │
│  H2O:       E5 embeddings → numeric features (NO Word2Vec)                  │
│  LAMA:      E5 embeddings → numeric features (NO FastText)                  │
│                                                                             │
│  + Кэширование эмбеддингов в .npy файлах                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ПРЕПРОЦЕССИНГ

### Возможности фреймворков

| Фреймворк | Текст | Categorical | Missing | Scaling |
|-----------|-------|-------------|---------|---------|
| **AutoGluon** | ✅ TextPredictor, AG_AUTOMM | ✅ Auto (label, target, onehot) | ✅ Auto | ✅ Для NN |
| **H2O** | ✅ Word2Vec | ✅ Auto (asfactor) | ✅ Auto | ✅ Для GLM/DL |
| **LightAutoML** | ✅ FastText | ✅ Label, Target, Frequency encoding | ✅ Median/Mode | ✅ Standardization |

### Реализация в проекте

| Аспект | run_automl_baselines.py | run_framework_benchmarks.py |
|--------|------------------------|----------------------------|
| **Текст→признаки** | TF-IDF+SVD (128D) sklearn | E5 embeddings (1024D) sentence-transformers |
| **Кэширование** | ❌ Нет | ✅ `.npy` файлы |
| **Текстовый препроцессинг фреймворков** | AutoGluon: используется (full mode), H2O/LAMA: обходится | Все: **обходится** (только numeric) |
| **Categorical** | Нет (только numeric) | Нет (только numeric) |
| **Missing** | Нет (TF-IDF всегда есть) | Нет (E5 всегда есть) |

---

## 3. ПРОСТРАНСТВО ПОИСКА МОДЕЛЕЙ

### AutoGluon

| Модель | Доступно | run_automl_baselines | run_framework_benchmarks |
|--------|----------|---------------------|--------------------------|
| **GBM** (LightGBM) | ✅ | ✅ | ✅ |
| **CAT** (CatBoost) | ✅ | ✅ | ✅ |
| **XGB** (XGBoost) | ✅ | ✅ | ✅ |
| **RF** (Random Forest) | ✅ | ❌ full mode (`excluded`) | ✅ |
| **XT** (ExtraTrees) | ✅ | ✅ | ✅ |
| **KNN** | ✅ | ❌ `excluded` | ✅ |
| **LR** (Linear) | ✅ | ✅ | ✅ |
| **NN_TORCH** | ✅ | ❌ `excluded` | ✅ |
| **FASTAI** | ✅ | ❌ `excluded` | ✅ |
| **AG_AUTOMM** | ✅ | ❌ | ❌ |
| **WeightedEnsemble** | ✅ | ✅ auto | ✅ auto |

**Код run_automl_baselines.py:443-445:**
```python
excluded = ["NN_TORCH", "FASTAI", "KNN"]
if mode == "full":
    excluded.append("RF")
```

**Код run_framework_benchmarks (autogluon_wrapper.py):**
```python
# excluded_model_types не указан → все модели доступны
```

### H2O

| Алгоритм | Доступно | run_automl_baselines | run_framework_benchmarks |
|----------|----------|---------------------|--------------------------|
| **GBM** | ✅ | ✅ | ✅ |
| **XGBoost** | ✅ | ✅ | ✅ |
| **GLM** | ✅ | ✅ | ✅ |
| **DRF** (Random Forest) | ✅ | ✅ | ✅ |
| **XRT** (ExtraTrees) | ✅ | ✅ | ✅ |
| **DeepLearning** | ✅ | ❌ CPU mode (`excluded`) | ✅ |
| **StackedEnsemble** | ✅ | ✅ **ВКЛЮЧЕН** | ❌ **ОТКЛЮЧЕН** |

**Код run_automl_baselines.py:367-371:**
```python
if ngpu > 0:
    aml_kwargs["include_algos"] = ["GBM", "XGBoost", "GLM", "DRF"]
else:
    aml_kwargs["exclude_algos"] = ["DeepLearning"]
# StackedEnsemble: НЕ исключён → ВКЛЮЧЕН
```

**Код run_framework_benchmarks (h2o_wrapper.py:181):**
```python
exclude_algos=["StackedEnsemble"],  # ОТКЛЮЧЕН (AssertionError на few-shot)
```

### LightAutoML

| Алгоритм | Доступно | run_automl_baselines | run_framework_benchmarks |
|----------|----------|---------------------|--------------------------|
| **LightGBM** (`lgb`) | ✅ | ✅ | ✅ |
| **LightGBM tuned** (`lgb_tuned`) | ✅ | ✅ | ✅ |
| **CatBoost** (`cb`) | ✅ | ✅ (default) | ✅ (default) |
| **XGBoost** (`xgb`) | ✅ | ❌ не в default | ❌ не в default |
| **LinearLBFGS** (`linear_l2`) | ✅ | ✅ | ✅ |
| **Random Forest** | ✅ | ❌ не в default | ❌ не в default |
| **Blending (2-level)** | ✅ | ✅ auto | ✅ auto |

**Обе реализации используют default preset:** LightGBM + LinearLBFGS + Blending

---

## 4. СТЕКИНГ / АНСАМБЛИРОВАНИЕ

| Фреймворк | Доступно | run_automl_baselines | run_framework_benchmarks |
|-----------|----------|---------------------|--------------------------|
| **AutoGluon** | WeightedEnsemble + Multi-layer Stacking | ✅ auto | ✅ auto |
| **H2O** | StackedEnsemble (All + Best-of-Family) | ✅ **ВКЛЮЧЕН** | ❌ **ОТКЛЮЧЕН** |
| **LightAutoML** | 2-level Blending | ✅ auto | ✅ auto |

**Причина отключения H2O StackedEnsemble в E5 wrappers:** `AssertionError` на few-shot данных.

---

## 5. HPO (ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ)

### Возможности фреймворков

| Фреймворк | HPO Engine | Метод | Что оптимизируется |
|-----------|------------|-------|-------------------|
| **AutoGluon** | Внутренний | Bayesian-like | Hyperparams + ensemble weights |
| **H2O** | Внутренний | Grid + Random Search | Hyperparams GBM/XGB/DL |
| **LightAutoML** | Optuna (TPE) | TPE + Grid | Hyperparams LightGBM/Linear |

### Реализация в проекте

| Аспект | run_automl_baselines | run_framework_benchmarks |
|--------|---------------------|--------------------------|
| **AutoGluon бюджет** | `time_limit_sec` (3600 full / 900 fewshot) | `time_limit=600` |
| **H2O бюджет** | `max_runtime_secs` + `max_models=16` | `max_models=5` |
| **LAMA бюджет** | `timeout=time_limit_sec` | `timeout=600` |
| **LAMA CV** | `cv=3` | default (5) |

**HPO используется полностью в обеих реализациях** — это внутренняя логика фреймворков, мы её не ограничиваем.

---

## 6. ПОДБОР ПОРОГА

| Фреймворк | Доступно во фреймворке | run_automl_baselines | run_framework_benchmarks |
|-----------|------------------------|---------------------|--------------------------|
| **AutoGluon** | ✅ `calibrate_decision_threshold()` | ❌ Фикс 0.5 (дефолт predict) | ❌ Фикс 0.5 |
| **H2O** | ✅ `find_threshold_by_max_metric()` | ❌ Фикс 0.5 (дефолт predict) | ❌ Фикс 0.5 |
| **LightAutoML** | ❌ Нет встроенного | ❌ Явный `(proba > 0.5)` | ❌ Явный `(proba > 0.5)` |

**В E5 wrappers есть `calibrate_threshold()` в base_binary.py, но он НЕ вызывается.**

---

## 7. МЕТРИКА ОПТИМИЗАЦИИ

| Фреймворк | run_automl_baselines | run_framework_benchmarks |
|-----------|---------------------|--------------------------|
| **AutoGluon** | `eval_metric="f1"` | default (не указан) |
| **H2O** | `sort_metric="F1"`, `balance_classes=True` | `sort_metric="AUC"` |
| **LightAutoML** | default (LogLoss) | default (LogLoss) |

---

## 8. СВОДНАЯ ТАБЛИЦА ОТЛИЧИЙ ДВУХ РЕАЛИЗАЦИЙ

| Аспект | run_automl_baselines (TF-IDF) | run_framework_benchmarks (E5) |
|--------|------------------------------|-------------------------------|
| **Признаки** | TF-IDF+SVD (128D) | E5 embeddings (1024D) |
| **Кэш эмбеддингов** | ❌ | ✅ `.npy` |
| **AG: excluded_model_types** | `["NN_TORCH", "FASTAI", "KNN"]` + RF | Нет (все модели) |
| **AG: eval_metric** | `"f1"` | default |
| **AG: time_limit** | 3600/900 сек | 600 сек |
| **H2O: StackedEnsemble** | ✅ Включен | ❌ Отключен |
| **H2O: sort_metric** | `"F1"` | `"AUC"` |
| **H2O: balance_classes** | ✅ True | ❌ Нет |
| **H2O: max_models** | 16 | 5 |
| **LAMA: cv** | 3 | default (5) |
| **LAMA: timeout** | time_limit_sec | 600 сек |
| **Grid runner** | Вручную (--all-seeds) | ✅ `run_grid()` + skip_existing |

---

## 9. КОД: КЛЮЧЕВЫЕ ФРАГМЕНТЫ

### run_automl_baselines.py

```python
# AutoGluon (строки 431-451)
predictor = TabularPredictor(
    label="label",
    problem_type="binary",
    eval_metric="f1",                    # ← явно F1
    learner_kwargs={"random_state": seed},
)
excluded = ["NN_TORCH", "FASTAI", "KNN"]  # ← исключаем модели
if mode == "full":
    excluded.append("RF")
predictor.fit(
    train_df,                            # ← text колонка (AG TextPredictor)
    time_limit=time_limit_sec,
    excluded_model_types=excluded,
)

# H2O (строки 358-372)
aml_kwargs = {
    "max_runtime_secs": time_limit_sec,
    "max_models": 16,
    "sort_metric": "F1",                 # ← явно F1
    "balance_classes": True,             # ← балансировка
}
if ngpu > 0:
    aml_kwargs["include_algos"] = ["GBM", "XGBoost", "GLM", "DRF"]
else:
    aml_kwargs["exclude_algos"] = ["DeepLearning"]
# StackedEnsemble: НЕ исключён

# LAMA (строки 560-578)
kwargs = {
    "task": Task("binary"),
    "timeout": time_limit_sec,
    "reader_params": {
        "random_state": seed,
        "cv": 3,                         # ← явно 3-fold
    },
}
```

### run_framework_benchmarks (wrappers)

```python
# AutoGluon (autogluon_wrapper.py:166-178)
self._predictor = TabularPredictor(
    label="label",
    problem_type="binary",
    # eval_metric не указан → default
    learner_kwargs={"random_state": self.seed},
)
self._predictor.fit(
    train_data=train_df,                 # ← только f_0...f_1023 (numeric)
    time_limit=self.time_limit,          # ← 600 сек
    # excluded_model_types не указан → все модели
)

# H2O (h2o_wrapper.py:177-187)
aml = H2OAutoML(
    max_models=self.max_models,          # ← 5
    seed=self.seed,
    sort_metric="AUC",                   # ← AUC (не F1)
    exclude_algos=["StackedEnsemble"],   # ← СТЕКИНГ ОТКЛЮЧЕН
    # balance_classes не указан
)

# LAMA (lama_wrapper.py:177-186)
automl = TabularAutoML(
    task=task,
    timeout=self.timeout,                # ← 600 сек
    cpu_limit=self.cpu_limit,            # ← 1 (Mac fix)
    reader_params={"random_state": self.seed},
    # cv не указан → default (5)
)
```

---

## 10. СРАВНЕНИЕ С AutoIntent

| Аспект | AutoIntent | AutoML (TF-IDF) | AutoML (E5 wrappers) |
|--------|------------|-----------------|----------------------|
| **Признаки** | E5 (внутри Pipeline) | TF-IDF+SVD (128D) | E5 (внешний, кэш) |
| **Модели** | knn, linear, mlknn, catboost, RF, DNNC | Зависит от фреймворка | Зависит от фреймворка |
| **HPO** | Optuna (scoring + decision) | Внутренний фреймворков | Внутренний фреймворков |
| **Подбор порога** | ✅ Decision Node HPO | ❌ Фикс 0.5 | ❌ Фикс 0.5 |
| **Стекинг** | ❌ Single best scorer | AG: ✅, H2O: ✅, LAMA: ✅ | AG: ✅, H2O: ❌, LAMA: ✅ |

---

**Sources:**
- [AutoGluon TabularPredictor.fit](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html)
- [H2O AutoML Documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [H2O exclude_algos](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/exclude_algos.html)
- [LightAutoML Documentation](https://lightautoml.readthedocs.io/en/latest/)
- [LightAutoML Paper (arXiv)](https://arxiv.org/pdf/2109.01528)
