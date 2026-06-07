# Архитектура скрипта AutoML обёрток

> Документация для `scripts/run_automl_baselines.py`

## 1. ОБЩАЯ СТРУКТУРА

```
┌─────────────────────────────────────────────────────────────────┐
│                         main()                                  │
│  CLI-парсер → цикл по seeds × shots → run_one()                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        run_one()                                │
│  1. load_train() → train_df                                     │
│  2. load_eval()  → test_df, y_true, data_types                  │
│  3. train_{framework}() → artifact + feat_pipe                  │
│  4. predict_{framework}() → y_pred, y_proba                     │
│  5. evaluate_jailbreak() → metrics                              │
│  6. save_metrics()                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. ПОДДЕРЖИВАЕМЫЕ ФРЕЙМВОРКИ

| Framework | Текстовое представление | Пространство поиска |
|-----------|------------------------|---------------------|
| **H2O AutoML** | TF-IDF + SVD (sklearn) | GBM, XGBoost, GLM, DRF, DeepLearning |
| **AutoGluon** | Встроенный text pipeline | GBM, XGB, CAT, LR, исключены NN/KNN/RF(full) |
| **LightAutoML** | TF-IDF + SVD (sklearn) | GBM + LR + стекинг (внутренний) |

---

## 3. РЕЖИМЫ РАБОТЫ

```python
DEFAULT_SEEDS = (42, 123, 456)
DEFAULT_N_SHOTS = (10, 20, 50)
```

| Режим | Данные | Time limit |
|-------|--------|------------|
| `full` | 100k train (`wildjailbreak_full100k_seed{seed}.json`) | 3600 сек (H2O: 2400) |
| `fewshot` | N-shot train (`train_shot{N}_seed{seed}.json`) | 900 сек |

---

## 4. ПРЕПРОЦЕССИНГ ТЕКСТА (для H2O и LightAutoML)

```python
def _tfidf_svd_pipeline(mode: str, seed: int) -> SkPipeline:
    max_feat, n_comp = _tfidf_dims(mode)  # (4000, 128)
    return SkPipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_feat,       # 4000 n-грамм
            ngram_range=(1, 2),          # униграммы + биграммы
            sublinear_tf=True,           # 1 + log(tf)
            min_df=2,                    # игнор редких
        )),
        ("svd", TruncatedSVD(n_components=n_comp, random_state=seed)),  # 128D
    ])
```

**Результат:** `text` → TF-IDF(4000) → SVD(128) → `f0...f127` (табличные признаки)

AutoGluon использует **встроенный text pipeline** (без внешнего TF-IDF).

---

## 5. H2O AutoML — ДЕТАЛИ

```python
def train_h2o(...):
    aml_kwargs = {
        "max_runtime_secs": time_limit_sec,      # общий лимит
        "max_models": 16,                         # макс. моделей в ансамбле
        "max_runtime_secs_per_model": 600,        # лимит на 1 модель
        "seed": seed,
        "sort_metric": "F1",                      # оптимизируем F1
        "balance_classes": True,                  # балансировка классов
    }

    # Алгоритмы:
    if ngpu > 0:
        include_algos = ["GBM", "XGBoost", "GLM", "DRF"]  # GPU режим
    else:
        exclude_algos = ["DeepLearning"]                   # CPU режим
```

**Пространство поиска H2O:**
- **GBM** — Gradient Boosting Machine (H2O native)
- **XGBoost** — внешний XGBoost (GPU-ускорение)
- **GLM** — Generalized Linear Model (логрегрессия)
- **DRF** — Distributed Random Forest
- **DeepLearning** — MLP (исключается на CPU)

**Оптимизация:** внутренний поиск гиперпараметров H2O + ансамблирование лучших моделей.

---

## 6. AutoGluon — ДЕТАЛИ

```python
def train_autogluon(...):
    predictor = TabularPredictor(
        label="label",
        problem_type="binary",
        eval_metric="f1",                        # оптимизируем F1
        learner_kwargs={"random_state": seed},
    )

    excluded = ["NN_TORCH", "FASTAI", "KNN"]     # исключены нейросети и KNN
    if mode == "full":
        excluded.append("RF")                     # RF исключён для full (медленный)

    predictor.fit(
        train_df,                                 # text колонка обрабатывается автоматически
        time_limit=time_limit_sec,
        presets=None,                             # без preset → дефолтное пространство
        excluded_model_types=excluded,
    )
```

**Пространство поиска AutoGluon:**
- **GBM** (LightGBM)
- **XGB** (XGBoost)
- **CAT** (CatBoost)
- **LR** (Linear Model)
- **XT** (ExtraTrees)
- Встроенный **TextPredictor** для text → embeddings

**Оптимизация:** Bagging + Stacking (multi-layer) + HPO на время.

---

## 7. LightAutoML — ДЕТАЛИ

```python
def train_lightautoml(...):
    roles = {
        "target": "label",
        "numeric": feat_cols,                    # f0...f127 от TF-IDF+SVD
    }

    kwargs = {
        "task": Task("binary"),
        "timeout": time_limit_sec,
        "memory_limit": 12,                      # GB RAM
        "reader_params": {
            "random_state": seed,
            "cv": 3,                              # 3-fold CV
            "n_jobs": 1,
        },
    }

    automl = TabularAutoML(**kwargs)
    automl.fit_predict(train_data=tab_train, roles=roles)
```

**Пространство поиска LightAutoML:**
- **Level 1:** LightGBM + LinearLBFGS (логрегрессия)
- **Level 2:** Blending (стекинг) поверх Level 1
- Автоматический feature selection
- Early stopping на CV

**Оптимизация:** последовательный пайплайн с блендингом, без полного HPO (в отличие от AG).

---

## 8. ИНФЕРЕНС И МЕТРИКИ

```python
# Предсказание
y_pred, y_proba = predict_{framework}(artifact, test_df, pipe)

# Метрики
metrics = evaluate_jailbreak(y_true, y_pred, data_types, oos_label=1)

# Включает:
# - f1, precision, recall
# - over_refusal_rate (FPR на safe)
# - recall_adversarial_harmful
# - recall_vanilla_harmful
```

### Применение порога (по фреймворкам)

| Фреймворк | Как применяется порог | Встроенный подбор порога | Статус в нашей реализации |
|-----------|----------------------|--------------------------|---------------------------|
| **H2O** | `pdf["predict"]` — дефолт 0.5 | ✅ `find_threshold_by_max_metric()` | **Наше упрощение** — не используем |
| **AutoGluon** | `predictor.predict()` — дефолт 0.5 | ✅ `calibrate_decision_threshold()` | **Наше упрощение** — не используем |
| **LightAutoML** | `(proba > 0.5).astype(int)` — явный | ❌ Нет встроенного | **Сознательное упрощение** — фреймворк не поддерживает |

```python
# H2O — используем метки из predict (дефолт фреймворка 0.5)
labels = pdf["predict"].astype(str).map({"0": 0, "1": 1, ...})
proba = pdf["p1"].to_numpy()  # P(jailbreak)
# Фреймворк поддерживает: model.find_threshold_by_max_metric("f1") — НЕ ИСПОЛЬЗУЕМ

# AutoGluon — predict() возвращает метки (дефолт фреймворка 0.5)
y_pred = predictor.predict(test_df).to_numpy().astype(int)
proba = predictor.predict_proba(test_df)[1].to_numpy()  # P(jailbreak)
# Фреймворк поддерживает: predictor.calibrate_decision_threshold(metric='f1') — НЕ ИСПОЛЬЗУЕМ

# LightAutoML — явно применяем порог 0.5
proba = automl.predict(tab_test)  # возвращает P(jailbreak)
y_pred = (proba > 0.5).astype(int)  # ← сознательное упрощение (фреймворк не поддерживает подбор)
```

**Возможное улучшение:** для H2O и AutoGluon — использовать встроенный подбор порога; для LightAutoML — реализовать вручную.

---

## 9. СХЕМА ДАННЫХ

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAIN                                    │
├─────────────────────────────────────────────────────────────────┤
│  full mode:  wildjailbreak_full100k_seed{42,123,456}.json       │
│              → 50k safe (intents[0].utterances)                 │
│              → 50k jailbreak (oos_utterances)                   │
│                                                                  │
│  fewshot:    train_shot{10,20,50}_seed{42,123,456}.json         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          EVAL                                    │
├─────────────────────────────────────────────────────────────────┤
│  test.json + wildjailbreak_eval_binary.jsonl                    │
│  → 2210 samples                                                  │
│  → data_type: adversarial_harmful, vanilla_harmful, benign       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. СВОДНАЯ ТАБЛИЦА РАЗЛИЧИЙ

| Аспект | H2O | AutoGluon | LightAutoML |
|--------|-----|-----------|-------------|
| **Текст→признаки** | TF-IDF+SVD (sklearn) | Встроенный TextPredictor | TF-IDF+SVD (sklearn) |
| **Размерность** | 128D | Авто | 128D |
| **Базовые модели** | GBM, XGB, GLM, DRF | GBM, XGB, CAT, LR, XT | LightGBM, LinearLBFGS |
| **Ансамбль** | Stacked Ensemble | Multi-layer Stacking | 2-level Blending |
| **HPO** | Grid/Random внутри | Bayesian (Optuna-style) | Минимальный |
| **CV** | Внутренний (H2O) | 5-fold (bagging) | 3-fold |
| **GPU** | XGBoost GPU | Опционально | Опционально |
| **Оптимизируемая метрика** | F1 | F1 | LogLoss (binary) |
| **Порог классификации** | 0.5 (наше упрощение) | 0.5 (наше упрощение) | 0.5 (сознательное упрощение) |
| **Встроенный подбор порога** | ✅ `find_threshold_by_max_metric` | ✅ `calibrate_decision_threshold` | ❌ Нет |

> **H2O, AutoGluon:** фреймворки поддерживают подбор порога, но мы не используем (наше упрощение).
> **LightAutoML:** фреймворк не поддерживает подбор порога, мы явно применяем 0.5 (сознательное упрощение).

---

## 11. CLI ИНТЕРФЕЙС

```bash
python run_automl_baselines.py \
    --framework autogluon \
    --mode full \
    --seed 42 \
    --all-seeds \           # запустить для всех 3 сидов
    --time-limit-sec 3600
```

**Флаги:**

| Флаг | Описание |
|------|----------|
| `--framework` | h2o \| autogluon \| lightautoml |
| `--mode` | full \| fewshot |
| `--seed` | 42 \| 123 \| 456 |
| `--all-seeds` | цикл по (42, 123, 456) |
| `--all-shots` | цикл по (10, 20, 50) для fewshot |
| `--time-limit-sec` | лимит времени в секундах |
| `--train-only` | только обучение, без инференса |
| `--eval-only` | только инференс (модель должна существовать) |
| `--metrics-only` | вывод только JSON метрик в stdout |
| `--no-save-files` | не сохранять metrics.json |

---

## 12. ВЫХОДНЫЕ ФАЙЛЫ

```
tasks/jailbreak_detection/
├── runs/
│   ├── {framework}_automl_baseline_full_seed{seed}/
│   │   ├── tfidf_svd.joblib          # препроцессор (H2O, LAMA)
│   │   ├── h2o_leader.txt            # H2O: описание лидера
│   │   └── model files...
│   └── {framework}_automl_baseline_{N}shot_seed{seed}/
│
└── results/
    └── metrics.json                   # все результаты
```

---

## 13. ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `JAILBREAK_NUM_GPUS` | Число GPU | auto-detect |
| `JAILBREAK_METRICS_ONLY` | Тихий режим | 0 |
| `H2O_MAX_MEM` | RAM для H2O JVM | 14G (Kaggle) / 18G |
| `H2O_MAX_MODELS` | Макс. моделей H2O | 16 |
| `JAILBREAK_TFIDF_MAX_FEATURES_FULL` | TF-IDF размер (full) | 4000 |
| `JAILBREAK_TFIDF_SVD_FULL` | SVD размерность (full) | 128 |
| `JAILBREAK_LAMA_CV` | CV folds для LAMA | 3 |
| `JAILBREAK_LAMA_MEMORY_LIMIT` | RAM для LAMA (GB) | 12 |
