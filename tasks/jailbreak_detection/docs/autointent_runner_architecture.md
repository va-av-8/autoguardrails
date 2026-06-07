# Архитектура скрипта AutoIntent Runner

> Документация для `scripts/run_autointent.py`

## 1. ОБЩАЯ СТРУКТУРА

```
┌─────────────────────────────────────────────────────────────────┐
│                         main()                                  │
│  CLI-парсер → цикл по seeds → train() + evaluate()             │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────────┐
│        train()          │     │          evaluate()             │
│  1. load_train()        │     │  1. Pipeline.load()             │
│  2. convert_to_ai_train │     │  2. load_test() + eval_binary   │
│  3. AIDataset.from_dict │     │  3. pipeline.predict()          │
│  4. Pipeline.from_preset│     │  4. evaluate_jailbreak()        │
│  5. pipeline.fit()      │     │  5. save_metrics()              │
│  6. pipeline.dump()     │     │  6. save_eval_scores()          │
└─────────────────────────┘     └─────────────────────────────────┘
```

---

## 2. SEARCH-SPACE ПРЕСЕТЫ AutoIntent

```python
SEARCH_SPACE_PRESETS = (
    "classic-heavy",
    "classic-light",
    "classic-medium",
    "nn-heavy",
    "nn-medium",
    "transformers-heavy",
    "transformers-light",
    "transformers-no-hpo",
    "zero-shot-llm",
    "zero-shot-encoders",
)

# Алиасы CLI → внутренний пресет
PRESET_ALIASES = {
    "bert-finetune": "transformers-light",
}
```

### Состав пресетов (autointent 0.2.0)

| Пресет | Scoring | Decision | HPO |
|--------|---------|----------|-----|
| **classic-light** | knn, linear, mlknn | threshold, argmax | Optuna trials |
| **classic-medium** | knn, linear, mlknn, catboost, RF | threshold, argmax | Optuna trials |
| **classic-heavy** | knn, linear, mlknn, catboost, RF, +variants | threshold, argmax | больше trials |
| **nn-medium** | knn, linear, dnnc | threshold, argmax | Optuna trials |
| **transformers-light** | BERT fine-tune | threshold, argmax | минимум HPO |
| **zero-shot-encoders** | encoder + descriptions | threshold | без HPO |
| **zero-shot-llm** | LLM zero-shot | - | без HPO |

---

## 3. РЕЖИМЫ РАБОТЫ

```python
DEFAULT_SEEDS = (42, 123, 456)
```

| Режим | Данные | Cross-Validation |
|-------|--------|------------------|
| `full` | `wildjailbreak_full100k_seed{seed}.json` (100k) | default AutoIntent |
| `fewshot` | `train_shot{N}_seed{seed}.json` | 3-fold CV |

---

## 4. ЭМБЕДДЕР (фиксированный)

```python
def get_embedder_name(pilot: bool) -> str:
    if pilot:
        return "intfloat/multilingual-e5-small"      # быстрый, для отладки
    return "intfloat/multilingual-e5-large-instruct"  # финальный, 1024D
```

**Особенности:**
- Эмбеддер **фиксирован** (не входит в пространство поиска)
- `--pilot` использует маленький e5-small для быстрой валидации
- `--query-prompt` — инструкция для E5-instruct (prefixтекста)

---

## 5. КОНВЕРТАЦИЯ ДАННЫХ

### Train формат (AutoIntent)

```python
def convert_to_autointent_train(train_data: dict) -> list[dict]:
    """
    Входной JSON:
    {
        "intents": [{"id": 0, "name": "safe", "utterances": [...]}],
        "oos_utterances": [...]
    }

    Выход:
    [
        {"utterance": "text", "label": 0},  # safe
        {"utterance": "text", "label": 1},  # jailbreak
        ...
    ]
    """
```

### Binary Intents (для zero-shot пресетов)

```python
JAILBREAK_BINARY_INTENTS = (
    {
        "id": 0,
        "name": "safe",
        "description": "A benign user request that should not be classified..."
    },
    {
        "id": 1,
        "name": "jailbreak",
        "description": "A harmful, adversarial, or jailbreak-style user request..."
    },
)
```

---

## 6. PIPELINE КОНФИГУРАЦИЯ

```python
# Создание пайплайна из пресета
pipeline = Pipeline.from_preset(autointent_preset, seed=args.seed)

# Фиксация эмбеддера
pipeline.set_config(EmbedderConfig(
    model_name=embedder_name,
    query_prompt=args.query_prompt,  # опционально
))

# Cross-validation для few-shot
if mode == "fewshot":
    pipeline.set_config(DataConfig(scheme="cv", n_folds=3))

# Логирование и сохранение
pipeline.set_config(LoggingConfig(
    project_dir=model_dir,
    dump_modules=True,
    clear_ram=False,
))
```

---

## 7. HPO (Optuna) ПРОГРЕСС

```python
def _install_automl_progress_hooks(pipeline, *, enable_optuna_bar: bool):
    """
    Патчит Optuna study.optimize и NodeOptimizer.fit для отображения:
    1. tqdm progress bar по trials в текущем узле
    2. Общий % HPO по всем узлам (scoring + decision)

    Пример вывода:
    [AutoIntent] HPO stage 1/2: 'scoring' (up to 50 Optuna trials per stage)
    [AutoIntent] Overall HPO (estimate): 45.2%  —  trial 23/50 in current stage
    """
```

**Узлы оптимизации:**
1. **Scoring Node** — выбор scorer (knn/linear/catboost/...) + его гиперпараметры
2. **Decision Node** — выбор decision module (threshold/argmax) + порог

---

## 8. ОБУЧЕНИЕ (train)

```python
def train(args, data_dir: Path, model_dir: Path) -> None:
    # 1. Загрузка данных
    train_raw = load_full_train(seed) if mode == "full" else load_fewshot_train(n_shots, seed)
    train_ai = convert_to_autointent_train(train_raw)

    # 2. Создание Dataset
    ai_dataset = AIDataset.from_dict({
        "train": train_ai,
        "test": test_ai,
        "intents": intents,  # binary: safe=0, jailbreak=1
    })

    # 3. Создание Pipeline
    pipeline = Pipeline.from_preset(autointent_preset, seed=args.seed)
    pipeline.set_config(EmbedderConfig(model_name=embedder_name))

    # 4. HPO hooks (опционально)
    if show_progress:
        restore_hooks = _install_automl_progress_hooks(pipeline, enable_optuna_bar=True)

    # 5. Обучение (AutoML + Optuna)
    pipeline.fit(ai_dataset)

    # 6. Сохранение модели
    pipeline.dump(model_dir)
```

---

## 9. ОЦЕНКА (evaluate)

```python
def evaluate(args, data_dir, model_dir, results_dir, runs_dir) -> None:
    # 1. Загрузка модели
    pipeline = Pipeline.load(model_dir)

    # 2. Загрузка данных
    test_raw = load_test(data_dir)
    eval_binary = load_eval_binary(data_dir)
    y_true = np.array([1 if r["binary_label"] == "jailbreak" else 0 for r in eval_binary])
    data_types = np.array([r["data_type"] for r in eval_binary])

    # 3. Предсказания
    raw_preds = pipeline.predict(test_texts)
    y_pred = np.array([1 if p is None else p for p in raw_preds])  # None → jailbreak

    # 4. Метрики
    metrics = evaluate_jailbreak(y_true, y_pred, data_types, oos_label=1)

    # 5. Дополнительные данные
    scores_eval_summary, scores_array = scoring_eval_summary_from_pipeline(pipeline, test_texts)
    eval_counts = confusion_and_rates_jailbreak_positive(y_true, y_pred)

    # 6. Сохранение
    save_metrics(result, results_dir)
    save_run_metrics_file(result, runs_dir)
    save_eval_scores(eval_scores_path, metadata, test_texts, y_true, y_pred, scores_array)
```

---

## 10. СХЕМА ДАННЫХ

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAIN                                    │
├─────────────────────────────────────────────────────────────────┤
│  full mode:  wildjailbreak_full100k_seed{42,123,456}.json       │
│              → 50k safe (intents[0].utterances)                 │
│              → 50k jailbreak (oos_utterances)                   │
│                                                                  │
│  fewshot:    train_shot{10,20,50}_seed{42,123,456}.json         │
│              → N safe + N jailbreak examples                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          EVAL                                    │
├─────────────────────────────────────────────────────────────────┤
│  test.json:                                                      │
│    {"utterances": [...], "labels": [...]}                       │
│                                                                  │
│  wildjailbreak_eval_binary.jsonl:                               │
│    {"prompt": ..., "binary_label": "jailbreak"|"safe",          │
│     "data_type": "adversarial_harmful"|"vanilla_harmful"|...}   │
│                                                                  │
│  → 2210 samples                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. МЕТРИКИ И ЛОГИРОВАНИЕ

### Основные метрики (evaluate_jailbreak)

```python
{
    "f1": ...,
    "precision": ...,
    "recall": ...,
    "over_refusal_rate": ...,      # FPR на safe (false jailbreak)
    "recall_vanilla_harmful": ...,
    "recall_adversarial_harmful": ...,
}
```

### Confusion Matrix

```python
def confusion_and_rates_jailbreak_positive(y_true, y_pred, positive_label=1):
    return {
        "tp": ..., "fp": ..., "fn": ..., "tn": ...,
        "fnr_jailbreak": ...,  # пропущенные jailbreak
        "fpr_safe": ...,       # ложные jailbreak (over-refusal)
        "n_eval": ...,
        "n_safe_true": ...,
        "n_jailbreak_true": ...,
    }
```

### Scores Summary

```python
def scoring_eval_summary_from_pipeline(pipeline, texts):
    """
    Статистика по скорам scoring модуля:
    - margin = score[jailbreak] - score[safe]
    - margin_mean, margin_std, margin_min, margin_max
    """
```

---

## 12. ВЫХОДНЫЕ ФАЙЛЫ

```
tasks/jailbreak_detection/
├── runs/
│   ├── autointent_{preset}_{e5large|pilot}_{mode}_seed{S}/
│   │   ├── train_metadata.json           # метаданные обучения
│   │   ├── scoring_module/               # scorer артефакты
│   │   │   ├── simple_attrs.json         # k, weights, ...
│   │   │   └── pydantic/embedder_config/ # HF model_name
│   │   └── decision_module/
│   │       └── simple_attrs.json         # threshold, ...
│   │
│   ├── metrics_{model_name}_{mode}_seed{S}.json   # per-run метрики
│   └── eval_scores_{model_name}_{mode}_seed{S}.jsonl  # полные скоры
│
└── results/
    └── metrics.json                       # все результаты (append)
```

---

## 13. CLI ИНТЕРФЕЙС

```bash
# Pilot (быстрая валидация с маленьким эмбеддером)
python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42 --pilot

# Final (e5-large-instruct)
python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42

# Другой пресет
python scripts/run_autointent.py --preset classic-medium --mode full --seed 42

# Все сиды
python scripts/run_autointent.py --preset classic-light --mode full --all-seeds

# Только оценка
python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42 --eval-only

# С query prompt для E5-instruct
python scripts/run_autointent.py --query-prompt "Classify if this is a jailbreak attempt:"
```

**Флаги:**

| Флаг | Описание |
|------|----------|
| `--mode` | fewshot \| full |
| `--preset` | classic-light \| classic-medium \| nn-medium \| ... |
| `--n_shots` | 10 \| 20 \| 50 (для fewshot) |
| `--seed` | 42 \| 123 \| 456 |
| `--pilot` | использовать e5-small вместо e5-large |
| `--all-seeds` | цикл по всем 3 сидам |
| `--train-only` | только обучение |
| `--eval-only` | только оценка (модель должна существовать) |
| `--no-automl-progress` | отключить tqdm и progress |
| `--print-metrics-json` | вывести JSON метрик в stdout |
| `--print-hypothesis-log` | вывести краткий JSON для отчёта |
| `--query-prompt` | инструкция для E5-instruct эмбеддера |

---

## 14. ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `OMP_NUM_THREADS` | Потоки OpenMP | 1 |
| `OPENBLAS_NUM_THREADS` | Потоки OpenBLAS | 1 |
| `MKL_NUM_THREADS` | Потоки MKL | 1 |
| `TOKENIZERS_PARALLELISM` | HF tokenizers | false |
| `JAILBREAK_QUIET_LOGS` | Тихий режим | 0 |
| `TRANSFORMERS_VERBOSITY` | HF transformers | error (Kaggle) |
| `CUDA_VISIBLE_DEVICES` | GPU для PyTorch | auto |

---

## 15. СВОДНАЯ ТАБЛИЦА: AutoIntent vs AutoML

| Аспект | AutoIntent | AutoML (H2O/AG/LAMA) |
|--------|------------|----------------------|
| **Текст→признаки** | E5 embeddings (1024D) | TF-IDF+SVD (128D) / AG text |
| **Пространство поиска** | Scorers: knn, linear, catboost, RF, DNNC | GBM, XGB, LR, RF, ... |
| **HPO** | Optuna (per-node trials) | Framework-specific |
| **Подбор порога** | ✅ Встроен (Decision Node HPO) | См. таблицу ниже |
| **Стекинг** | Нет (single best scorer) | Да (AG, LAMA) |
| **Few-shot** | CV 3-fold | Поддерживается (train_shot{N}) |
| **Оптимизируемая метрика** | ROC-AUC (internal) | F1 |

### Подбор порога: AutoML фреймворки vs наша реализация

| Фреймворк | Встроенный подбор порога | В нашей реализации |
|-----------|--------------------------|-------------------|
| **H2O** | ✅ `find_threshold_by_max_metric()` | Не используем (наше упрощение) → 0.5 |
| **AutoGluon** | ✅ `calibrate_decision_threshold()` | Не используем (наше упрощение) → 0.5 |
| **LightAutoML** | ❌ Нет | Явный порог 0.5 (сознательное упрощение) |
| **AutoIntent** | ✅ Decision Node + Optuna | Используется (встроено) |

---

## 16. АРХИТЕКТУРА AutoIntent Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     AutoIntent Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Embedder   │───►│   Scorer    │───►│  Decision   │          │
│  │  (E5-large) │    │ (knn/linear)│    │ (threshold) │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│        │                   │                   │                 │
│        ▼                   ▼                   ▼                 │
│   text → 1024D        scores [0,1]       label: 0|1             │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  HPO (Optuna):                                                   │
│  - Scoring Node: выбор scorer + гиперпараметры                  │
│  - Decision Node: выбор decision + threshold                    │
└─────────────────────────────────────────────────────────────────┘
```

### Scoring модуль (примеры)

| Scorer | Гиперпараметры |
|--------|----------------|
| **knn** | k, weights (uniform/distance), metric |
| **linear** | C, class_weight |
| **mlknn** | k, s (smoothing) |
| **catboost** | iterations, depth, learning_rate |
| **RandomForest** | n_estimators, max_depth |

### Decision модуль

| Decision | Описание |
|----------|----------|
| **threshold** | if score[1] > thresh → jailbreak |
| **argmax** | argmax(scores) |
| **tunable_threshold** | threshold + HPO для оптимального порога |
