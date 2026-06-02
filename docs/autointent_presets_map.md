# AutoIntent Presets Map: Embedder Selection

**Версия:** autointent 0.2.0
**Источник:** PyPI (`https://pypi.org/simple`), установлен через `uv`
**Тип установки:** обычная (не editable)
**Путь к пакету:** `.venv/lib/python3.11/site-packages/autointent/`
**Дата:** 2026-05-30

**Назначение документа:** карта того, какие пресеты AutoIntent перебирают эмбеддер через Optuna, а какие фиксируют его статически.

---

## Краткий вывод

> **Ни один из 10 стандартных пресетов autointent 0.2.0 НЕ содержит `node_type: embedding` в `search_space`.**
>
> Это означает, что **эмбеддер нигде не перебирается** — он всегда фиксирован:
> - `classic-light`, `classic-medium`, `classic-heavy`, `zero-shot-encoders` → `intfloat/multilingual-e5-large-instruct`
> - `nn-medium`, `nn-heavy`, `transformers-*`, `zero-shot-llm` → `sentence-transformers/all-MiniLM-L6-v2` (default в `EmbedderConfig`)

---

## Итоговая таблица-карта

| Preset | Embedding node? | Эмбеддер фиксирован? | Модель эмбеддера | Scoring-модули | Decision-модули |
|--------|-----------------|---------------------|------------------|----------------|-----------------|
| `classic-light` | нет | да | `intfloat/multilingual-e5-large-instruct` | knn, linear, mlknn | threshold, argmax, jinoos, tunable, adaptive |
| `classic-medium` | нет | да | `intfloat/multilingual-e5-large-instruct` | knn, linear, mlknn, catboost, sklearn | threshold, argmax, jinoos, tunable, adaptive |
| `classic-heavy` | нет | да | `intfloat/multilingual-e5-large-instruct` | knn, linear, mlknn, catboost, sklearn | threshold, argmax, jinoos, tunable, adaptive |
| `nn-medium` | нет | да | `sentence-transformers/all-MiniLM-L6-v2` (default) | cnn, rnn | threshold, argmax, jinoos, tunable, adaptive |
| `nn-heavy` | нет | да | `sentence-transformers/all-MiniLM-L6-v2` (default) | cnn, rnn | threshold, argmax, jinoos, tunable, adaptive |
| `transformers-light` | нет | да | `sentence-transformers/all-MiniLM-L6-v2` (default) | bert (deberta-v3-small) | threshold, argmax, jinoos, tunable, adaptive |
| `transformers-heavy` | нет | да | `sentence-transformers/all-MiniLM-L6-v2` (default) | bert (deberta-v3-large) | threshold, argmax, jinoos, tunable, adaptive |
| `transformers-no-hpo` | нет | да | `sentence-transformers/all-MiniLM-L6-v2` (default) | bert (deberta-v3-small) | threshold, argmax, jinoos, tunable, adaptive |
| `zero-shot-encoders` | нет | да | `intfloat/multilingual-e5-large-instruct` | description_bi, description_cross | threshold, argmax, jinoos, tunable, adaptive |
| `zero-shot-llm` | нет | да | `sentence-transformers/all-MiniLM-L6-v2` (default) | description_llm | threshold, argmax, jinoos, tunable, adaptive |

---

## Исходные YAML каждого пресета

Файлы из `.venv/lib/python3.11/site-packages/autointent/_presets/`.

### classic-light.yaml

```yaml
search_space:
  - node_type: scoring
    target_metric: scoring_f1
    search_space:
      - module_name: knn
        k:
          low: 1
          high: 20
        weights: [uniform, distance, closest]
      - module_name: linear
      - module_name: mlknn
        k:
          low: 1
          high: 20
  - node_type: decision
    target_metric: decision_accuracy
    search_space:
      - module_name: threshold
        thresh:
          low: 0.1
          high: 0.9
      - module_name: argmax
      - module_name: jinoos
      - module_name: tunable
      - module_name: adaptive
hpo_config:
  sampler: tpe
  n_trials: 20
  n_startup_trials: 10
embedder_config:
  model_name: intfloat/multilingual-e5-large-instruct
```

### classic-medium.yaml

```yaml
search_space:
  - node_type: scoring
    target_metric: scoring_f1
    search_space:
      - module_name: knn
        k:
          low: 1
          high: 20
        weights: [uniform, distance, closest]
      - module_name: linear
      - module_name: mlknn
        k:
          low: 1
          high: 20
      - module_name: catboost
      - module_name: sklearn
        clf_name: [RandomForestClassifier]
        n_estimators: [150]
        max_depth: [100]
        n_jobs: [8]
  - node_type: decision
    target_metric: decision_accuracy
    search_space:
      - module_name: threshold
        thresh:
          low: 0.1
          high: 0.9
      - module_name: argmax
      - module_name: jinoos
      - module_name: tunable
      - module_name: adaptive
hpo_config:
  sampler: tpe
  n_trials: 20
  n_startup_trials: 10
embedder_config:
  model_name: intfloat/multilingual-e5-large-instruct
```

### classic-heavy.yaml

```yaml
search_space:
  - node_type: scoring
    target_metric: scoring_f1
    search_space:
      - module_name: knn
        k:
          low: 1
          high: 20
        weights: [uniform, distance, closest]
      - module_name: linear
      - module_name: mlknn
        k:
          low: 1
          high: 20
      - module_name: catboost
        depth: [3, 6, 10]
        features_type: ["text", "embedding", "both"]
      - module_name: sklearn
        clf_name: [RandomForestClassifier]
        n_estimators: [200, 300, 500]
        max_depth: [50, 100, 150]
        max_features: [sqrt, log2]
        n_jobs: [8]
  - node_type: decision
    target_metric: decision_accuracy
    search_space:
      - module_name: threshold
        thresh:
          low: 0.1
          high: 0.9
      - module_name: argmax
      - module_name: jinoos
      - module_name: tunable
      - module_name: adaptive
hpo_config:
  sampler: tpe
  n_trials: 55
  n_startup_trials: 20
embedder_config:
  model_name: intfloat/multilingual-e5-large-instruct
```

### nn-medium.yaml

```yaml
search_space:
  - node_type: scoring
    target_metric: scoring_f1
    search_space:
      - module_name: cnn
        dropout:
          low: 0.1
          high: 0.3
        batch_size: [32, 64, 128]
        learning_rate:
          low: 5.0e-4
          high: 1.0e-2
          log: True
        num_train_epochs: [60]
        embed_dim: [64]
        kernel_sizes: [[3, 4, 5]]
        num_filters: [64]
      - module_name: rnn
        dropout:
          low: 0.1
          high: 0.3
        batch_size: [32, 64, 128]
        learning_rate:
          low: 5.0e-4
          high: 1.0e-2
          log: True
        num_train_epochs: [60]
        embed_dim: [64]
        hidden_dim: [128]
  - node_type: decision
    target_metric: decision_accuracy
    search_space:
      - module_name: threshold
        thresh:
          low: 0.1
          high: 0.9
      - module_name: argmax
      - module_name: jinoos
      - module_name: tunable
      - module_name: adaptive
hpo_config:
  sampler: tpe
  n_trials: 55
  n_startup_trials: 20
```

### nn-heavy.yaml

```yaml
search_space:
  - node_type: scoring
    target_metric: scoring_f1
    search_space:
      - module_name: cnn
        dropout:
          low: 0.1
          high: 0.3
        batch_size: [32, 64, 128]
        learning_rate:
          low: 5.0e-4
          high: 1.0e-2
          log: True
        num_train_epochs: [60]
        embed_dim: [64, 96, 128]
        kernel_sizes: [[3, 4, 5]]
        num_filters: [64, 96, 128]
      - module_name: rnn
        dropout:
          low: 0.1
          high: 0.3
        batch_size: [32, 64, 128]
        learning_rate:
          low: 5.0e-4
          high: 1.0e-2
          log: True
        num_train_epochs: [60]
        embed_dim: [64, 96, 128]
        hidden_dim: [128, 256, 512]
  - node_type: decision
    target_metric: decision_accuracy
    search_space:
      - module_name: threshold
        thresh:
          low: 0.1
          high: 0.9
      - module_name: argmax
      - module_name: jinoos
      - module_name: tunable
      - module_name: adaptive
hpo_config:
  sampler: tpe
  n_trials: 55
  n_startup_trials: 20
```

### transformers-light.yaml

```yaml
search_space:
  - node_type: scoring
    target_metric: scoring_f1
    search_space:
      - module_name: bert
        classification_model_config:
          - model_name: microsoft/deberta-v3-small
        num_train_epochs: [30]
        batch_size: [32, 64, 128]
        learning_rate:
          low: 1.0e-5
          high: 1.0e-4
          log: True
  - node_type: decision
    target_metric: decision_accuracy
    search_space:
      - module_name: threshold
        thresh:
          low: 0.1
          high: 0.9
      - module_name: argmax
      - module_name: jinoos
      - module_name: tunable
      - module_name: adaptive
hpo_config:
  sampler: tpe
  n_trials: 40
  n_startup_trials: 20
```

### transformers-heavy.yaml

```yaml
search_space:
  - node_type: scoring
    target_metric: scoring_f1
    search_space:
      - module_name: bert
        classification_model_config:
          - model_name: microsoft/deberta-v3-large
        num_train_epochs: [30]
        batch_size: [32, 64]
        learning_rate:
          low: 1.0e-5
          high: 1.0e-4
          log: True
  - node_type: decision
    target_metric: decision_accuracy
    search_space:
      - module_name: threshold
        thresh:
          low: 0.1
          high: 0.9
      - module_name: argmax
      - module_name: jinoos
      - module_name: tunable
      - module_name: adaptive
hpo_config:
  sampler: tpe
  n_trials: 40
  n_startup_trials: 20
```

### transformers-no-hpo.yaml

```yaml
search_space:
  - node_type: scoring
    target_metric: scoring_f1
    search_space:
      - module_name: bert
        classification_model_config:
          - model_name: microsoft/deberta-v3-small
        num_train_epochs: [30]
        batch_size: [96]
        learning_rate: [7.0e-5]
  - node_type: decision
    target_metric: decision_accuracy
    search_space:
      - module_name: threshold
        thresh:
          low: 0.1
          high: 0.9
      - module_name: argmax
      - module_name: jinoos
      - module_name: tunable
      - module_name: adaptive
hpo_config:
  sampler: tpe
  n_trials: 40
  n_startup_trials: 20
```

### zero-shot-encoders.yaml

```yaml
search_space:
  - node_type: scoring
    target_metric: scoring_f1
    search_space:
      - module_name: description_bi
      - module_name: description_cross
  - node_type: decision
    target_metric: decision_accuracy
    search_space:
      - module_name: threshold
        thresh:
          low: 0.1
          high: 0.9
      - module_name: argmax
      - module_name: jinoos
      - module_name: tunable
      - module_name: adaptive
hpo_config:
  sampler: tpe
  n_trials: 30
  n_startup_trials: 10
cross_encoder_config:
  model_name: BAAI/bge-reranker-v2-m3
  train_head: false
embedder_config:
  model_name: intfloat/multilingual-e5-large-instruct
```

### zero-shot-llm.yaml

```yaml
search_space:
  - node_type: scoring
    target_metric: scoring_f1
    search_space:
      - module_name: description_llm
  - node_type: decision
    target_metric: decision_accuracy
    search_space:
      - module_name: threshold
        thresh:
          low: 0.1
          high: 0.9
      - module_name: argmax
      - module_name: jinoos
      - module_name: tunable
      - module_name: adaptive
hpo_config:
  sampler: tpe
  n_trials: 30
  n_startup_trials: 10
```

---

## query_prompt / classification_prompt

Во всех 10 пресетах:
- `query_prompt` = `None` (не задан в YAML)
- `classification_prompt` = `None` (не задан в YAML)

Эти поля **статичны** — они **не входят в `search_space`** и не перебираются Optuna.

### Механизм переопределения

```python
pipeline.set_config(EmbedderConfig(
    model_name="intfloat/multilingual-e5-large-instruct",
    query_prompt="Classify if this request is a jailbreak attempt: "
))
```

Вызов `set_config(EmbedderConfig(...))` **полностью перезаписывает** `embedder_config` из пресета. Это происходит **до старта Optuna**, поэтому значения `query_prompt` и других полей **не перебиваются** во время оптимизации — они фиксированы на всё время обучения.

---

## Классы выбора эмбеддера (не используются ни одним пресетом)

В AutoIntent 0.2.0 существуют два модуля для оптимизации эмбеддера, но **ни один стандартный пресет их не использует**:

### RetrievalAimedEmbedding

- **Имя модуля:** `retrieval`
- **Путь:** `autointent/modules/embedding/_retrieval.py`
- **Метрика:** retrieval quality (precision@k, recall@k) — proxy-метрика
- **Параметры:** `embedder_config`, `k` (число соседей)
- **Список моделей:** нет встроенного — модель задаётся в `search_space`

### LogregAimedEmbedding

- **Имя модуля:** `logreg_embedding`
- **Путь:** `autointent/modules/embedding/_logreg.py`
- **Метрика:** F1/Accuracy на LogisticRegression поверх эмбеддингов — downstream-классификация
- **Параметры:** `embedder_config`, `cv` (число фолдов)
- **Список моделей:** нет встроенного — модель задаётся в `search_space`

---

## Как включить выбор эмбеддера

Поскольку ни один стандартный пресет не содержит `node_type: embedding`, выбор эмбеддера возможен только через **ручной search_space**:

```yaml
search_space:
  - node_type: embedding
    target_metric: retrieval_precision
    search_space:
      - module_name: retrieval
        embedder_config:
          - model_name: intfloat/multilingual-e5-large-instruct
          - model_name: sentence-transformers/all-MiniLM-L6-v2
          - model_name: BAAI/bge-m3
        k: [5, 10, 20]
  - node_type: scoring
    target_metric: scoring_f1
    search_space:
      - module_name: knn
        k:
          low: 1
          high: 20
        weights: [uniform, distance, closest]
      - module_name: linear
  - node_type: decision
    target_metric: decision_accuracy
    search_space:
      - module_name: threshold
        thresh:
          low: 0.1
          high: 0.9
      - module_name: argmax
```

При таком конфиге Optuna будет перебирать модели эмбеддера из списка `embedder_config` и параметр `k`.

---

## Расхождение с dev (post-0.3.0)

| Файл | autointent 0.2.0 (установлен) | GitHub dev (post-0.3.0) |
|------|------------------------------|-------------------------|
| `classic-medium.yaml` | содержит `catboost` модуль | **НЕ содержит** `catboost` модуль |

Все остальные пресеты **идентичны** между версиями 0.2.0 и dev.

**Примечание:** На PyPI доступна версия 0.3.0, в которой `catboost` убран из `classic-medium`. Текущий проект использует 0.2.0, где `catboost` ещё присутствует.
