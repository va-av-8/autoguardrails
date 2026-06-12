# Обёртки AutoML-фреймворков

## 1. Общая архитектура

### Вариант А: эмбеддинги + табличный AutoML

Все три обёртки (AutoGluon, H2O, LAMA) реализуют **вариант А**:
1. Текст преобразуется в эмбеддинги через SentenceTransformer
2. Фреймворку подаётся **числовая матрица** [n_samples, 1024], не сырой текст

**Эмбеддер:** `intfloat/multilingual-e5-large-instruct`

| Обёртка | Файл:строка |
|---------|-------------|
| AutoGluon | `autogluon_wrapper.py:26` |
| H2O | `h2o_wrapper.py:51` |
| LAMA | `lama_wrapper.py:31` |

**Инструкция-префикс e5 НЕ подаётся** — вызов `model.encode()` передаёт тексты напрямую, без "query:" / "passage:":

| Обёртка | Файл:строки |
|---------|-------------|
| AutoGluon | `autogluon_wrapper.py:61-66` |
| H2O | `h2o_wrapper.py:99-104` |
| LAMA | `lama_wrapper.py:66-71` |

**Вариант Б** (фреймворк сам обрабатывает сырой текст) — не реализован.

### Кэш эмбеддингов

Устраняет двойное вычисление при повторных вызовах predict на тех же текстах:

| Обёртка | Объявление кэша | Проверка/возврат | Сохранение |
|---------|-----------------|------------------|------------|
| AutoGluon | `autogluon_wrapper.py:48` | `:57-58` | `:67` |
| H2O | `h2o_wrapper.py:75` | `:95-96` | `:105` |
| LAMA | `lama_wrapper.py:53` | `:62-63` | `:72` |

### Сид: фиксирован в конструкторе, НЕ пробрасывается снаружи

Каждая обёртка имеет `seed: int = 42` в конструкторе:

| Обёртка | Файл:строка |
|---------|-------------|
| AutoGluon | `autogluon_wrapper.py:29` |
| H2O | `h2o_wrapper.py:54` |
| LAMA | `lama_wrapper.py:34` |

**Цепочка:** `args.seed` (`run_framework_benchmarks.py`) влияет только на few-shot выборку (`experiment_runner.py:40`), не на внутреннюю оптимизацию фреймворка.

---

## 2. AutoGluon

### Модели (по leaderboard)

Из `results/scores/autogluon_threshold_10shot_42_deeppavlov_models.json`:

| model | stack_level |
|-------|-------------|
| WeightedEnsemble_L2 | 2 |
| RandomForestEntr | 1 |
| ExtraTreesGini | 1 |
| ExtraTreesEntr | 1 |
| RandomForestGini | 1 |
| CatBoost | 1 |
| XGBoost | 1 |

### Ансамблирование

- **WeightedEnsemble_L2** — взвешенная комбинация базовых моделей L1 (`autogluon_threshold_10shot_42_deeppavlov_models.json:8,27`)
- **Bagging** — ВЫКЛЮЧЕН (`num_bag_folds` не передаётся → дефолт 0). Признак: нет моделей `*_BAG` в leaderboard
- **Stacking** — ВЫКЛЮЧЕН (`num_stack_levels` не передаётся → дефолт 0). Признак: нет L2/L3 кроме WeightedEnsemble

### Исключённые модели

`autogluon_wrapper.py:98`:
```python
excluded_model_types=["FASTAI"]
```
Причина: FastAI несовместим с torch 2.x/MPS на Apple Silicon.

### Метрика выбора модели

**Дефолт: `accuracy`** — не определено явно в коде (`autogluon_wrapper.py:90-94`), используется дефолт AutoGluon для multiclass.

Конструктор `TabularPredictor`:
```python
self._predictor = TabularPredictor(
    label="label",
    problem_type="multiclass",
    learner_kwargs={"random_state": self.seed},
)
```
`eval_metric` не передаётся.

### Бюджет

| Параметр | Значение | Файл:строка |
|----------|----------|-------------|
| `time_limit` | 600 | `autogluon_wrapper.py:27` |
| `num_cpus` | 1 | `autogluon_wrapper.py:28` |

Передаются в `fit()`: `autogluon_wrapper.py:97,99`

**Оговорка:** AutoGluon заполняет весь `time_limit` (лимит = план, не потолок).

### Сид внутри фреймворка

`autogluon_wrapper.py:93`:
```python
learner_kwargs={"random_state": self.seed}
```

---

## 3. H2O

### Модели

Сохранённые leaderboard (`*h2o*_models.json`) — **не найдены** в `results/scores/`. По коду: H2O AutoML обучает до `max_models` моделей.

### Ансамблирование

H2O AutoML по умолчанию строит **Stacked Ensemble** из лучших моделей. Не определено явно в коде — используется дефолтное поведение H2OAutoML.

### Метрика выбора модели

**Дефолт: `mean_per_class_error`** — не определено явно в коде (`h2o_wrapper.py:150-153`), используется дефолт H2O для multinomial.

Конструктор `H2OAutoML`:
```python
aml = H2OAutoML(
    max_models=self.max_models,
    seed=self.seed,
)
```
`sort_metric` не передаётся.

### Бюджет

| Параметр | Значение | Файл:строка | Примечание |
|----------|----------|-------------|------------|
| `max_models` | 5 | `h2o_wrapper.py:52` | Единственный лимит |
| `max_runtime_secs` | None | `h2o_wrapper.py:53` | НЕ используется |
| `nthreads` | 2 (env) | `h2o_wrapper.py:119` | Из `H2O_NTHREADS` |

Передаётся в `H2OAutoML`: `h2o_wrapper.py:151` (только `max_models` и `seed`)

**Оговорка:** H2O ограничен числом моделей (`max_models=5`), НЕ временем — рекомендованный H2O способ воспроизводимости.


### Сид внутри фреймворка

`h2o_wrapper.py:152`:
```python
seed=self.seed
```

---

## 4. LAMA (LightAutoML)

### Модели (по model_info)

Из `results/scores/lama_threshold_10shot_42_standard_models.json`:
```json
{
  "framework": "lama",
  "reader_class": "PandasToPandasReader",
  "blender_class": "WeightedBlender",
  "blender_weights": ["1.0"],
  "pipes": ["NestedTabularMLPipeline"]
}
```

### Ансамблирование

LAMA использует **blending** (WeightedBlender). Не определено явно в коде — дефолтное поведение TabularAutoML.

### Метрика выбора модели

**Дефолт: `crossentropy`** — не определено явно в коде (`lama_wrapper.py:98-105`), используется дефолт LAMA для multiclass.

Создание Task и AutoML:
```python
task = Task("multiclass")
automl = TabularAutoML(
    task=task,
    timeout=self.timeout,
    cpu_limit=self.cpu_limit,
    reader_params={"random_state": self.seed},
    timing_params={"mode": 2},
)
```
`metric` не передаётся в `Task()`.

### Бюджет

| Параметр | Значение | Файл:строка |
|----------|----------|-------------|
| `timeout` | 600 | `lama_wrapper.py:32` |
| `cpu_limit` | 1 | `lama_wrapper.py:33` |
| `timing_params` | `{"mode": 2}` | `lama_wrapper.py:104` |

Передаются в `TabularAutoML()`: `lama_wrapper.py:101-104`

**Оговорка:** `timing_params mode=2` — hard timeout (Benchmarking mode). Даже в hard-режиме таймер не прерывает обучение *внутри* алгоритма → фактическое время может превышать номинальный лимит. Репортится фактический `train_sec` (`experiment_runner.py:91`).

### Сид внутри фреймворка

`lama_wrapper.py:103`:
```python
reader_params={"random_state": self.seed}
```

---

## 5. Способы OOS-скора (prediction_mode)

### threshold

1. **OOS исключается из train** — `base.py:39-40`:
   ```python
   x_texts = [t for t, y in zip(train_texts, train_labels) if y != self.oos_label]
   y_labels = [y for y in train_labels if y != self.oos_label]
   ```
   Результат: 150 классов (in-scope only)

2. **OOS-скор = 1 - max(proba)** — в каждой обёртке:
   | Обёртка | Файл:строка |
   |---------|-------------|
   | AutoGluon | `autogluon_wrapper.py:135` |
   | H2O | `h2o_wrapper.py:204` |
   | LAMA | `lama_wrapper.py:130` |

3. **Порог калибруется на VALIDATION** (не test), максимизация F1 OOS — `base.py:72-109`

### argmax

1. **OOS как 151-й класс в train** — `base.py:37-38`:
   ```python
   if self.prediction_mode == "argmax":
       return list(train_texts), list(train_labels)
   ```
   Результат: 151 класс (включая OOS = -1)

2. **OOS-скор = вероятность класса OOS напрямую**:
   | Обёртка | Файл:строки |
   |---------|-------------|
   | AutoGluon | `autogluon_wrapper.py:130-134` |
   | H2O | `h2o_wrapper.py:198-199` |
   | LAMA | `lama_wrapper.py:127-129` |

### Важно

**Это РАЗНЫЕ обученные модели** (разный train set) — режим нельзя переключить без перепрогона.

### Сохранение артефактов

| Артефакт | Путь | Файл:строки |
|----------|------|-------------|
| Скоры | `results/scores/*_scores.npz` | `evaluation.py:140-148` |
| Leaderboard | `results/scores/*_models.json` | `evaluation.py:151-187` |
| Метрики | `results/metrics.json` | `evaluation.py:263-296` |

Поля в `metrics.json.extra`: `source`, `prediction_mode`, `train_sec`, `calibrate_sec`, `timestamp` (`experiment_runner.py:139-145`, `evaluation.py:260`).

---

## 6. Сопоставимость и ограничения

### Что уравнено

| Аспект | Реализация |
|--------|------------|
| Эмбеддинги | Одинаковые e5-large-instruct для всех |
| Few-shot выборки | Одинаковые на каждый seed (через `load_fewshot`) |
| Тестовый сплит | Один и тот же для всех |
| Способ оценки | Единый `Evaluator` с одинаковыми метриками |

### Что РАЗНОЕ by design (дефолты фреймворков, не ошибки)

| Аспект | AutoGluon | H2O | LAMA |
|--------|-----------|-----|------|
| Набор моделей¹ | GBM, CatBoost, XGBoost, RF, ExtraTrees, KNN, Linear, NN_TORCH | DRF, GLM, GBM, DeepLearning, StackedEnsemble² | lgb, lgb_tuned, cb, cb_tuned, linear_l2 |
| Тип ансамбля | WeightedEnsemble | Stacked Ensemble | Blending (WeightedBlender) |
| Фолды CV | 0 (bagging off) | 5 (дефолт) | 5 (дефолт) |
| **Метрика выбора** | accuracy | mean_per_class_error | crossentropy |
| Механизм бюджета | Время (600s) | Число моделей (5) | Soft-время (600s) |

¹ Доступный набор в нашей конфигурации. AutoGluon: FASTAI исключён явно (несовместим с torch 2.x/MPS на Apple Silicon); полный дефолтный набор доки также включает FASTAI. LAMA: дефолт `auto` для multiclass разворачивается в указанные 5 алгоритмов; доступны также rf/rf_tuned (опционально, не в дефолте).

² На Apple Silicon (ARM) XGBoost недоступен (нет нативных бинарников H2O для ARM). На x86/Windows набор H2O включает XGBoost: DRF, GLM, GBM, **XGBoost**, DeepLearning, StackedEnsemble. **H2O-прогоны планируются на x86/Windows** (текущих записей нет).

**Метрику НЕ унифицируем** — у каждого своя дефолтная, это часть поведения "из коробки".

**Единый РАВНЫЙ бюджет недостижим** из-за разных механизмов контроля (время vs число моделей vs soft-время).


### Эмбеддер без инструкции-префикса

e5-instruct рекомендует prefix "query:" / "passage:", но `encode()` вызывается без него. Одинаково у всех → на сравнение между системами не влияет (возможно занижает абсолютный уровень).

---

**Дата:** 2026-06-12
**Проект в процессе работы**
