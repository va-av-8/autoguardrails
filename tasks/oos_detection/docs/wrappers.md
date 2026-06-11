# Обёртки над AutoML-фреймворками: фактическая реализация

## 1. Общая архитектура

### Вариант B: эмбеддинги + табличный AutoML

Все три обёртки (AutoGluon, H2O, LAMA) реализуют **вариант B**:
1. Текст преобразуется в эмбеддинги через SentenceTransformer
2. Фреймворку подаётся **числовая матрица** [n_samples, 1024], не сырой текст

**Эмбеддер:** `intfloat/multilingual-e5-large-instruct`
| Обёртка | Файл:строка |
|---------|-------------|
| AutoGluon | `autogluon_wrapper.py:26` |
| H2O | `h2o_wrapper.py:51` |
| LAMA | `lama_wrapper.py:31` |

**Инструкция-префикс e5 НЕ подаётся** — вызов `model.encode()` передаёт только тексты:
| Обёртка | Файл:строки |
|---------|-------------|
| AutoGluon | `autogluon_wrapper.py:61-66` — аргументы: `texts`, `normalize_embeddings=True`, `show_progress_bar=False` |
| H2O | `h2o_wrapper.py:99-104` |
| LAMA | `lama_wrapper.py:66-71` |

**Вариант A** (фреймворк сам обрабатывает сырой текст) — не реализован.

### Кэш эмбеддингов

Устраняет двойное вычисление эмбеддингов при повторных вызовах predict на тех же текстах:

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

**Цепочка подтверждения:**
- `experiment_runner.py:80` — `wrapper_kwargs` формируется как `{"prediction_mode": ..., **(wrapper_kwargs or {})}`
- `experiment_runner.py:251` — `wrapper_kwargs` передаётся в `run_single_experiment`
- `args.seed` (`run_framework_benchmarks.py:84`) влияет только на выбор few-shot выборки (`experiment_runner.py:40`), не на внутреннюю оптимизацию фреймворка

---

## 2. AutoGluon

### Обученные модели (по факту из leaderboard)

Из `results/scores/autogluon_threshold_10shot_42_deeppavlov_models.json`:
| model | stack_level | строка |
|-------|-------------|--------|
| WeightedEnsemble_L2 | 2 | :8 |
| RandomForestEntr | 1 | :541 |
| ExtraTreesGini | 1 | :1630 |
| ExtraTreesEntr | 1 | :2719 |
| RandomForestGini | 1 | :3806 |
| CatBoost | 1 | :4893 |
| XGBoost | 1 | :5981 |

Из `results/scores/autogluon_argmax_10shot_42_deeppavlov_models.json`:
| model | stack_level | строка |
|-------|-------------|--------|
| LightGBMXT | 1 | :8 |
| WeightedEnsemble_L2 | 2 | :1094 |
| LightGBM | 1 | :1326 |

### Ансамблирование

- **WeightedEnsemble_L2** — взвешенная комбинация базовых моделей уровня L1 (присутствует в обоих leaderboard)
- **Bagging** — ВЫКЛЮЧЕН (`num_bag_folds` не передаётся → дефолт 0). Признак: нет моделей с суффиксом `_BAG`
- **Stacking** — ВЫКЛЮЧЕН (`num_stack_levels` не передаётся → дефолт 0). Признак: нет моделей L2/L3 кроме WeightedEnsemble

### Исключённые модели

`autogluon_wrapper.py:98`:
```python
excluded_model_types=["FASTAI"]
```
Причина: FastAI опционален и несовместим с torch ≥2.10/MPS (требует torch<2.10).

### Число фолдов CV

Bagging выключен (`num_bag_folds=0` по умолчанию) — не определено в коде явно, используется дефолт AutoGluon.

### Time-лимит

`autogluon_wrapper.py:27`: `time_limit: int | None = 3600`
`autogluon_wrapper.py:97`: передаётся в `fit()`

### Сид внутри фреймворка

`autogluon_wrapper.py:93`:
```python
learner_kwargs={"random_state": self.seed}
```

---

## 3. H2O

### Обученные модели

Сохранённые leaderboard (`*h2o*_models.json`) — **не найдены** в `results/scores/`. По коду: H2O AutoML обучает до `max_models` моделей и строит Stacked Ensemble (`h2o_wrapper.py:150-156`).

### Ансамблирование

H2O AutoML по умолчанию строит **Stacked Ensemble** из лучших моделей. Не определено явно в коде — используется дефолтное поведение H2OAutoML.

### Число фолдов CV

`nfolds` не передаётся в H2OAutoML — используется дефолт H2O (5 фолдов). Не определено в коде явно.

### Time-лимит и max_models

| Параметр | Значение | Файл:строка |
|----------|----------|-------------|
| `max_runtime_secs` | 3600 | `h2o_wrapper.py:53` |
| `max_models` | 5 | `h2o_wrapper.py:52` |

Оба передаются в `H2OAutoML()`: `h2o_wrapper.py:151-152`

### Сид внутри фреймворка

`h2o_wrapper.py:153`:
```python
seed=self.seed
```

---

## 4. LAMA (LightAutoML)

### Обученные модели

Сохранённые leaderboard (`*lama*_models.json`) — **не найдены** в `results/scores/`. По коду: LAMA использует `TabularAutoML` с задачей `multiclass` (`lama_wrapper.py:98-99`).

### Ансамблирование

LAMA по умолчанию использует **blending** (взвешенное усреднение предсказаний). Не определено явно в коде — используется дефолтное поведение TabularAutoML.

### Число фолдов CV

`cv` не передаётся в TabularAutoML — используется дефолт LAMA (5 фолдов). Не определено в коде явно.

### Time-лимит

`lama_wrapper.py:32`: `timeout: int = 3600`
`lama_wrapper.py:101`: передаётся в `TabularAutoML()`

**Soft timeout:** `timing_params` не передаётся → дефолт `mode=1` (soft). LAMA может превышать номинальный лимит.

### Сид внутри фреймворка

`lama_wrapper.py:103`:
```python
reader_params={"random_state": self.seed}
```

---

## 3. Способы вычисления OOS-скора (prediction_mode)

### threshold

1. **OOS исключается из train** — `base.py:39-40`:
   ```python
   x_texts = [t for t, y in zip(train_texts, train_labels) if y != self.oos_label]
   y_labels = [y for y in train_labels if y != self.oos_label]
   ```
   Результат: 150 классов (in-scope only)

2. **OOS-скор = 1 - max(proba)** — в каждой обёртке:
   - `autogluon_wrapper.py:135`: `return 1.0 - proba.max(axis=1)`
   - `h2o_wrapper.py:207`: `return 1.0 - proba.max(axis=1)`
   - `lama_wrapper.py:129`: `return 1.0 - proba.max(axis=1)`

3. **Порог калибруется на VALIDATION** (не test), максимизация F1 — `base.py:72-109`:
   - Перебор порогов: `base.py:97-106`
   - Сохранение лучшего: `base.py:108`

### argmax

1. **OOS как 151-й класс в train** — `base.py:37-38`:
   ```python
   if self.prediction_mode == "argmax":
       return list(train_texts), list(train_labels)
   ```
   Результат: 151 класс (включая OOS = -1)

2. **OOS-скор = вероятность класса OOS напрямую** — в каждой обёртке:
   - `autogluon_wrapper.py:130-134`
   - `h2o_wrapper.py:200-202`
   - `lama_wrapper.py:126-128`

### Важно

**Это РАЗНЫЕ обученные модели** (разный train set) — режим нельзя переключить без перепрогона.

### Сохранение для анализа

**Скоры** → `results/scores/{model}_{mode}_{seed}_{source}_scores.npz`
`evaluation.py:140-148`:
```python
np.savez_compressed(
    scores_file,
    y_pred=y_pred,
    y_scores=y_scores,
    proba_matrix=proba_matrix,
    classes=classes,
    y_true=self.labels,
    texts=np.array(self.texts, dtype=object),
)
```

**Leaderboard/модели** → `results/scores/{model}_{mode}_{seed}_{source}_models.json`
`evaluation.py:151-187`

**Метрики** → `results/metrics.json`
`experiment_runner.py:139-145`:
```python
result.extra = {
    "framework": framework_name,
    "source": source,
    "prediction_mode": prediction_mode,
    "train_sec": round(train_sec, 2),
    "calibrate_sec": round(calibrate_sec, 2),
    **result.extra,
}
```
Timestamp: `evaluation.py:260`

---

## 4. Сопоставимость и ограничения

### Что уравнено (равные условия)

| Аспект | Реализация |
|--------|------------|
| Эмбеддинги | Одинаковые e5-large-instruct для всех |
| Few-shot выборки | Одинаковые на каждый seed (через `load_fewshot`) |
| Тестовый сплит | Один и тот же для всех |
| Способ оценки | Единый `Evaluator` с одинаковыми метриками |

### Что РАЗНОЕ by design

| Аспект | AutoGluon | H2O | LAMA | Обоснование |
|--------|-----------|-----|------|-------------|
| Набор моделей | RF, ET, CatBoost, XGBoost, LightGBM | Авто (до max_models) | Авто (LightGBM, Linear) | Дефолты фреймворков |
| Тип ансамбля | WeightedEnsemble | Stacked Ensemble | Blending | Дефолты фреймворков |
| Фолды CV | 0 (bagging off) | 5 (дефолт) | 5 (дефолт) | Не выравнивались |
| Механизм бюджета | Время | Число моделей + время | Soft-время | Разная архитектура |

**Прямое уравнивание вычислительного бюджета невозможно** из-за разных механизмов контроля. Выбран **щедрый единый потолок 3600s**, при котором каждый фреймворк работает в близком к дефолтному режиме.

### Осознанные отклонения от дефолтов

| Параметр | Значение | Дефолт | Причина |
|----------|----------|--------|---------|
| H2O `max_models` | 5 | None (unlimited) | Сопоставимость вычислительного масштаба (`h2o_wrapper.py:52`) |
| AutoGluon `excluded_model_types` | ["FASTAI"] | [] | Несовместимость с torch ≥2.10 (`autogluon_wrapper.py:98`) |

### Известные ограничения

1. **Эмбеддер без инструкции-префикса** — e5-instruct рекомендует prefix "query:" / "passage:", но `encode()` вызывается без него. Метрики могут быть ниже потенциала. Одинаково у всех → на сравнение между системами не влияет.

2. **LAMA soft-timeout** — может превышать номинальный лимит 3600s. Дефолтное поведение, не изменялось.

3. **Вариант A (native text)** — не реализован. Планируется отдельно.

4. **Фолды CV не выравнены** — AutoGluon: 0, H2O: 5, LAMA: 5. Используются дефолты каждого фреймворка.

---

**Дата составления:** 2025-06-11

**Состояние кода:** после итерации B (сохранение скоров, единый запуск argmax/threshold, FastAI исключён, лимит 3600s).
