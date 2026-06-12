# AutoIntent: состояние на момент заморозки OOS

## 1. Общая архитектура

AutoIntent (пакет `autointent` 0.2.0) — embedding-centric pipeline с тремя узлами: **embedding → scoring → decision**. Оптимизация параметров через Optuna (20 trials в пресете classic-light, `configs/classic-light-oosf1.yaml:28`).

### Точки входа

| Скрипт | Назначение | Файл |
|--------|------------|------|
| `run_autointent.py` | Обёртка train + eval | `run_autointent.py:1-167` |
| `train_autointent.py` | Обучение модели | `train_autointent.py:1-287` |
| `eval_autointent.py` | Оценка модели | `eval_autointent.py:1-222` |

**Реально используется:** `run_autointent.py` вызывает `train_autointent.py` через subprocess (`run_autointent.py:119-127,149`), затем `eval_autointent.py` (`run_autointent.py:138-142,160`).

### Эмбеддер

Фиксирован пресетом classic-light: `intfloat/multilingual-e5-large-instruct` (`train_autointent.py:75`, `configs/classic-light-oosf1.yaml:31`). Тот же эмбеддер, что у фреймворков. Инструкция-префикс НЕ подаётся (аналогично фреймворкам).

**Эмбеддер НЕ оптимизируется:** пресеты AutoIntent 0.2.0 не содержат `node_type: embedding` в `search_space` (`configs/classic-light-oosf1.yaml:1-25` — только `scoring` и `decision`).

### Сид

Внутренний сид AutoIntent фиксирован (LinearScorer seed=0, не управляем извне). Аргумент `args.seed` влияет только на few-shot выборку (`train_autointent.py:200`). На режиме `full` метрики идентичны по сидам (данные те же).

---

## 2. Три узла оптимизации

### scoring

Оптимизируется под `target_metric: scoring_f1` (`configs/classic-light-oosf1.yaml:3`).

**Модули** (`configs/classic-light-oosf1.yaml:4-14`):
- `knn` — k-nearest neighbors (k: 1-20, weights: uniform/distance/closest)
- `linear` — линейный скорер
- `mlknn` — multi-label kNN (k: 1-20)

### decision

Оптимизируется под `target_metric` (см. раздел 3) (`configs/classic-light-oosf1.yaml:16`).

**Модули** (`configs/classic-light-oosf1.yaml:17-25`):
- `threshold` — порог на OOS-скор (thresh: 0.1-0.9)
- `argmax` — argmax без порога
- `jinoos` — JIN-OOS метод
- `tunable` — настраиваемый порог
- `adaptive` — адаптивный порог

---

## 3. Проблема decision_accuracy и патч oos_f1

### Дефолтное поведение (decision_accuracy)

Пресет classic-light оптимизирует decision под `target_metric: decision_accuracy`. При ~10% OOS в тесте это даёт:
- Низкий порог (~0.13) → модель редко предсказывает OOS
- Высокий in-domain accuracy (~95%)
- Низкий OOS recall (~55-66%)
- f1_oos ≈ 0.65-0.82 (по всем режимам и seeds)

**Подтверждение из metrics.json:**
```
autointent_classic-light | standard | full | f1_oos=0.6989 | oos_recall=0.5570
autointent_classic-light | standard | 10shot | seed=42 | f1_oos=0.6717 | oos_recall=0.5340
```

### Асимметрия с фреймворками

Фреймворки калибруют порог под **f1_oos** (`base.py:103`):
```python
score = f1_oos(y_true, y_pred, oos_label=self.oos_label)
```

AutoIntent с дефолтом калибрует под **decision_accuracy**. Разные цели оптимизации → несимметричное сравнение на OOS-метриках.

### Патч: кастомная метрика oos_f1

**Регистрация метрики** (`train_autointent.py:113-127`):
```python
def register_oos_f1_metric() -> None:
    from autointent.metrics import DECISION_METRICS

    def oos_f1(y_true, y_pred):
        """Бинарный F1 по OOS-классу. OOS=None → positive class."""
        y_true_bin = [1 if y is None else 0 for y in y_true]
        y_pred_bin = [1 if y is None else 0 for y in y_pred]
        return float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    DECISION_METRICS["oos_f1"] = oos_f1
```

Метрика идентична `f1_oos` из `src/metrics.py`, но работает с форматом AutoIntent (OOS=None).

**Кастомный конфиг** создаётся программно (`train_autointent.py:78-110`):
- Читает оригинальный `classic-light.yaml` из пакета autointent
- Заменяет `target_metric: decision_accuracy` → `target_metric: oos_f1` (только в decision-ноде)
- Сохраняет в `configs/classic-light-oosf1.yaml`

**Флаг выбора** (`train_autointent.py:163-168`):
```python
parser.add_argument(
    "--decision-metric",
    choices=["decision_accuracy", "oos_f1"],
    default="decision_accuracy",
)
```

**Создание pipeline** (`train_autointent.py:220-229`):
```python
if decision_metric == "oos_f1":
    register_oos_f1_metric()
    config_path = ensure_oosf1_config()
    pipeline = Pipeline.from_optimization_config(config_path)
else:
    pipeline = Pipeline.from_preset("classic-light")
```

**Различение записей:** суффикс `_oosf1` в `model_name` (`train_autointent.py:65-69`):
```python
def get_model_name(pilot: bool, decision_metric: str) -> str:
    suffix = "_oosf1" if decision_metric == "oos_f1" else ""
    ...
    return f"autointent_classic-light{suffix}"
```

**Scoring не трогается:** `target_metric: scoring_f1` остаётся без изменений.

---

## 4. Что НЕ входит в матрицу фреймворков

AutoIntent — отдельная система, не в общей AutoML-матрице (AutoGluon, H2O, LAMA). Сравнивается с фреймворками по итоговым метрикам на том же тестовом наборе, но:
- Другая архитектура (embedding-centric vs tabular AutoML)
- Другой механизм оптимизации (Optuna vs дефолты фреймворков)
- Другая метрика оптимизации (decision_accuracy или oos_f1 vs accuracy/mean_per_class_error/crossentropy)

---

## 5. Ограничения и статус

### Расхождение с литературой

Локальный AutoIntent с дефолтным `decision_accuracy` даёт OOS-F1 ≈ 0.65-0.82 (варьируется по режимам и seeds) против **76.79** в статье (arxiv 2509.21138, Table 3).

### oos_f1-вариант

Реализован и SMOKE-проверен (`train_autointent.py:220-227`). Результаты из metrics.json:
```
autointent_classic-light_oosf1 | deeppavlov | 10shot | seed=42 | f1_oos=0.7518 | oos_recall=0.9040
```

Сравнение 10shot seed=42 deeppavlov:
| Вариант | Порог | OOS Recall | F1 OOS |
|---------|-------|------------|--------|
| decision_accuracy | ~0.137 | ~0.669 | ~0.70 |
| oos_f1 | ~0.296 | **0.904** | **0.752** |

**Полная сетка обоих режимов НЕ выполнена.**

## Source-зависимость AutoIntent (в отличие от фреймворков)

AutoIntent **использует OOS-примеры** в обучении: в few-shot выборку они
попадают как записи без поля `label` (= `None` в формате AutoIntent,
`data_utils.py:206-211`) и участвуют в калибровке decision-порога
(`train_autointent.py`). Поэтому число OOS в train **влияет** на метрики
AutoIntent — в отличие от AutoML-фреймворков, которые в threshold-режиме
исключают OOS из train (`base.py`) и калибруют порог только на validation,
из-за чего source на них не влияет (standard == deeppavlov).

Число OOS, реально попадающих в few-shot (ограничено доступным в train):

| n_shots | запрошено (150×n×0.1) | standard (доступно 100) | deeppavlov (доступно 200) |
|---------|----------------------|-------------------------|----------------------------|
| 10 | 150 | 100 (потолок) | 150 |
| 20 | 300 | 100 (потолок) | 200 (потолок) |
| 50 | 750 | 100 (потолок) | 200 (потолок) |

**Следствие для интерпретации:** на `standard` число OOS упирается в 100
уже на 10-shot и не растёт дальше — AutoIntent не может улучшать калибровку
порога с ростом n_shots по OOS. На `deeppavlov` доступно вдвое больше OOS,
что даёт систематически более высокий f1_oos (наблюдается dp ≥ std почти
на всех конфигурациях). Это структурное свойство данных, а не «лучшесть»
одного source.

Это корректное поведение по дизайну (AutoIntent — OOS-aware), не баг.
Методологически отражает преимущество AutoIntent: он извлекает пользу из
OOS-примеров, которые табличные фреймворки в threshold-режиме игнорируют.

### Невоспроизводимые данные

Прежние Kaggle-метрики AutoIntent (`FULL_AUTO_BASELINE` в `kaggle_oos_autointent.ipynb`) — захардкоженные литералы, источник вычисления отсутствует, невоспроизводимы → **удалены из metrics.json**.

---

**Дата:** 2026-06-12

Результаты предварительные, полная сетка oos_f1 не выполнена.
