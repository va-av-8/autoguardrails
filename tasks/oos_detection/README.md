# Out-of-Scope Detection

## Исследовательский вопрос

Применим ли AutoIntent как OOS-guardrail — в том числе при ограниченном
количестве обучающих данных (реалистичный production-сценарий)? И как он
соотносится с табличными AutoML-фреймворками (AutoGluon, H2O, LightAutoML)
при сравнении на равных входных признаках?

## Мотивация

При развёртывании агентной системы разработчик располагает описанием
системы и ограниченным набором примеров (10–50 на intent). Нужен
надёжный OOS-детектор без глубокой ML-экспертизы. AutoIntent как
AutoML-движок — кандидат на роль такого инструмента "из коробки".
Для честного сравнения мы прогоняем и универсальные табличные
AutoML-фреймворки на тех же эмбеддингах, что и AutoIntent.

> **Статус:** результаты в этом документе предварительные. После
> расследования метрик (см. `docs/oos_findings.md`) эксперименты требуют
> единообразного перепрогона (единые лимиты, H2O на x86/Windows,
> oos_f1-вариант AutoIntent). Прежние Kaggle-числа невалидны (см. ниже).

## Датасет

**CLINC150** (Larson et al., 2019). Используются два source с **идентичным
тест-сплитом**, различающиеся числом OOS-примеров в train:

| Source | Train (всего / in-scope / OOS) | Validation | Test (всего / in-scope / OOS) | Источник |
|--------|-------------------------------|------------|-------------------------------|----------|
| `standard` | 15 100 / 15 000 / **100** | 3 100 / 3 000 / 100 | 5 500 / 4 500 / 1 000 | github.com/clinc/oos-eval |
| `deeppavlov` | 15 200 / 15 000 / **200** | 3 100 / 3 000 / 100 | 5 500 / 4 500 / 1 000 | HuggingFace `DeepPavlov/clinc150` |

- 150 intent-классов, 10 доменов.
- Тест-сплит одинаков у обоих source (4 500 in-scope + 1 000 OOS).
- Метки (стандартный формат): OOS = `-1`, in-scope = `0..149`.

**Few-shot сэмплинг:** из train сэмплируем n примеров на intent
(n ∈ {10, 20, 50}), OOS пропорционально (n_oos = n_intents × n_shots × 0.1).
Few-shot выборки предгенерированы с фиксированным сидом и общие для всех
систем. Тест-сплит всегда фиксированный.

## Метрики

Схема соответствует академической OOS NLP-литературе.

### Основные

| Метрика | Описание |
|---|---|
| **OOS Recall** | TPR на OOS-классе |
| **In-domain Accuracy** | Accuracy на in-scope примерах |
| **F1 OOS** | бинарный F1 по OOS-классу (OOS = positive) |

### Вспомогательные

| Метрика | Описание |
|---|---|
| **AUROC** | по continuous OOS-скорам |
| **AU-IOC** | площадь in-domain/OOS-кривой |
| **Latency (ms)** | среднее время инференса на 1 запрос |

Continuous OOS-скоры для AutoIntent извлекаются через
`pipeline.predict_with_metadata()` (`utterances[].score`), с фолбэком на
бинарный скор, если метаданные недоступны (`src/metrics.py`).

## Сравниваемые системы

### AutoIntent
Embedding-centric pipeline (embedding → scoring → decision), пресет
`classic-light`, эмбеддер `multilingual-e5-large-instruct` (фиксирован
пресетом). Подробно — `docs/autointent.md`.

### AutoML-фреймворки (бейзлайны)
AutoGluon, H2O, LightAutoML — работают на тех же e5-эмбеддингах, что и
AutoIntent (числовая матрица, не сырой текст). Подробно — `docs/wrappers.md`.

### prediction_mode (для фреймворков)
- **threshold:** OOS исключён из train (150 классов), OOS-скор = 1 − max(proba),
  порог калибруется на validation под F1 OOS.
- **argmax:** OOS как 151-й класс в train, OOS-скор = proba(OOS).

Это разные обученные модели; режим не переключается без перепрогона.        
Фреймворки не имеют узла оптимизации выбора решения и отдают сырые скоры.

## Сравнение с литературой

| Метод | Протокол | Сравнимость |
|---|---|---|
| AutoIntent (Table 3, arxiv 2509.21138) | CLINC150, 150 интентов + отдельный OOS-класс | **Напрямую сравним** |
| ADB / DA-ADB / DETER | TEXTOIR (варьируемая доля известных интентов 25/50/75%) | Не сравним напрямую |

В нашем протоколе все 150 интентов фиксированы, OOS — отдельный класс.
Прямое сравнение чисел с TEXTOIR-методами некорректно. Единственный
напрямую сравнимый референс — AutoIntent Table 3 (In-domain Acc 96.13,
OOS-F1 76.79).

## Запуск

### Подготовка данных

```bash
uv run python scripts/prepare_data.py --source all
# или по отдельности:
uv run python scripts/prepare_data.py --source standard    # 100 OOS train
uv run python scripts/prepare_data.py --source deeppavlov  # 200 OOS train
```

Структура после подготовки: `data/processed/<source>/{full.json, fewshot.json, meta.json}`.

**Формат AutoIntent** (при обучении): in-scope — `{"utterance": "...", "label": 0..149}`,
OOS — `{"utterance": "..."}` (без поля `label`). Конвертация:
`load_split_autointent()`, `load_fewshot_autointent()`.

### AutoIntent

```bash
# Раздельный train / eval (используется для прогонов)
uv run python scripts/train_autointent.py --source <source> --mode fewshot --n_shots <n> --seed <seed>
uv run python scripts/eval_autointent.py --model_dir runs/autointent_classic-light_<source>_<n>shot_seed<seed>

# Обёртка train+eval одним вызовом
uv run python scripts/run_autointent.py --source <source> --mode full
uv run python scripts/run_autointent.py --source <source> --mode fewshot --n_shots <n> --seed <seed>

# Оптимизация decision-порога под OOS-F1
uv run python scripts/run_autointent.py --source <source> --mode fewshot --n_shots <n> --seed <seed> --decision-metric oos_f1

# Быстрая валидация на лёгком эмбеддере e5-small
uv run python scripts/run_autointent.py --source <source> --mode fewshot --n_shots <n> --seed <seed> --pilot
```

**Директории моделей:**
- `runs/autointent_classic-light_<source>_<mode>_seed<seed>`
- с oos_f1: `runs/autointent_classic-light_oosf1_<source>_<mode>_seed<seed>`
- pilot: суффикс `_pilot`

### Framework Benchmarks

```bash
# threshold mode (in-scope train + калиброванный OOS-скор)
uv run python scripts/run_framework_benchmarks.py \
  --frameworks autogluon h2o lama \
  --sources standard deeppavlov \
  --n-shots 10 20 50 --seeds 42 123 456 \
  --run-full --run-fewshot

# argmax mode (OOS как 151-й класс)
uv run python scripts/run_framework_benchmarks.py \
  --frameworks autogluon lama \
  --sources standard deeppavlov \
  --n-shots 10 20 50 --seeds 42 123 456 \
  --run-full --run-fewshot --prediction-mode argmax

# Краткий summary
uv run python scripts/summarize_results.py
```

Результаты — в `results/metrics.json`.

> **H2O запускается на x86/Windows**, а не на Apple Silicon: на ARM
> недоступен нативный XGBoost, и H2O не строит модели в разумном бюджете.
> H2O ограничен числом моделей (`max_models=5`), а не временем — это
> рекомендованный H2O способ воспроизводимости. Подробности и оговорки
> по сопоставимости — `docs/wrappers.md`.

### Параметры

| Параметр | Значения | Описание |
|----------|----------|----------|
| `--source` | `standard`, `deeppavlov` | Датасет |
| `--mode` | `full`, `fewshot` | Режим обучения |
| `--n_shots` | `10`, `20`, `50` | Примеров на интент (few-shot) |
| `--seed` | `42`, `123`, `456` | Random seed (few-shot; на full не варьируется) |
| `--pilot` | flag | Лёгкий эмбеддер e5-small (быстрая валидация) |
| `--decision-metric` | `decision_accuracy`, `oos_f1` | Метрика оптимизации decision-узла AutoIntent |
| `--prediction-mode` | `threshold`, `argmax` | Режим OOS-скора (фреймворки) |

## Результаты

> **Предварительные результаты.** Source `standard` (100 OOS в train),
> threshold-режим, few-shot. Метрики — среднее ± std по 3 сидам (42, 123, 456).
> AutoML-фреймворки на единых e5-эмбеддингах (бюджет 600 с). AutoIntent в
> дефолтном `classic-light` (decision-порог под `decision_accuracy`).

### F1 OOS (standard, few-shot)

| Модель | 10-shot | 20-shot | 50-shot |
|--------|---------|---------|---------|
| **AutoIntent** (classic-light) | **0.722 ± 0.036** | **0.742 ± 0.065** | **0.725 ± 0.011** |
| AutoGluon (threshold) | 0.573 ± 0.037 | 0.617 ± 0.064 | 0.554 ± 0.119 |
| LightAutoML (threshold) | 0.698 ± 0.024 | 0.712 ± 0.021 | 0.661 ± 0.041 |

### OOS Recall (standard, few-shot)

| Модель | 10-shot | 20-shot | 50-shot |
|--------|---------|---------|---------|
| **AutoIntent** | **0.611 ± 0.055** | **0.644 ± 0.092** | **0.592 ± 0.014** |
| AutoGluon | 0.510 ± 0.067 | 0.554 ± 0.113 | 0.546 ± 0.155 |
| LightAutoML | 0.586 ± 0.039 | 0.607 ± 0.051 | 0.527 ± 0.056 |

### In-domain Accuracy (standard, few-shot)

| Модель | 10-shot | 20-shot | 50-shot |
|--------|---------|---------|---------|
| AutoIntent | 0.922 ± 0.002 | 0.937 ± 0.004 | 0.956 ± 0.003 |
| AutoGluon | 0.826 ± 0.012 | 0.871 ± 0.007 | 0.807 ± 0.025 |
| **LightAutoML** | **0.932 ± 0.003** | **0.947 ± 0.005** | **0.961 ± 0.003** |

**Вычислительная эффективность (предварительно):**
AutoIntent (classic-light) обучается за ~110 с — фиксированный бюджет из
20 Optuna trials, доходит до конца независимо от лимита времени.
AutoML-фреймворкам задан временно́й бюджет 600 с, но фактическое время
сильно различается из-за разных механизмов его контроля: AutoGluon
заполняет бюджет почти полностью (~620–640 с с учётом эмбеддинга),
а LightAutoML его существенно превышает (~1600 с на 10-shot, до ~4700 с
на 50-shot) — несмотря на заданный hard-режим таймера (`timing_params.mode=2`),
LightAutoML не прерывает обучение внутри отдельного алгоритма, поэтому
доходит до конца текущей модели независимо от лимита. Прямое сравнение
времени некорректно из-за этих различий; порядок величины показывает, что
AutoIntent на 1–2 порядка экономнее. Inference latency не приводится —
требует переизмерения (границы замера у AutoIntent и фреймворков различаются).

**Наблюдения (предварительно):**
- AutoIntent лидирует по **F1 OOS** на всех few-shot уровнях при сравнении
  на равных e5-эмбеддингах.
- AutoIntent использует OOS-примеры в калибровке (source влияет: deeppavlov
  с 200 OOS даёт выше f1_oos, чем standard со 100). Фреймворки в threshold
  исключают OOS из train (source-инвариантны). Это различие — в пользу
  AutoIntent: он эксплуатирует OOS-данные, которые фреймворки игнорируют.
- LightAutoML даёт наивысший **in-domain accuracy**, но уступает AutoIntent
  по OOS-метрикам — баланс смещён в сторону in-scope.
- AutoGluon на единых эмбеддингах слабее обоих по OOS; высокая дисперсия
  на 50-shot (±0.119) отражает недетерминизм при заполнении временно́го бюджета.
- Числа AutoIntent — для дефолтного `decision_accuracy`; вариант
  `--decision-metric oos_f1` ожидаемо повысит OOS-метрики (не включён в
  таблицу — полная сетка не прогонялась).

### Установленные методологические факты (см. `docs/oos_findings.md`)

- **Локальный AutoIntent с дефолтным `classic-light` даёт OOS-F1 ≈ 0.72**,
  ниже Table 3 (76.79), при совпадающем in-domain. Причина в процессе выяснения.    
- **На равных эмбеддингах (e5-large) разрыв между AutoIntent и
  фреймворками сокращается** относительно статьи: предположительно, в Table 3 фреймворки
  использовали собственную обработку текста (H2O word2vec, AutoGluon
  fine-tuning, LAMA TF-IDF), что давало им низкие числа. На общих e5-эмбеддингах
  фреймворки сильнее. Основное преимущество AutoIntent при выравнивании —
  **эффективность** (обучение ~100 с против минут у фреймворков).

## Проверка гипотез

| # | Гипотеза | Вывод |
|---|---|---|
| HYP-001 | Per-intent threshold calibration | Отложена: в few-shot недостаточно данных на кластер |
| HYP-002 | Асимметричная cost function (α·FNR + β·FPR) | Опровергнута: невыгодный trade-off, нестабильна при 2:1 |

## Открытые вопросы к авторам AutoIntent

1. **Конфигурация CLINC150 в Table 3** — `full` (100 OOS train) или
   `plus` (≈200–250 OOS train)? В статье указано только "CLINC150
   (Larson et al., 2019)". - **Закрыт**: source = deeppavlov, 200 OOS train.
2. **AUROC в AutoIntent** — `pipeline.predict()` возвращает только бинарные
   предсказания; мы берём continuous-скоры через `predict_with_metadata()`.
   Это публичный API или внутренний? - **Закрыт**: публичный.
3. **Метрика оптимизации decision в Table 3** — статья подчёркивает
   "dedicated confidence thresholds tuning". Дефолтный `classic-light`
   оптимизирует decision под `decision_accuracy`, что для OOS-детекции
   на несбалансированных данных даёт низкий OOS-F1. Под какую метрику
   настраивался decision-узел в Table 3?
4. **Embedder в Table 3** — фиксировался ли явно? Пресет `classic-light`
   фиксирует эмбеддер и не оптимизирует его выбор (в 0.2.0 нет node_type
   embedding). - **Закрыт**: эмбеддер `multilingual-e5-large-instruct`
5. **Режим обработки текстов AutoML фреймворков в Table 3** - использовалась встроенная или единый эмбеддер?

## Ноутбуки

| Ноутбук | Содержание |
|---|---|
| `01_eda.ipynb` | EDA CLINC150, сравнение standard vs deeppavlov |
| `02_hypothesis_asymmetric_cost.ipynb` | HYP-002: асимметричная cost function |
| `03_results_summary.ipynb` | Итоговое сравнение (требует обновления после перепрогона) |

Эксперименты запускаются скриптами, результаты — в `results/metrics.json`.

## Документация

| Документ | Содержание |
|---|---|
| `docs/wrappers.md` | Реализация обёрток AutoML (модели, ансамбли, метрики выбора, бюджет, оговорки по сопоставимости) |
| `docs/autointent.md` | Обёртка AutoIntent, узлы оптимизации, патч `oos_f1` |
| `docs/oos_findings.md` | Карта выводов расследования метрик (что невалидно и почему) |

## Известные проблемы качества данных

- **Оба источника:** 2 текста пересекаются между train и test с разными метками.
- **deeppavlov:** train OOS содержит аномально длинные тексты (mean ~69 слов
  против ~7 в standard).

Для более чистых экспериментов — `--source standard`; для сравнения с
условиями статьи — `--source deeppavlov`.

## Ссылки

- [CLINC150](https://aclanthology.org/D19-1131/) — Larson et al., EMNLP 2019
- [AutoIntent](https://arxiv.org/abs/2509.21138) — EMNLP 2025
- [AutoIntent GitHub](https://github.com/deeppavlov/AutoIntent)
- [ADB](https://arxiv.org/abs/2012.10209) — Zhang et al., AAAI 2021
- [DA-ADB](https://arxiv.org/abs/2203.05687) — Zhang et al., TASLP 2023
- [DETER](https://arxiv.org/abs/2405.19967) — Rashwan et al., LREC-COLING 2024

**AutoML-фреймворки:**
- [AutoGluon](https://github.com/autogluon/autogluon) — [docs](https://auto.gluon.ai/) · Erickson et al., 2020 ([arxiv](https://arxiv.org/abs/2003.06505))
- [H2O AutoML](https://github.com/h2oai/h2o-3) — [docs](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) · LeDell & Poirier, ICML AutoML 2020
- [LightAutoML (LAMA)](https://github.com/sb-ai-lab/LightAutoML) — [docs](https://lightautoml.readthedocs.io/) · Vakhrushev et al., 2022 ([arxiv](https://arxiv.org/abs/2109.01528))
