# Out-of-Scope Detection

## Исследовательский вопрос

Применим ли AutoIntent как OOS-guardrail — в том числе при ограниченном
количестве обучающих данных (реалистичный production-сценарий)?

## Мотивация

При развёртывании агентной системы разработчик располагает описанием
системы и ограниченным набором примеров (10–50 на intent). Нужен
надёжный OOS-детектор без глубокой ML-экспертизы. AutoIntent как
AutoML-движок — кандидат на роль такого инструмента "из коробки".

## Датасет

**CLINC150** (Larson et al., 2019), конфигурация `plus`
- 150 intent-классов, 10 доменов
- Train: 15 250 примеров (15 000 in-scope + 250 OOS)
- Validation: 3 100 (3 000 + 100 OOS)
- Test: 5 500 (4 500 + 1 000 OOS), OOS base rate 18.2%
- HuggingFace: `DeepPavlov/clinc150`

**Few-shot сэмплинг:**
Из train сэмплируем n примеров на intent (n ∈ {10, 20, 50}).
OOS-примеры пропорционально: n_oos = n_intents × n_shots × 0.1.
Тест-сплит всегда фиксированный. Threshold калибруется через
cross-validation на train (AutoIntent) — внешний val не используется.

## Метрики

Схема соответствует академической OOS NLP-литературе для
прямого сравнения с ADB и AutoIntent.

### Основные

| Метрика | Описание | Литературный прецедент |
|---|---|---|
| **OOS Recall** | TPR на OOS-классе | ADB (AAAI 2021), AutoIntent (EMNLP 2025) |
| **In-domain Accuracy** | Accuracy на in-scope примерах | AutoIntent Table 3, ADB |
| **F1 OOS** | F1-score на OOS-классе | AutoIntent Table 3, ADB |

### Вспомогательные

| Метрика | Описание |
|---|---|
| **AUROC** | Continuous OOS-скоры через `pipeline._nodes[0]` |
| **Latency (ms)** | Среднее время инференса на 1 запрос |

## Эксперименты

### Бейзлайны
- TF-IDF + LogReg с threshold calibration
- Cosine similarity + threshold (embedders: BERT, MiniLM, E5-Large-Instruct)

### AutoIntent
- Few-shot: n ∈ {10, 20, 50}, 3 seeds (42, 123, 456)
- Full train: seed=42
- Preset: `classic-light` (multilingual-e5-large-instruct)

Референс из статьи (AutoIntent Table 3): F1 OOS = 76.79,
In-domain Accuracy = 96.13.

## Запуск

### Подготовка данных

```bash
# Скачать и подготовить оба датасета
uv run python scripts/prepare_data.py --source all

# Или по отдельности
uv run python scripts/prepare_data.py --source standard    # github.com/clinc/oos-eval (100 OOS train)
uv run python scripts/prepare_data.py --source deeppavlov  # HuggingFace DeepPavlov/clinc150 (200 OOS train)
```

**Структура данных после подготовки:**

```
data/processed/
├── standard/
│   ├── full.json      # полные сплиты
│   ├── fewshot.json   # few-shot выборки
│   └── meta.json      # метаданные
└── deeppavlov/
    ├── full.json
    ├── fewshot.json
    └── meta.json
```

**Формат файлов:**

`full.json` — полные сплиты:
```json
{
  "train": {"texts": ["...", ...], "labels": [0, 5, -1, ...]},
  "validation": {"texts": [...], "labels": [...]},
  "test": {"texts": [...], "labels": [...]}
}
```

`fewshot.json` — few-shot выборки (n_shots × seeds):
```json
{
  "n10": {
    "seed42": {"texts": [...], "labels": [...]},
    "seed123": {...},
    "seed456": {...}
  },
  "n20": {...},
  "n50": {...}
}
```

`meta.json` — метаданные:
```json
{
  "source": "deeppavlov",
  "n_intents": 150,
  "oos_label": -1,
  "splits": {"train": {"total": 15100, "n_inscope": 14900, "n_oos": 200}, ...},
  "fewshot": {"n_shots": [10, 20, 50], "seeds": [42, 123, 456], "oos_ratio": 0.1},
  "intents": [{"id": 0, "name": "intent_0"}, ...]
}
```

**Метки (стандартный формат):** OOS = `-1`, in-scope = `0..149`

**Формат AutoIntent** (используется при обучении):
- in-scope: `{"utterance": "текст", "label": 0..149}`
- OOS: `{"utterance": "текст"}` (без поля `label`)

Конвертация: `load_split_autointent()`, `load_fewshot_autointent()`

### Baselines

```bash
# Full train
uv run python scripts/run_baseline.py --source <source> --model all --mode full

# Few-shot
uv run python scripts/run_baseline.py --source <source> --model all --mode fewshot --n_shots <n> --seed <seed>

# Только TF-IDF или Cosine
uv run python scripts/run_baseline.py --source <source> --model tfidf --mode fewshot --n_shots <n> --seed <seed>
uv run python scripts/run_baseline.py --source <source> --model cosine --mode fewshot --n_shots <n> --seed <seed>
```

**Batch-запуск всех бейзлайнов:**

```bash
for source in standard deeppavlov; do
  # Full train
  uv run python scripts/run_baseline.py --source $source --model all --mode full
  # Few-shot
  for n in 10 20 50; do
    for seed in 42 123 456; do
      uv run python scripts/run_baseline.py --source $source --model all --mode fewshot --n_shots $n --seed $seed
    done
  done
done
```

### AutoIntent

```bash
# Pilot (e5-small, быстрая валидация)
uv run python scripts/run_autointent.py --source <source> --mode fewshot --n_shots <n> --seed <seed> --pilot

# С фиксацией embedder (e5-large-instruct, сравнимо с Table 3)
uv run python scripts/run_autointent.py --source <source> --mode full
uv run python scripts/run_autointent.py --source <source> --mode fewshot --n_shots <n> --seed <seed>

# Раздельный запуск train/eval
uv run python scripts/train_autointent.py --source <source> --mode fewshot --n_shots <n> --seed <seed>
uv run python scripts/eval_autointent.py --model_dir runs/autointent_classic-light_<source>_<n>shot_seed<seed>
```

**Структура директорий моделей:**
- Final: `runs/autointent_classic-light_<source>_<mode>_seed<seed>`
- Pilot: `runs/autointent_classic-light_pilot_<source>_<mode>_seed<seed>`

**Batch-запуск всех экспериментов:**

```bash
# Batch-запуск всех экспериментов
for source in standard deeppavlov; do
  uv run python scripts/run_autointent.py --source $source --mode full
  for n in 10 20 50; do
    for seed in 42 123 456; do
      uv run python scripts/run_autointent.py --source $source --mode fewshot --n_shots $n --seed $seed
    done
  done
done
```

### Framework Benchmarks (CLI, без ноутбука)

```bash
# Default AutoML: OOS как 151-й класс (argmax), без threshold calibration
uv run python tasks/oos_detection/scripts/run_framework_benchmarks.py \
  --frameworks autogluon h2o lama \
  --sources deeppavlov \
  --n-shots 10 20 50 \
  --seeds 42 123 456 \
  --run-full \
  --run-fewshot \
  --prediction-mode argmax

# Threshold mode (in-scope only train + calibrated OOS score)
uv run python tasks/oos_detection/scripts/run_framework_benchmarks.py \
  --frameworks autogluon h2o lama \
  --sources standard deeppavlov \
  --n-shots 10 20 50 \
  --seeds 42 123 456 \
  --run-full \
  --run-fewshot

# Краткий summary по metrics.json
uv run python tasks/oos_detection/scripts/summarize_results.py
```

Результаты сохраняются в `tasks/oos_detection/results/metrics.json` через существующий `Evaluator.save(...)`.

**Kaggle (догоняющий AutoML argmax):** `notebooks/kaggle_oos_automl_frameworks.ipynb` — код репозитория вшит в ноутбук (без git), `HF_TOKEN` в ячейке 2, очередь прогонов строится из `metrics.json` (`python scripts/build_kaggle_automl_notebook.py`). Логи: только `RUN_START` / `RUN_FINISH`. После прогона: `python scripts/append_kaggle_automl_argmax_metrics.py` (или вручную из output).

### Параметры

| Параметр | Значения | Описание |
|----------|----------|----------|
| `--source` | `standard`, `deeppavlov` | Датасет |
| `--mode` | `full`, `fewshot` | Режим обучения |
| `--n_shots` | `10`, `20`, `50` | Примеров на интент (few-shot) |
| `--seed` | `42`, `123`, `456` | Random seed (few-shot) |
| `--model` | `tfidf`, `cosine`, `all` | Бейзлайн (run_baseline.py) |
| `--pilot` | flag | Быстрый embedder e5-small |
| `--decision-metric` | `decision_accuracy`, `oos_f1` | Метрика для decision node |

### Датасеты

| Source | OOS в train | Источник |
|--------|-------------|----------|
| `standard` | 100 | github.com/clinc/oos-eval |
| `deeppavlov` | 200 | HuggingFace DeepPavlov/clinc150 |

**Известные проблемы качества данных:**
- **Оба источника:** 2 текста пересекаются между train и test с РАЗНЫМИ метками
- **deeppavlov:** Train OOS содержит аномально длинные тексты (restaurant reviews, mean ~69 слов vs ~7 в standard)

**Рекомендация:** используйте `--source standard` для более чистых экспериментов (нет аномальных OOS),
`--source deeppavlov` для сравнения с Table 3.

### Примечание о сравнении с литературой

Специализированные OOS-методы (ADB, DA-ADB, DETER) публикуют
результаты в протоколе TEXTOIR: варьируемая доля известных
интентов (25%/50%/75%), остальные становятся OOS при тесте.
В нашем протоколе все 150 интентов фиксированы, OOS — отдельный
класс. Прямое сравнение чисел некорректно. Единственный
напрямую сравнимый референс — AutoIntent Table 3.

## Результаты

> **TODO:** Результаты ниже устарели (были получены на неверном датасете).
> После перезапуска экспериментов на standard/deeppavlov обновить через `06_results_summary.ipynb`.

### Few-shot (F1 OOS, mean ± std, 3 seeds)

| Модель | 10-shot | 20-shot | 50-shot |
|---|---|---|---|
| **AutoIntent classic-light** | **0.724 ± 0.022** | **0.819 ± 0.012** | **0.730 ± 0.007** |
| cosine_e5large_threshold | 0.660 ± 0.057 | 0.685 ± 0.011 | 0.693 ± 0.040 |
| cosine_minilm_threshold | 0.624 ± 0.077 | 0.649 ± 0.076 | 0.671 ± 0.012 |
| tfidf_threshold | 0.221 ± 0.019 | 0.218 ± 0.043 | 0.314 ± 0.077 |

### Full train

| Модель | OOS Recall | In-Domain Acc | F1 OOS | AUROC | Latency (ms) |
|---|---|---|---|---|---|
| **AutoIntent classic-light** | **0.835** | **0.940** | **0.841** | **0.974** | 0.19 |
| cosine_e5large_threshold | 0.595 | 0.908 | 0.719 | 0.961 | 19.24 |
| cosine_minilm_threshold | 0.494 | 0.875 | 0.642 | 0.963 | 7.12 |
| tfidf_threshold | 0.277 | 0.884 | 0.417 | 0.898 | 3.14 |
| AutoIntent Table 3 (Golubev et al., 2025)† | — | 0.961 | 0.768 | — | — |

† Reported numbers, same protocol (CLINC150, all 150 intents, separate OOS class).

### Ключевые наблюдения

- **AutoIntent применим как guardrail.** На 20-shot достигает F1=0.819,
  превосходя все бейзлайны на +13–19 п.п. при 20 примерах на интент.
- **Embedder критичен.** E5-Large-Instruct vs E5-Small: +5 п.п. F1,
  +5.7 п.п. In-Domain Accuracy.
- **AutoML стабильнее бейзлайнов** в 3–4 раза (CV ~3% vs ~9–12%).
- **Аномалия 50-shot** — не баг алгоритма: AutoML стабильно выбирает
  `linear + threshold`, провал объясняется переобучением threshold
  на val-фолдах (val/test gap +0.218 vs +0.073 на 20-shot).
- **Рекомендация:** 20 примеров на интент — оптимальный порог
  для production. Threshold по умолчанию менять не нужно
  (FPR=0.066 при OOS Recall=0.899).

## Проверка гипотез

| # | Гипотеза | Вывод |
|---|---|---|
| HYP-001 | Per-intent threshold calibration | Отложена: в few-shot режиме недостаточно данных на кластер |
| HYP-002 | Асимметричная cost function (α·FNR + β·FPR) | Опровергнута: невыгодный trade-off, нестабильна при 2:1 |

## Ноутбуки

| Ноутбук | Содержание |
|---|---|
| `01_eda.ipynb` | EDA датасета CLINC150, сравнение standard vs deeppavlov |
| `05_hypothesis_asymmetric_cost.ipynb` | HYP-002: асимметричная cost function |
| `06_results_summary.ipynb` | Итоговое сравнение всех моделей и выводы |

Эксперименты (бейзлайны, AutoIntent) запускаются скриптами, результаты в `results/metrics.json`.

## Ограничение по датасету

Мы используем `DeepPavlov/clinc150` — датасет, подготовленный самой
командой AutoIntent из `cmaldona/All-Generalization-OOD-CLINC150`.
По структуре он соответствует `plus`-конфигурации оригинального
CLINC150 (250 OOS в train, 15 250 строк train всего).

Статья AutoIntent (Table 3) указывает только "CLINC150 (Larson et al.,
2019)" без уточнения конфигурации (`full` vs `plus`). Поскольку авторы
сами создали `DeepPavlov/clinc150`, высока вероятность что они
использовали именно его — то есть ту же конфигурацию что и мы.
Тогда расхождение наших результатов (F1=0.841) с Table 3 (F1=0.768)
объясняется обновлением библиотеки, а не разными данными.

Большинство специализированных OOS-работ (ADB, DETER) используют
`clinc/clinc_oos` (`full`, 100 OOS train) — другой датасет
с меньшим числом OOS-примеров в train.

## Открытые вопросы к авторам AutoIntent

1. **Какая конфигурация CLINC150 использована в Table 3?**
   `full` (100 OOS train) или `plus` (250 OOS train)?
   Если `DeepPavlov/clinc150` — это подтверждает сравнимость
   наших результатов, если `full` — нужен перезапуск.

2. **Как вычисляется AUROC в AutoIntent?**
   `pipeline.predict()` возвращает только бинарные предсказания.
   Мы получаем continuous скоры через `pipeline._nodes[0]`
   (`predict_with_metadata`) — это официально поддерживаемый способ
   или внутренний API, который может измениться?

3. **Аномалия на 50-shot — известное поведение?**
   Мы наблюдаем F1: 10-shot=0.724 → 20-shot=0.819 → 50-shot=0.730.
   Анализ показывает val/test gap при threshold calibration:
   +0.073 на 20-shot vs +0.218 на 50-shot. Это известное ограничение
   TunableDecision при CV на малых данных?

4. **Какой embedder использован в Table 3?**
   Мы фиксировали `multilingual-e5-large-instruct` явно через
   `EmbedderConfig`. При запуске `classic-light` без фиксации
   embedder оптимизируется автоматически и может выбрать другую
   модель (по Table 5 в статье лучший — `stella_en_400M_v5`).
   Был ли embedder зафиксирован в экспериментах Table 3?

## Дальнейшие шаги

- **Оценка на `clinc/clinc_oos` (full, 100 OOS train)** — запустить
  AutoIntent и все бейзлайны на стандартном бенчмарк-датасете.
  Тест-сплит тот же, меняется только количество OOS в train (100 vs 250).
  Это даст числа сравнимые с ADB/DETER и позволит оценить влияние
  конфигурации датасета на результаты.
- **Поиск SOTA в нашем протоколе** через Papers with Code leaderboard
  на CLINC150 — методы с фиксированными 150 интентами и отдельным
  OOS-классом.
- **Сравнение в протоколе TEXTOIR** (25%/50%/75% известных интентов)
  для прямого соотнесения с ADB, DA-ADB, DETER.
- **HYP-001** (per-intent threshold calibration) — отложена,
  актуальна при наличии большего числа примеров на интент.

## Ссылки

- [CLINC150](https://aclanthology.org/D19-1131/) — Larson et al., EMNLP 2019
- [AutoIntent](https://arxiv.org/abs/2509.21138) — Golubev et al., EMNLP 2025
- [AutoIntent GitHub](https://github.com/deeppavlov/AutoIntent)
- [ADB](https://arxiv.org/abs/2012.10209) — Zhang et al., AAAI 2021
- [DA-ADB](https://arxiv.org/abs/2203.05687) — Zhang et al., TASLP 2023
- [DETER](https://arxiv.org/abs/2405.19967) — Rashwan et al., LREC-COLING 2024
