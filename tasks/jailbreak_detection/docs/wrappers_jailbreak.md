# AutoML-обёртки для Jailbreak Detection

> Описывает ДВЕ реализации:
> - **E5-обёртки** (`run_framework_benchmarks.py`) — общие e5-эмбеддинги (1024D). ОСНОВНАЯ.
> - **baseline** (`run_automl_baselines.py`) — TF-IDF+SVD (128D). Будет переделана «из коробки».
> Числа — из metrics_summary_agg.csv.

---

## 1. Две реализации

| Аспект | E5-обёртки (основная) | baseline (переделывается) |
|--------|------------------------|----------------------------|
| Текст→признаки | e5-large-instruct (1024D), общий кэш | TF-IDF+SVD (128D) sklearn / AG n-gram |
| Родная обработка текста фреймворка | НЕ используется (только numeric) | частично (AG), H2O/LAMA на внешнем TF-IDF |
| Decision rule | predict_proba + (scores>=0.5) | смешанная (см. §Противоречия) |
| Статус метрик | валидны (кроме H2O) | удалены, перепрогон «из коробки» запланирован |

> baseline в текущем виде НЕ «из коробки» (AG на тексте, H2O/LAMA на внешнем TF-IDF, разные
> decision rules) — поэтому метрики удалены. Ниже архитектура «из коробки» приведена как
> ЦЕЛЕВОЕ состояние фреймворков, по которому baseline будет переделан.

---

## 2. Архитектура фреймворков «из коробки» (полный состав)

То, что каждый фреймворк делает по умолчанию (search space, ансамбль, текст). Это эталон для
«из коробки» сравнения; в E5-обёртках часть этого намеренно обойдена (только numeric-вход).

### AutoGluon (TabularPredictor)
- Модели по дефолту: GBM (LightGBM), CAT (CatBoost), XGB, RF, XT (ExtraTrees), KNN, LR,
  NN_TORCH, FASTAI.
- Текст «из коробки»: AutoMLPipelineFeatureGenerator (n-gram + текстовая статистика), НЕ
  fine-tuning трансформера, НЕ TextPredictor.
- Ансамбль: WeightedEnsemble (L2). Bagging/Stacking — включаются параметрами (по дефолту в
  нашей конфигурации не включены).
- eval_metric по дефолту для binary: **accuracy**.

### H2O (H2OAutoML)
- Алгоритмы по дефолту: GLM, GBM, XGBoost, DRF, XRT (ExtraTrees), DeepLearning, StackedEnsemble.
- На Apple Silicon (ARM): XGBoost недоступен (нет нативных бинарников) → набор вырожден.
- Текст «из коробки»: H2OWord2vecEstimator (Word2Vec).
- Ансамбль: StackedEnsemble (All + Best-of-Family).
- sort_metric по дефолту: AUC-подобная (для binary).

### LightAutoML (LAMA)
- TabularAutoML по дефолту (numeric): lgb, lgb_tuned, cb, cb_tuned, linear_l2.
- Текст «из коробки»: TabularNLPAutoML (FastText/трансформер-эмбеддинги) — отдельный класс,
  НЕ TabularAutoML.
- Ансамбль: WeightedBlender (blending).
- Метрика: дефолт Task("binary") → logloss.

> Эти три набора намеренно РАЗНЫЕ — это часть поведения «из коробки», не приводим к единому.

> **Лидерборды НЕ сохраняются** ни у одной AutoML-обёртки (нет `*_models.json` в results).
> Поэтому конкретный состав реально обученных моделей и их вклад в ансамбль из артефактов
> установить нельзя — это технический долг (добавить сохранение leaderboard).

---

## 3. E5-обёртки: общий каркас

```
run_framework_benchmarks.py → experiment_runner.run_grid → run_single:
  load → get_or_compute_embeddings (кэш .npy) → create_wrapper → fit(precomputed)
       → predict_proba → (scores >= 0.5) → evaluate_jailbreak → append metrics
```

Все три фреймворка получают ОДНИ И ТЕ ЖЕ кэшированные e5-эмбеддинги (1024D, L2-norm) как
numeric-фичи. Текстовая колонка не подаётся; родная обработка текста (Word2Vec/FastText/
AG-text) НЕ используется → сравнение «движков» на равных фичах.

### Базовый класс BinaryFrameworkWrapper
```python
positive_label = 1          # jailbreak
default_threshold = 0.5
def predict(...): return (scores >= threshold).astype(int)   # base_binary.py:175
def calibrate_threshold(...): ...   # ОПЦИЯ, по умолчанию НЕ вызывается
```
Защита от вырожденных скоров: при std(scores) < 1e-6 → RuntimeError.

---

## 4. E5-обёртки: дефолт → наша реализация → что обучено

### AutoGluon (E5)
| Уровень | Значение |
|---------|----------|
| Дефолт фреймворка | модели GBM/CAT/XGB/RF/XT/KNN/LR/NN_TORCH/FASTAI; eval_metric=accuracy; WeightedEnsemble |
| Наша реализация | вход = e5 (numeric), text не подаётся; eval_metric НЕ задан → accuracy; FASTAI НЕ исключён; time_limit=600; random_state=seed |
| Что обучено | не установить — leaderboard не сохраняется (см. §2, техдолг) |

calibrate_decision_threshold при accuracy ОТКЛЮЧЕНА → финальный порог = наш 0.5 (base_binary),
не внутренняя калибровка AG.

### H2O (E5)
| Уровень | Значение |
|---------|----------|
| Дефолт фреймворка | GLM/GBM/XGB/DRF/XRT/DeepLearning/StackedEnsemble; AUC; StackedEnsemble вкл. |
| Наша реализация | вход = e5; sort_metric="AUC" (задан явно); **StackedEnsemble ОТКЛЮЧЁН** (exclude_algos); max_models=5; seed=seed |
| Что обучено | НЕВАЛИДНО на ARM (нет XGBoost) → перепрогон на x86/Windows |

Отклонения (вернуть к дефолту при перепрогоне на Windows):
- StackedEnsemble отключён (`exclude_algos=["StackedEnsemble"]`) → включить.
- sort_metric="AUC" задан явно → вернуть к дефолтной метрике отбора H2O (не задавать).

### LAMA (E5)
| Уровень | Значение |
|---------|----------|
| Дефолт фреймворка | lgb/lgb_tuned/cb/cb_tuned/linear_l2; logloss; WeightedBlender |
| Наша реализация | вход = e5; Task("binary")→logloss (дефолт); timeout=600; cpu_limit=1; random_state=seed; timing_params НЕ задан |
| Что обучено | не установить — leaderboard не сохраняется (см. §2, техдолг) |

---

## 5. Decision rule (E5) — единый

Все три: `predict_proba` → `(scores >= 0.5)` (base_binary.py:175). positive_label=1
(base_binary.py:32), default_threshold=0.5 (base_binary.py:37). При тае P=0.5 → **jailbreak (1)**.

**Подбор порога не выполняется.** Ко всем трём применяется фиксированный порог 0.5 (наш, из
base_binary). Базовый класс содержит опциональный метод `calibrate_threshold()`, но он в
run_single НЕ вызывается. Это НЕ про отключение какой-либо встроенной калибровки фреймворка —
мы просто берём вероятности (predict_proba) и режем по 0.5 сами, одинаково для всех.

(Отличие от AutoIntent: argmax при тае → safe. См. autointent_jailbreak.md и findings §2.)

---

## 6. seed (E5) — варьируется у всех

| Обёртка | Передача |
|---------|----------|
| AutoGluon | learner_kwargs={"random_state": seed} |
| H2O | H2OAutoML(seed=seed) |
| LAMA | reader_params={"random_state": seed} |

Внутренний seed варьируется по args (42/123/456) — равные условия. (В OOS фиксировали 42.)

---

## 7. Бюджет (E5)

| Фреймворк | Механизм | Значение |
|-----------|----------|----------|
| AutoGluon | time_limit | 600 |
| H2O | max_models | 5 (max_runtime не задан) |
| LAMA | timeout | 600 (timing_params не задан) |

train_sec/fit_sec НЕ записываются → фактический бюджет из metrics не проверить. Бюджеты по
разным механизмам (время vs число моделей) — в лоб не сравнивать. 

---

## 8. Результаты (Full Train, 3 сида) — валидные E5

| Модель | F1 | Recall | ORR | ROC-AUC |
|--------|-----|--------|-----|---------|
| AutoGluon | 0.8746±0.0221 | 0.8082±0.0379 | 0.3746±0.0310 | 0.8065±0.0141 |
| LAMA | 0.8868±0.0015 | 0.8280±0.0023 | 0.3746±0.0073 | 0.8149±0.0020 |

AutoGluon и LAMA — валидны. Невалиден ТОЛЬКО H2O (ARM, нет XGBoost) → не приводится,
перепрогон на Windows.
Сравнение с AutoIntent (KNN/Linear head) — в findings_jailbreak.md.

---

## 9. Потоки данных и файлы

```
Train: wildjailbreak_full100k_seed{S}.json (50k safe + 50k jailbreak) / train_shot{N}_seed{S}.json
Eval:  test.json + wildjailbreak_eval_binary.jsonl → 2210 (data_type: adversarial_harmful/...)
Кэш:   data/processed/embeddings_cache/{model}_{split}.npy
src/framework_wrappers/{base_binary,autogluon_wrapper,h2o_wrapper,lama_wrapper}.py
results/metrics.json (+ extra.scores для восстановления ROC)
```

CLI: `--frameworks autogluon h2o lama --n-shots 10 20 50 --seeds 42 123 456 --run-full
--run-fewshot --skip-existing --continue-on-error`.

---

## 10. Сводное сравнение (E5)

| Обёртка | Бюджет | Метрика отбора | Ансамбль | Отклонения | Стабильность F1 |
|---------|--------|----------------|----------|------------|------------------|
| AutoGluon | time_limit=600 | accuracy (дефолт) | WeightedEnsemble | — | std 0.022 |
| H2O | max_models=5 | AUC (задан — отклонение) | — (Stacked откл.) | StackedEnsemble откл.; sort_metric задан; ARM-невалид | N/A |
| LAMA | timeout=600 | logloss (дефолт) | WeightedBlender | timing_params не задан | std 0.001 |

Общее для всех трёх: вход — e5 (numeric, родная текст-обработка не используется); порог 0.5 без
калибровки; seed варьируется.

---
