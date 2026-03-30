# Out-of-Scope Detection

## Исследовательский вопрос

Может ли AutoIntent, настроенный на небольшом количестве примеров
(реалистичный production-сценарий), обеспечить качество OOS-детекции,
сопоставимое со специализированными SOTA-методами?

## Мотивация

При развёртывании агентной системы разработчик располагает описанием
системы и ограниченным набором примеров (10–50 на intent). Нужен
надёжный OOS-детектор без глубокой ML-экспертизы. AutoIntent как
AutoML-движок — кандидат на роль такого инструмента "из коробки".

## Датасет

**CLINC150** (Larson et al., 2019)
- 150 intent-классов, 22 500 in-scope примеров
- 1 200 OOS примеров в трёх уровнях сложности: easy / medium / hard
- HuggingFace: `clinc_oos` (splits: train / validation / test)
- Hard OOS — семантически близко к in-scope, основной challenge

**Few-shot сэмплинг:**
Из train-сплита сэмплируем n примеров на intent (n ∈ {10, 20, 50}).
OOS-примеры для train берём пропорционально: n_oos = n_intents × n_shots × 0.1
Тест-сплит всегда фиксированный (стандартный CLINC150 test).

**Формат для AutoIntent:**
OOS-примеры передаются без поля `label` (согласно документации AutoIntent).

## Метрики

Схема метрик соответствует академической OOS NLP-литературе.
Основные три метрики позволяют напрямую сравниться с опубликованными
результатами ADB, DETER и AutoIntent.

### Основные (прямое сравнение с литературой)

| Метрика | Описание | Литературный прецедент |
|---|---|---|
| **OOS Recall** | TPR на OOS-классе: доля пойманных нарушений мандата | ADB (AAAI 2021), DETER (2024), AutoIntent (EMNLP 2025) |
| **In-domain Accuracy** | Accuracy только на in-scope примерах | AutoIntent Table 3 (96.13), ADB, большинство OOS-работ |
| **F1 OOS** | F1-score на OOS-классе | AutoIntent Table 3 (76.79), ADB, DETER |

Мотивация выбора OOS Recall как ключевой метрики детекции:
пропустить OOS-запрос (FN) критичнее, чем ложно заблокировать
in-scope (FP): FP вызывает fallback с просьбой переформулировать,
тогда как FN порождает неверный ответ системы.

Примечание: "in-domain accuracy" и "in-scope accuracy" — синонимы
в OOS-литературе. Используем "in-domain" для согласованности
с AutoIntent Table 3.

### Вспомогательные

| Метрика | Описание | Литературный прецедент |
|---|---|---|
| **AUROC** | Area Under ROC Curve | "Intent Detection in the Age of LLMs" (EMNLP Industry 2024) |
| **AU-IOC** | Area Under In-scope/Out-of-scope Characteristic curve | Springer Applied Intelligence (2024) |
| **Latency (ms)** | Время инференса на 1 запрос | Нет прецедента в OOS-литературе; важна для guardrail-аргументации |

AU-IOC специально разработана для OOS-задачи: строит кривую
(X = in-domain accuracy, Y = OOS recall) при изменении порога.
Отражает guardrail trade-off между защитой и usability —
в отличие от AUROC, учитывает качество классификации внутри in-scope.

## Эксперименты

### Шаг 1. Бейзлайны (нижняя граница)
Простейшие решения без специальной поддержки OOS.

**Бейзлайн A:** TF-IDF + LogReg, OOS как доп. класс
**Бейзлайн B:** Cosine similarity threshold поверх sentence embeddings

Конфиги: `configs/baseline_tfidf.yaml`, `configs/baseline_cosine.yaml`
Скрипт: `scripts/run_baseline.py`

### Шаг 2. SOTA (верхняя граница)

**Метод:** DETER (Dual Encoder for Threshold-Based Re-Classification)
Публикация: arxiv.org/abs/2405.19967 (2024)
GitHub: github.com/Hossam-Mohammed-tech/Intent_Classification_OOS

Dual encoder (USE + TSDAE) с синтетическими outliers и threshold re-classification.
Превосходит ADB (AAAI 2021): +13% F1 known, +5% F1 unknown на CLINC150.

Конфиг: `configs/deter.yaml`
Скрипт: `scripts/run_deter.py`

### Шаг 2b. Guardrail Reference (отдельно от основного сравнения)

**Модель:** govtech/stsb-roberta-base-off-topic
Публикация: arxiv.org/abs/2411.12946 (Chua et al., 2024)

Единственная открытая модель, специально обученная под off-topic guardrail для LLM.
Включается как reference по постановке задачи, не как SOTA по метрикам.
Обучена на синтетических данных — результаты на CLINC150 интерпретируются отдельно.

Конфиг: `configs/guardrail_reference.yaml`
Скрипт: `scripts/run_guardrail_reference.py`

### Шаг 3. AutoIntent — few-shot режим
Основной эксперимент: реалистичный production-сценарий.

Запускаем AutoIntent при n ∈ {10, 20, 50} примеров на intent.
Вопрос: при каком n AutoIntent достигает приемлемого качества
(OOS Recall ≥ 0.85, In-domain Accuracy ≥ 0.90)?

Конфиг: `configs/autointent_fewshot.yaml`
Скрипт: `scripts/run_autointent.py --mode fewshot --n_shots 10 20 50`

### Шаг 4. AutoIntent — full train
Воспроизведение результата из статьи (OOS F1 = 76.79) на full train.
Служит точкой сравнения и проверкой воспроизводимости.

Конфиг: `configs/autointent_full.yaml`
Скрипт: `scripts/run_autointent.py --mode full`

### Шаг 5. Гипотеза
**HYP-001:** Per-intent threshold calibration улучшает OOS Recall на hard OOS
без потери In-domain Accuracy.

Мотивация: разные intent-кластеры имеют разную плотность в
embedding-пространстве — глобальный порог неоптимален.

Скрипт: `scripts/run_autointent.py --mode fewshot --hypothesis per_intent_threshold`

## Результаты

| Модель | Режим | OOS Recall | In-domain Acc | F1 OOS | AUROC | AU-IOC | Latency (ms) |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | full train | — | — | — | — | — | — |
| Cosine, bert-base-uncased | full train | — | — | — | — | — | — |
| Cosine, all-MiniLM-L6-v2 | full train | — | — | — | — | — | — |
| DETER | full train | — | — | — | — | — | — |
| AutoIntent | 10-shot | — | — | — | — | — | — |
| AutoIntent | 20-shot | — | — | — | — | — | — |
| AutoIntent | 50-shot | — | — | — | — | — | — |
| AutoIntent | full train | — | — | — | — | — | — |
| AutoIntent + HYP-001 | 50-shot | — | — | — | — | — | — |
| [reference] Guardrail (Chua 2024) | zero-shot | — | — | — | — | — | — |

Референсные значения из статьи AutoIntent (Table 3, CLINC150):
- AutoIntent: In-domain Accuracy = 96.13, F1 OOS = 76.79
- AutoGluon (OOS как доп. класс): F1 OOS = 48.53
- H2O (OOS как доп. класс): F1 OOS = 40.69

Строка [reference] не участвует в основном сравнении —
модель обучена на другом распределении (синтетические данные).

## Ссылки

- [CLINC150 paper](https://aclanthology.org/D19-1131/)
- [AutoIntent paper](https://arxiv.org/abs/2509.21138)
- [AutoIntent GitHub](https://github.com/deeppavlov/AutoIntent)
- [DETER paper](https://arxiv.org/abs/2405.19967)
- [DETER GitHub](https://github.com/Hossam-Mohammed-tech/Intent_Classification_OOS)
- [ADB paper](https://arxiv.org/abs/2012.10209) — предшественник DETER, AAAI 2021
- [Chua et al. 2024](https://arxiv.org/abs/2411.12946) — guardrail reference
- [govtech/stsb-roberta-base-off-topic](https://huggingface.co/govtech/stsb-roberta-base-off-topic)
