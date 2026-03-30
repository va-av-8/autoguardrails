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

Метрики выбраны под guardrail-контекст, где пропустить нарушение
дороже, чем ложно заблокировать легитимный запрос.

| Метрика | Описание | Почему важна |
|---|---|---|
| **Accuracy** | Точность на всех классах | Стандартная метрика, сравнение с литературой |
| **Recall@FPR=0.05** | TPR при 5% ложных блокировок | Основная рабочая точка для guardrail |
| **Recall@FPR=0.10** | TPR при 10% ложных блокировок | Мягкая рабочая точка для guardrail |
| **AUROC** | Area under ROC curve | Общее качество ранжирования |
| **F1 (OOS)** | F1 на OOS-классе | Сравнение с результатами из статьи AutoIntent |
| **Latency (ms)** | Время инференса на 1 запрос | Guardrail в критическом пути |

**Референсные значения из статьи AutoIntent (Table 3, CLINC150):**
- AutoIntent: in-domain accuracy = 96.13, OOS F1 = 76.79
- AutoGluon (OOS как доп. класс): OOS F1 = 48.53
- H2O (OOS как доп. класс): OOS F1 = 40.69

Примечание: сплит (easy/medium/hard) в статье не указан — уточнить
при воспроизведении.

## Эксперименты

### Шаг 1. Бейзлайны (нижняя граница)
Простейшие решения без специальной поддержки OOS.

**Бейзлайн A:** TF-IDF + LogReg, OOS как доп. класс
**Бейзлайн B:** Cosine similarity threshold поверх sentence embeddings

Конфиги: `configs/baseline_tfidf.yaml`, `configs/baseline_cosine.yaml`
Скрипт: `scripts/run_baseline.py`

### Шаг 2. SOTA (верхняя граница)
Специализированный метод OOS-детекции.

**Метод:** DA-ADB (Distance-based Adaptive Boundary, Zhang et al., 2021)
Лучший результат на CLINC150 hard среди открытых методов: F1 ~85–88%.

Конфиг: `configs/dadb.yaml`
Скрипт: `scripts/run_dadb.py`

### Шаг 3. AutoIntent — few-shot режим
Основной эксперимент: реалистичный production-сценарий.

Запускаем AutoIntent при n ∈ {10, 20, 50} примеров на intent.
Вопрос: при каком n AutoIntent достигает приемлемого качества
(Recall ≥ 0.90 @ FPR=0.10)?

Конфиг: `configs/autointent_fewshot.yaml`
Скрипт: `scripts/run_autointent.py --mode fewshot --n_shots 10 20 50`

### Шаг 4. AutoIntent — full train
Воспроизведение результата из статьи (OOS F1 = 76.79) на full train.
Служит точкой сравнения и проверкой воспроизводимости.

Конфиг: `configs/autointent_full.yaml`
Скрипт: `scripts/run_autointent.py --mode full`

### Шаг 5. Гипотеза
**HYP-001:** Per-intent threshold calibration улучшает Recall@FPR на hard OOS
без потери precision.

Мотивация: разные intent-кластеры имеют разную плотность в
embedding-пространстве — глобальный порог неоптимален.

Скрипт: `scripts/run_autointent.py --mode fewshot --hypothesis per_intent_threshold`

## Результаты

| Модель | Режим | Accuracy | Recall@FPR=0.05 | Recall@FPR=0.10 | AUROC | F1 (OOS) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | full train | — | — | — | — | — | — |
| Cosine threshold | full train | — | — | — | — | — | — |
| DA-ADB | full train | — | — | — | — | — | — |
| AutoIntent | 10-shot | — | — | — | — | — | — |
| AutoIntent | 20-shot | — | — | — | — | — | — |
| AutoIntent | 50-shot | — | — | — | — | — | — |
| AutoIntent | full train | — | — | — | — | — | — |
| AutoIntent + HYP-001 | 50-shot | — | — | — | — | — | — |

## Ссылки

- [CLINC150 paper](https://aclanthology.org/D19-1131/)
- [AutoIntent paper](https://arxiv.org/abs/2509.21138)
- [AutoIntent GitHub](https://github.com/deeppavlov/AutoIntent)
- [ADB paper (AAAI 2021)](https://hanleizhang.github.io/publication/ADB)
- [DA-ADB paper (IEEE TASLP 2023)](https://arxiv.org/abs/2203.05823)
- [ADB / DA-ADB GitHub](https://github.com/thuiar/Adaptive-Decision-Boundary)
