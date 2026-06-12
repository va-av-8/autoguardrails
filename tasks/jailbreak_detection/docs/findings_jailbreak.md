# Находки по Jailbreak Detection

## Сводка аудита метрик

**Дата аудита**: 2026-06-12
**Исходных записей**: 111
**После очистки**: 75 записей
**Бэкап**: `results/metrics.json.bak_20260612_205042`

### Удалённые записи (36)

| Категория | Кол-во | Причина |
|-----------|--------|---------|
| H2O (E5 + baseline) | 12 | Apple Silicon ARM — XGBoost недоступен, модели вырождены |
| AutoML baseline (autogluon/lama) | 24 | Kaggle, без сохранённых артефактов; режим не «из коробки» — переделка запланирована |

### Валидность оставшихся записей

| Группа | Кол-во | Статус | Примечание |
|--------|--------|--------|------------|
| autogluon (E5) | 12 | Валидно | extra.scores сохранены |
| lama (E5) | 12 | Валидно | extra.scores сохранены |
| autointent classic-light (KNN head) | 12 | Валидно | runs + eval_scores |
| autointent classic-light_qp (Linear head) | 12 | Валидно | runs + eval_scores |
| autointent classic-medium | 9 | Справочно (Kaggle) | помечены в metrics.json; в csv пометка пока не выводится |
| autointent nn-medium | 9 | Справочно (Kaggle) | другая среда (не M1), перепрогон в планах |
| autointent zero-shot-encoders | 9 | Справочно (Kaggle) | без обучения; перепрогон в планах |

> **Kaggle-справочные (27 записей)** нельзя сравнивать в лоб с локальными M1-прогонами:
> другая среда. Только few-shot, full-режима нет, roc_auc не сохранён.

---

## Сводка метрик (валидные локальные)

### Full Train (3 сида: 42, 123, 456)

В full-режиме есть 4 модели. Два варианта AutoIntent classic-light различаются выбранной
головой (scorer), что отражено в названии:
- **classic-light (KNN head)** — HPO выбрал KNN scorer (прогон без query_prompt).
- **classic-light_qp (Linear head)** — прогон с флагом query_prompt; HPO выбрал Linear scorer
  (query_prompt сломал KNN — см. autointent_jailbreak.md). По сути это linear-голова, а не
  «AutoIntent с инструкцией».

| Модель | F1 | Recall | ORR | ROC-AUC |
|--------|-----|--------|-----|---------|
| AutoGluon | 0.8746±0.0221 | 0.8082±0.0379 | 0.3746±0.0310 | 0.8065±0.0141 |
| LAMA | 0.8868±0.0015 | 0.8280±0.0023 | 0.3746±0.0073 | 0.8149±0.0020 |
| AutoIntent classic-light (KNN head) | 0.9083±0.0094 | 0.8863±0.0197 | 0.6206±0.0324 | 0.7569±0.0179 |
| AutoIntent classic-light_qp (Linear head) | 0.8599±0.0020 | 0.7830±0.0035 | 0.3635±0.0099 | 0.7889±0.0038 |

**Чтение для guardrail (ORR = доля заблокированных safe):**
- KNN head даёт высокий F1 (0.908), но ORR 0.62 — блокирует ~62% safe-запросов.
- Linear head (qp) даёт ORR 0.36 — на уровне AutoML, при F1 0.86.
- AutoML (AG/LAMA) — ORR ~0.37 при F1 0.87–0.89.
- Все модели на этих фичах упираются в схожий потолок ранжирования (ROC ~0.76–0.81);
  различия ORR — это разные рабочие точки, не разное качество (см. nb15 ROC-сравнение).

### Few-shot пресеты AutoIntent (Kaggle, справочно)

Только few-shot, roc_auc не сохранён. НЕ сравнивать в лоб с локальными.

| Пресет | Лучший режим | F1 | Recall | ORR |
|--------|--------------|-----|--------|-----|
| classic-medium | 20shot | 0.7960±0.0625 | 0.7007±0.1084 | 0.5048±0.2089 |
| nn-medium | 20shot | 0.7926±0.1731 | 0.7365±0.2690 | 0.7540±0.2526 |
| zero-shot-encoders | (без обучения) | 0.9133±0.0000 | 0.9080±0.0000 | 0.7667±0.0000 |

> zero-shot-encoders не обучается — метрики идентичны по сидам (std=0, тест один).
> Высокий F1 при ORR 0.77 — блокирует большинство safe, непригоден для guardrail как есть.
> nn-medium крайне нестабилен (std до 0.27); на 10shot почти не работает (F1 0.23).

---

## Ключевые находки

### 1. Бинарная классификация (jailbreak), не OOS

Детекция jailbreak — бинарная (safe=0, jailbreak=1). Оба класса участвуют в обучении.
(В OOS-задаче примеры OOS также идут в обучение — это НЕ отличие jailbreak от OOS.)

Верификация:
- `base_binary.py:32`: `positive_label: int = 1`
- `base_binary.py:175`: `(scores >= threshold).astype(int)`

### 2. Поведение на границе (ties) — РАЗНОЕ у AutoIntent и AutoML

Это ключевое различие рабочих точек, влияет на ORR.

- **AutoML E5** (autogluon/lama): `(scores >= 0.5)` → при ровно P=0.5 → **jailbreak (1)**.
  (`base_binary.py:175`)
- **AutoIntent**: decision-узел argmax → при тае [0.5, 0.5] → **safe (0)**.

Следствие: на тех же скорах AutoIntent относит граничные примеры в safe, AutoML — в jailbreak.
KNN-скоры дискретны (кратны 1/k), поэтому тай P=0.5 у AutoIntent KNN частый. Замер (nb15):
расхождение argmax vs ≥0.5 для AutoIntent KNN — до ~14 п.п. ORR. У моделей с непрерывными
скорами (linear, AutoML) ties≈0, эффект отсутствует.

### 3. eval_metric в AutoGluon = accuracy (не roc_auc)

- eval_metric не задан → дефолт для binary = **accuracy** (дока AutoGluon 1.2+).
- При accuracy внутренняя `calibrate_decision_threshold` ОТКЛЮЧЕНА.
- Финальный порог = наш 0.5 из base_binary, не внутренняя калибровка AG.
- `autogluon_wrapper.py:166-172` (eval_metric не указан).

### 4. H2O на Apple Silicon — невалиден

H2O на ARM не строит XGBoost → модели вырождены. Все H2O-записи удалены.
**Перепрогон на x86/Windows** (заодно включить StackedEnsemble, решить FASTAI/timing).

### 5. query_prompt у AutoIntent не доходит до скоров

Флаг query_prompt задаёт префикс только для query-эмбеддингов; passage_prompt и
classification_prompt = null.
- В qp-прогонах query_prompt ломает KNN (train без префикса / eval с префиксом → асимметрия),
  поэтому HPO выбирает **Linear** scorer для большинства qp-записей.
- Linear использует classification_prompt (= null) → префикс игнорируется.
- Итог: instruction-префикс фактически НЕ повлиял на финальные скоры. qp-записи — это
  linear-голова, не «AutoIntent с инструкцией».

### 6. seed варьируется по args у всех (в отличие от OOS)

| Обёртка | Передача seed |
|---------|---------------|
| AutoGluon | `learner_kwargs={"random_state": seed}` |
| LAMA | `reader_params={"random_state": seed}` |
| H2O | `H2OAutoML(seed=seed)` |
| AutoIntent | `Pipeline.from_preset(..., seed=args.seed)` |

В jailbreak внутренний seed варьируется (42/123/456) у всех — равные условия.
(В OOS внутренний seed фиксировали = 42; это различие между задачами.)

### 7. Decision-узел AutoIntent — argmax выбран HPO, не зашитый порог

В прогонах decision-узел выбрал **argmax** (из search space {argmax, threshold, tunable,
jinoos}, оптимизация decision_accuracy). Это результат HPO, а не архитектурное «отсутствие
калибровки» — узел МОЖЕТ калибровать порог, но на этих данных выиграл argmax.

---

## Запланировано

- **H2O** — перепрогон на x86/Windows (XGBoost; включить StackedEnsemble; решить FASTAI/timing).
- **AutoML baseline** — переделать строго «из коробки» (родная текстовая обработка каждого
  фреймворка: AG n-gram, H2O Word2Vec, LAMA TabularNLPAutoML), только CPU.
- **AutoIntent Kaggle-пресеты** (medium/nn/zero) — перепрогнать локально на M1.

## Технический долг

| Проблема | Файл | Действие |
|----------|------|----------|
| train_sec/fit_sec не пишутся | wrappers, run_autointent | бюджет прогонов из metrics не проверить |
| roc_auc не сохраняется в metrics | src/metrics.py | восстанавливается из scores (nb15) |
| AutoML leaderboard не сохраняется | *_wrapper | добавить *_models.json |
| Kaggle-пометка не выводится в csv | скрипт генерации csv | поправить (флаг есть в metrics.json) |

---

## Схема metrics.json

```json
{
  "model_name": "autogluon",
  "mode": "full",
  "seed": 42,
  "f1": 0.8746,
  "precision": 0.8895,
  "recall": 0.8082,
  "over_refusal_rate": 0.3746,
  "recall_adversarial_harmful": 0.8082,
  "timestamp": "2026-05-11T22:17:56+00:00",
  "extra": {
    "embedder_hf_model": "intfloat/multilingual-e5-large-instruct",
    "embedder_fixed": true,
    "scores": [...],
    "eval_counts": {"tp": 1616, "fp": 79, "fn": 384, "tn": 131}
  }
}
```

| Поле | Описание |
|------|----------|
| `f1` | F1 на классе jailbreak |
| `recall` | TPR на jailbreak |
| `over_refusal_rate` | FPR на safe (доля заблокированных безопасных) |
| `precision` | TP/(TP+FP) |
| `extra.scores` | P(jailbreak) по примерам (для восстановления ROC) |
| `extra.eval_counts` | матрица ошибок |
