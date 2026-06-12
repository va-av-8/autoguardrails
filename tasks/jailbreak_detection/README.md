# Jailbreak Detection

## Постановка задачи

Детекция попыток jailbreak на input, до передачи промпта в LLM.
OOS-переформулировка: safe = in-scope интент, jailbreak = OOS-класс.

Задача структурно аналогична OOS-детекции, но данные adversarial по природе —
jailbreak-промпты специально сконструированы, чтобы обходить защиты LLM.

## Датасет

**WildJailbreak** (Jiang et al., NeurIPS 2024)
- HuggingFace: `allenai/wildjailbreak`
- Train: 261 559 примеров
- Eval: 2 210 примеров (используется как test)
- Prompt-колонка: `adversarial` если непустой, иначе `vanilla`
- Бинарная метка: `*_harmful` → jailbreak (1), `*_benign` → safe (0)

**Распределение eval:**
| data_type | count | binary_label |
|-----------|-------|--------------|
| adversarial_harmful | 2000 | jailbreak |
| adversarial_benign | 210 | safe |

Vanilla-примеров в eval нет. Дисбаланс ~10:1 (jailbreak:safe).

**Предупреждение:** датасет содержит примеры тяжёлого вреда, включая CSAM-категорию.
При отображении примеров избегать воспроизведения контента из этих категорий.

## Подготовка данных

### Few-shot протокол
- Balanced sampling: N примеров на класс. n_shots [10, 20, 50], seeds [42, 123, 456].
- Естественное распределение (~10:1) сохранено только в test.
- `python scripts/prepare_data.py`

### Full-train подвыборки (100K)
- Стратифицированные 100K из train: 50/50 safe vs jailbreak, внутри класса сохранена
  пропорция vanilla/adversarial.
- `python scripts/prepare_data.py --full_subset`
- Выход: `wildjailbreak_full100k_seed{42,123,456}.json`

## Метрики

Все в `src/metrics.py`, `oos_label=1`. Агрегатор `evaluate_jailbreak(...)`.

| Метрика | Описание |
|---------|----------|
| **F1** | F1 на jailbreak-классе |
| **Recall** | TPR на jailbreak |
| **Over-refusal Rate (ORR)** | FPR на safe (доля заблокированных безопасных) |
| **ROC-AUC** | ранжирование (порого-независимо) |
| **recall_adversarial_harmful** | recall на adversarial-подмножестве |

---

## Ключевые находки

Сводка расследования (детали — в `findings_jailbreak.md`, `autointent_jailbreak.md`,
`wrappers_jailbreak.md`; диагностики — в ноутбуках 06–15).

**Потолок разделимости.** На общих e5-эмбеддингах все модели упираются в близкий потолок
ранжирования (ROC-AUC ~0.76–0.81). Вне эмбеддинга извлекаемого сигнала нет: символьная
обфускация и структурные обёртки (role-play и пр.) распределены симметрично между safe и
harmful, e5 их слабо разделяет (nb14). Это потолок представления, а не конкретной модели.

**Бустинг закрывает разрыв с AutoML.** Отставание AutoIntent от AutoML-фреймворков на тех же
e5 объясняется тем, что у фреймворков в поиске есть бустинги, а у classic-light — нет.
Одиночный LightGBM на e5 даёт метрики на уровне фреймворков (даже чуть выше). Стекинг сверх
бустинга прироста почти не даёт (+~0.006 ROC-AUC); по ROC-кривым видно, что сдвиг порога
проблему не решает.

**KNN vs Linear голова.** В classic-light HPO выбирает KNN-scorer, который на этой задаче
слабее линейного: хуже ранжирует (ROC 0.76 vs 0.79) и даёт вдвое больший ORR (0.62 vs 0.36).
Это вскрылось при отладке проброса query_prompt: префикс сломал KNN на eval, и HPO выбрал
Linear (см. `autointent_jailbreak.md`, §6). Сам instruction-префикс до скоров не доходит —
остаётся неиспробованным рычагом.

**Препроцессинг текста бесполезен** на WildJailbreak: сомнительные паттерны и обёртки
распределены одинаково по safe/harmful, bag-of-words и нормализация не дают сигнала.

**ORR высок у всех.** Лучший достижимый ORR ~0.36 (Linear head, AutoML) — для guardrail это
всё ещё много (каждый третий безопасный запрос блокируется). Это следствие потолка
разделимости, а не конкретной модели. Баланс F1/ORR не достигается сдвигом порога.

### Метрики (Full Train, 3 сида)

| Модель | F1 | Recall | ORR | ROC-AUC |
|--------|-----|--------|-----|---------|
| AutoGluon (E5) | 0.8746±0.0221 | 0.8082±0.0379 | 0.3746±0.0310 | 0.8065±0.0141 |
| LAMA (E5) | 0.8868±0.0015 | 0.8280±0.0023 | 0.3746±0.0073 | 0.8149±0.0020 |
| AutoIntent classic-light — **KNN head** | 0.9083±0.0094 | 0.8863±0.0197 | 0.6206±0.0324 | 0.7569±0.0179 |
| AutoIntent classic-light — **Linear head** (qp) | 0.8599±0.0020 | 0.7830±0.0035 | 0.3635±0.0099 | 0.7889±0.0038 |

H2O невалиден на Apple Silicon (нет XGBoost) — перепрогон на x86/Windows.
classic-medium / nn-medium / zero-shot — Kaggle-справочные (только few-shot), не сравнивать
в лоб с локальными M1.

### AutoIntent classic-light few-shot (KNN head)

| n_shots | F1 | Recall | ORR |
|---------|-----|--------|-----|
| 10-shot | 0.651±0.133 | 0.519 | 0.467 |
| 20-shot | 0.736±0.145 | 0.629 | 0.462 |
| 50-shot | 0.682±0.119 | 0.550 | 0.417 |

Few-shot нестабилен (высокий std); на 10-shot модели близки к случайному ранжированию.

---

## SOTA-модели (zero-shot, на test)

| Модель | Параметры | Тип | Источник |
|--------|-----------|-----|----------|
| PromptGuard 2 | 86M | BERT-style classifier | LlamaFirewall, arxiv:2505.03574 |
| Qwen3Guard-Gen | 4B / 8B | Generative LLM | arxiv:2510.14276 |

- WildGuard исключён: обучен на данных, включающих WildJailbreak → data leakage.
- Qwen3Guard независимо оценён (arxiv:2511.22047): generalization gap −57 п.п. на novel prompts.

---

## Открытые направления

1. **Починить query_prompt** симметрично (passage/classification prompt) — instruction-префикс
   уникальное преимущество AutoIntent (Табл.1 статьи), пока не задействован.
2. **Склейка с safety-классификатором** (напр. скрытое состояние / скор Qwen3Guard) как внешняя
   фича — план Б, если префикс не поднимет разделимость.
3. **Бустинг-scorer в classic-light.** Разрыв по ROC объясняется отсутствием бустинга в
   classic-light. В AutoIntent бустинг-scorer есть (CatBoostScorer, входит в medium-пресеты) —
   стоит прогнать classic-medium локально или добавить CatBoost-scorer в classic-light.
4. **H2O — перепрогон на x86/Windows** (XGBoost; вернуть StackedEnsemble и дефолтную метрику).
5. **Baseline AutoML — переделать строго «из коробки»** (родная текст-обработка каждого
   фреймворка), CPU.
6. **Стабилизация few-shot** — надежда на более сильные эмбеддинги или стекинг поверх них.

---

## Ноутбуки

| Ноутбук | Содержание |
|---------|------------|
| `01_eda.ipynb` | EDA WildJailbreak |
| `04_jailbreak_assymetric_cost_hypothesis.ipynb` | Asymmetric cost (опровергнута) |
| `06_scores_diagnostics.ipynb` | Диагностика скоров |
| `07_fulltrain_diagnostics.ipynb` | Диагностика full-train |
| `08_embedding_separability.ipynb` | Разделимость в эмбеддингах |
| `09_adversarial_ceiling.ipynb` | Потолок на adversarial |
| `10_knn_vs_linear_scorer.ipynb` | KNN vs Linear scorer |
| `11_stacking_scorers.ipynb` | Стекинг scorer'ов |
| `12_closing_gap_heads.ipynb` | Закрытие разрыва (головы) |
| `13_orr_tradeoff.ipynb` | Trade-off ORR |
| `14_attack_vectors.ipynb` | Векторы атак |
| `15_roc_comparison.ipynb` | Сводное ROC-сравнение всех участников |

## Документация

| Файл | Содержание |
|------|------------|
| `autointent_jailbreak.md` | Обёртка AutoIntent: узлы, HPO, query_prompt, seed |
| `wrappers_jailbreak.md` | AutoML-обёртки E5: архитектура «из коробки», decision rule, бюджеты |
| `findings_jailbreak.md` | Карта валидности метрик, ties, техдолг |

## Ограничения

- **Eval только adversarial:** vanilla_harmful/benign отсутствуют → не проверить vanilla vs
  adversarial gap.
- **Дисбаланс ~10:1** влияет на интерпретацию precision/ORR.
- **train_sec/roc_auc** не пишутся в metrics.json (ROC восстанавливается из scores).
- **Лидерборды AutoML** не сохраняются — состав обученных моделей из артефактов не установить.

## Ссылки

- [WildJailbreak](https://arxiv.org/abs/2406.18510) — Jiang et al., NeurIPS 2024
- [WildGuard](https://arxiv.org/abs/2406.18495) — Han et al., NeurIPS 2024
- [AutoIntent](https://arxiv.org/abs/2509.21138) — Golubev et al., EMNLP 2025
- [PromptGuard 2](https://arxiv.org/abs/2505.03574) — Meta, 2025
- [Qwen3Guard](https://arxiv.org/abs/2510.14276) — Alibaba, 2025
