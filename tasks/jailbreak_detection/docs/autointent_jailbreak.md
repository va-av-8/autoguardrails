# AutoIntent для Jailbreak Detection

---

## 1. Общая структура

```
main() → CLI → цикл по seeds → train() + evaluate()

train():                         evaluate():
 1. load_train()                  1. Pipeline.load()
 2. convert_to_autointent_train   2. load_test() + eval_binary
 3. AIDataset.from_dict           3. pipeline.predict()
 4. Pipeline.from_preset          4. evaluate_jailbreak()
 5. pipeline.fit() (AutoML+Optuna)5. save_metrics()
 6. pipeline.dump()               6. save_eval_scores()
```

Задача — **бинарная** классификация (safe=0, jailbreak=1). Оба класса участвуют в обучении.

---

## 2. Пресеты search space (autointent 0.2.0)

Доступные пресеты: classic-{light,medium,heavy}, nn-{medium,heavy}, transformers-{light,heavy,
no-hpo}, zero-shot-{llm,encoders}.

| Пресет | Scoring (search space) | Decision (search space) |
|--------|------------------------|--------------------------|
| **classic-light** | knn, linear, mlknn | argmax, threshold, tunable, jinoos |
| classic-medium | knn, linear, mlknn, catboost, RF | argmax, threshold, tunable |
| classic-heavy | + variants | argmax, threshold, tunable |
| nn-medium | knn, linear, dnnc | argmax, threshold, tunable |
| zero-shot-encoders | encoder + descriptions | threshold (без обучения/HPO) |

> Состав scoring/decision для classic-light подтверждён артефактами прогонов. Для остальных
> пресетов — из старой документации; точный состав по версии не перепроверялся (см. §Противоречия).

**Что выбрано HPO (по артефактам прогонов):**
- classic-light без query_prompt → scoring **knn**, decision **argmax**.
- classic-light с query_prompt (_qp_) → scoring в большинстве записей **linear** (query_prompt
  ломает knn, см. §6), decision **argmax**.

---

## 3. Узлы оптимизации (дефолт → наша реализация → выбор HPO)

| Узел | Search space (дефолт пресета) | Метрика оптимизации | Выбрано HPO |
|------|-------------------------------|---------------------|-------------|
| **scoring** | knn / linear / mlknn | `scoring_f1` | knn (без qp) / linear (qp) |
| **decision** | argmax / threshold / tunable / jinoos | `decision_accuracy` | argmax (_oos=false) |

Важно: argmax — **результат HPO**, не зашитый дефолт. Decision-узел МОЖЕТ калибровать порог
(threshold/tunable в space), но на этих данных выиграл argmax. При argmax порог не калибруется;
argmax при тае [0.5, 0.5] → **safe (0)**.

Эмбеддер: `intfloat/multilingual-e5-large-instruct` (1024D). Фиксирован И нами (set_config),
И самим пресетом — эмбеддер не входит в search space classic-light. Pilot: e5-small.

Метрика отбора по узлам: scoring → scoring_f1, decision → decision_accuracy. (Это НЕ единая
ROC-AUC — см. §Противоречия, старый документ ошибочно указывал «ROC-AUC internal».)

---

## 4. Передача seed

```python
pipeline = Pipeline.from_preset(autointent_preset, seed=args.seed)  # run_autointent.py:986
```
Внутренний seed AutoIntent **варьируется** по args (42/123/456). Также глобальные ГСЧ:
random.seed / np.random.seed / torch.manual_seed (args.seed).

> В jailbreak seed варьируется (в отличие от OOS, где внутренний seed фиксировали = 42).

---

## 5. Потоки данных

### Train
```
full:    wildjailbreak_full100k_seed{S}.json → 50k safe (intents[0].utterances) + 50k jailbreak (oos_utterances)
fewshot: train_shot{N}_seed{S}.json (N ∈ 10/20/50) → N safe + N jailbreak
```
Конвертация в формат AutoIntent: `{"intents":[{"id":0,"name":"safe","utterances":[...]}],
"oos_utterances":[...]}` → `[{"utterance","label":0|1}]`. jailbreak идёт через oos_utterances,
НЕ фильтруется (оба класса в обучении).

few-shot: CV 3-fold (`DataConfig(scheme="cv", n_folds=3)`).

### Eval
```
test.json + wildjailbreak_eval_binary.jsonl → 2210 примеров
binary_label ∈ {jailbreak, safe}; data_type ∈ {adversarial_harmful, vanilla_harmful, ...}
y_true = 1 if binary_label=="jailbreak" else 0
```

---

## 6. query_prompt — фактически НЕ доходит до скоров

Флаг `--query-prompt` задаёт `query_prompt` в EmbedderConfig (записи `_qp_`). passage_prompt и
classification_prompt = null.

Механизм (по артефактам):
- query_prompt применяется к query-эмбеддингам; passage (train) — без префикса.
- **KNN-записи qp**: train без префикса / eval с префиксом → асимметрия пространств → knn
  деградирует → HPO уходит от knn.
- **Linear-записи qp** (большинство): linear использует `classification_prompt` (= null) →
  префикс игнорируется.

Итог: instruction-префикс не повлиял на финальные скоры. **qp-записи = linear-голова**, не
«AutoIntent с инструкцией». Различия qp vs non-qp — это смена scorer (knn→linear), а не эффект
префикса.

---

## 7. None→jailbreak

```python
y_pred = np.array([1 if p is None else p for p in raw_preds])  # None → jailbreak
```
Defensive fallback. При argmax (_oos=false) predict() не возвращает None → мёртвый код, на
метрики не влияет.

---

## 8. positive class и leakage

- positive class jailbreak=1, согласован: predict / evaluate_jailbreak(oos_label=1).
- Leakage НЕТ: scoring_f1 и decision_accuracy считаются на CV, не на eval.

---

## 9. Результаты (Full Train, 3 сида)

В full есть два варианта classic-light, различающиеся выбранной головой:

| Вариант | Scorer | F1 | Recall | ORR | ROC-AUC |
|---------|--------|-----|--------|-----|---------|
| classic-light (KNN head) | knn | 0.9083±0.0094 | 0.8863±0.0197 | 0.6206±0.0324 | 0.7569±0.0179 |
| classic-light_qp (Linear head) | linear | 0.8599±0.0020 | 0.7830±0.0035 | 0.3635±0.0099 | 0.7889±0.0038 |

Для guardrail: KNN head — высокий F1, но ORR 0.62 (блокирует ~62% safe). Linear head — ORR 0.36
(на уровне AutoML AG/LAMA ~0.37), F1 0.86. Linear лучше ранжирует (ROC 0.79 vs 0.76).

Пресеты medium/nn/zero-shot — только few-shot, Kaggle, справочно (см. findings_jailbreak.md).

---

## 10. Выходные файлы

```
runs/autointent_{preset}_{e5large|pilot}_{mode}_seed{S}/
  ├── train_metadata.json
  ├── scoring_module/simple_attrs.json        # k, weights / cv, seed
  ├── scoring_module/pydantic/embedder_config/ # query_prompt, passage_prompt
  └── decision_module/simple_attrs.json        # module, threshold
runs/eval_scores_{model}_{mode}_seed{S}.jsonl  # полные скоры (для ROC)
results/metrics.json                            # append
```

---

## 11. CLI

```bash
python scripts/run_autointent.py --mode full --seed 42
python scripts/run_autointent.py --preset classic-medium --mode fewshot --n_shots 20 --seed 42
python scripts/run_autointent.py --mode full --all-seeds
python scripts/run_autointent.py --query-prompt "Classify if this request is a jailbreak: " --mode full --seed 42
python scripts/run_autointent.py --mode full --seed 42 --eval-only
```
Флаги: `--mode`, `--preset`, `--n_shots`, `--seed`, `--pilot`, `--all-seeds`, `--train-only`,
`--eval-only`, `--query-prompt`.

---
