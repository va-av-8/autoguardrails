# metrics.json — каталог прогонов

Файл: `tasks/jailbreak_detection/results/metrics.json`

**Формат строки** (как пишет `run_autointent.py` → `save_metrics`, урезанный `extra`):

- Верхний уровень: `model_name`, `mode`, `n_shots`, `seed`, `f1`, `precision`, `recall`, `over_refusal_rate`, `recall_adversarial_harmful`, `timestamp`
- `extra`: `preset`, `embedder`, `embedder_hf_model`, `embedder_fixed`, `pilot`, `model_dir`, `eval_counts`, `decision_module_attrs`, `scores_eval_summary`

Полные логи (prediction_summary, timings, data_summary, …) — в архиве  
`metrics_jailbreak_successful_14_05_full_rows.json`, **не** дублируются в `metrics.json`.

Обновлено: 2026-05-20 15:42 UTC

---

## Блок 1 — Kaggle few-shot grid (14.05.2026)

**Источник при слиянии:** `metrics_jailbreak_successful_14_05_full_rows.json`  
(сводка без nested-логов: `metrics_jailbreak_successful_14_05.json`)

| Что | Значение |
|-----|----------|
| Ноутбук / пайплайн | `notebooks/kaggle_heavy_presets_jailbreak.ipynb` |
| Скрипт | `scripts/run_autointent_logged.py` → `run_autointent.py` |
| Режим | few-shot (`mode`: `10shot` / `20shot` / `50shot`) |
| Shots | 10, 20, 50 |
| Seeds | 42, 123, 456 |
| Eval | `wildjailbreak_eval_binary.jsonl`, n=2210 (adversarial_harmful 2000 + adversarial_benign 210) |
| Флаги | `--no-fix-embedder`, пресеты classic-medium / nn-medium / zero-shot-encoders |

**Строк в блоке:** 27 (3 пресета × 3 shots × 3 seeds)

### classic-medium (9 строк)
AutoIntent search-space `classic-medium`, embedder подбирается AutoML → обычно `intfloat/multilingual-e5-large-instruct`.

### zero-shot-encoders (9 строк)
Zero-shot по описаниям интентов; **все 9 конфигураций дают одинаковые метрики** (shots/seed не влияют на eval).  
Анализ: `notebooks/05_jailbreak_successful_runs_14_05.ipynb`.

### nn-medium (9 строк)
Пресет `nn-medium`; `embedder_hf_model` в логах часто null. Сильная вариативность по seed; осторожно с пиком F1 при ORR=1.

---

## Блок 2 — classic-light (ранние прогоны, до heavy-presets grid)

**Относится к:** `notebooks/kaggle_full_pipeline.ipynb` / standalone Kaggle jailbreak  
**Пресет:** `classic-light`, embedder `intfloat/multilingual-e5-large-instruct`

| Подблок | mode | строк |
|---------|------|-------|
| Few-shot grid | `10shot` / `20shot` / `50shot` × seeds 42/123/456 | 9 |
| Full train | `full`, `n_shots=null` × seeds 42/123/456; train `wildjailbreak_full100k_seed{S}.json` | 3 |

**Строк в блоке:** 12 (уже были в `metrics.json` до слияния 14.05)

---

## Индекс строк в metrics.json

| # | preset | mode | n_shots | seed | F1 | ORR | model_name |
|---|--------|------|---------|------|-----|-----|------------|
| 1 | classic-light | 10shot | 10 | 42 | 0.7572 | 0.9190 | autointent_classic-light_autoembedder |
| 2 | classic-light | 10shot | 10 | 123 | 0.7307 | 0.3619 | autointent_classic-light_autoembedder |
| 3 | classic-light | 10shot | 10 | 456 | 0.7828 | 0.2714 | autointent_classic-light_autoembedder |
| 4 | classic-light | 20shot | 20 | 42 | 0.8671 | 0.7429 | autointent_classic-light_autoembedder |
| 5 | classic-light | 20shot | 20 | 123 | 0.7497 | 0.4190 | autointent_classic-light_autoembedder |
| 6 | classic-light | 20shot | 20 | 456 | 0.7713 | 0.3524 | autointent_classic-light_autoembedder |
| 7 | classic-light | 50shot | 50 | 42 | 0.8189 | 0.6000 | autointent_classic-light_autoembedder |
| 8 | classic-light | 50shot | 50 | 123 | 0.8133 | 0.6762 | autointent_classic-light_autoembedder |
| 9 | classic-light | 50shot | 50 | 456 | 0.5602 | 0.1810 | autointent_classic-light_autoembedder |
| 10 | classic-light | full | None | 42 | 0.9031 | 0.6095 | autointent_classic-light_autoembedder |
| 11 | classic-light | full | None | 123 | 0.9045 | 0.6048 | autointent_classic-light_autoembedder |
| 12 | classic-light | full | None | 456 | 0.9162 | 0.6048 | autointent_classic-light_autoembedder |
| 13 | classic-medium | 10shot | 10 | 42 | 0.7572 | 0.9190 | autointent_classic-medium_autoembedder |
| 14 | classic-medium | 10shot | 10 | 123 | 0.7307 | 0.3619 | autointent_classic-medium_autoembedder |
| 15 | classic-medium | 10shot | 10 | 456 | 0.7825 | 0.2714 | autointent_classic-medium_autoembedder |
| 16 | classic-medium | 20shot | 20 | 42 | 0.8671 | 0.7429 | autointent_classic-medium_autoembedder |
| 17 | classic-medium | 20shot | 20 | 123 | 0.7497 | 0.4190 | autointent_classic-medium_autoembedder |
| 18 | classic-medium | 20shot | 20 | 456 | 0.7713 | 0.3524 | autointent_classic-medium_autoembedder |
| 19 | classic-medium | 50shot | 50 | 42 | 0.7668 | 0.5095 | autointent_classic-medium_autoembedder |
| 20 | classic-medium | 50shot | 50 | 123 | 0.6861 | 0.4714 | autointent_classic-medium_autoembedder |
| 21 | classic-medium | 50shot | 50 | 456 | 0.5602 | 0.1810 | autointent_classic-medium_autoembedder |
| 22 | zero-shot-encoders | 10shot | 10 | 42 | 0.9133 | 0.7667 | autointent_zero-shot-encoders_autoembedd |
| 23 | zero-shot-encoders | 10shot | 10 | 123 | 0.9133 | 0.7667 | autointent_zero-shot-encoders_autoembedd |
| 24 | zero-shot-encoders | 10shot | 10 | 456 | 0.9133 | 0.7667 | autointent_zero-shot-encoders_autoembedd |
| 25 | zero-shot-encoders | 20shot | 20 | 42 | 0.9133 | 0.7667 | autointent_zero-shot-encoders_autoembedd |
| 26 | zero-shot-encoders | 20shot | 20 | 123 | 0.9133 | 0.7667 | autointent_zero-shot-encoders_autoembedd |
| 27 | zero-shot-encoders | 20shot | 20 | 456 | 0.9133 | 0.7667 | autointent_zero-shot-encoders_autoembedd |
| 28 | zero-shot-encoders | 50shot | 50 | 42 | 0.9133 | 0.7667 | autointent_zero-shot-encoders_autoembedd |
| 29 | zero-shot-encoders | 50shot | 50 | 123 | 0.9133 | 0.7667 | autointent_zero-shot-encoders_autoembedd |
| 30 | zero-shot-encoders | 50shot | 50 | 456 | 0.9133 | 0.7667 | autointent_zero-shot-encoders_autoembedd |
| 31 | nn-medium | 10shot | 10 | 42 | 0.6215 | 0.4571 | autointent_nn-medium_autoembedder |
| 32 | nn-medium | 10shot | 10 | 123 | 0.0217 | 0.0286 | autointent_nn-medium_autoembedder |
| 33 | nn-medium | 10shot | 10 | 456 | 0.0334 | 0.0048 | autointent_nn-medium_autoembedder |
| 34 | nn-medium | 20shot | 20 | 42 | 0.9466 | 1.0000 | autointent_nn-medium_autoembedder |
| 35 | nn-medium | 20shot | 20 | 123 | 0.6052 | 0.4952 | autointent_nn-medium_autoembedder |
| 36 | nn-medium | 20shot | 20 | 456 | 0.8259 | 0.7667 | autointent_nn-medium_autoembedder |
| 37 | nn-medium | 50shot | 50 | 42 | 0.5738 | 0.4429 | autointent_nn-medium_autoembedder |
| 38 | nn-medium | 50shot | 50 | 123 | 0.6219 | 0.4476 | autointent_nn-medium_autoembedder |
| 39 | nn-medium | 50shot | 50 | 456 | 0.8670 | 0.8381 | autointent_nn-medium_autoembedder |

---

## Ключ прогона (дедупликация)

При повторном append `save_metrics` заменяет строку с тем же  
`(model_name, mode, n_shots, seed)`.
