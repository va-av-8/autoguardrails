# Hypothesis Log — Jailbreak Detection

## HYP-JB-003: AutoIntent как бинарный классификатор на adversarial данных

- **Задача:** Jailbreak Detection
- **Гипотеза:** AutoIntent в режиме бинарной классификации (safe=0,
  jailbreak=1) достигает конкурентного качества на adversarial
  промптах WildJailbreak в few-shot режиме
- **Мотивация:** OOS-переформулировка невозможна — scoring-модуль
  требует 2+ класса. Бинарная постановка более естественна для
  jailbreak: у нас есть явные примеры обоих классов, adversarial
  промпты — не случайные аномалии, а структурированные атаки
  с распознаваемыми паттернами
- **Реализация:** tasks/jailbreak_detection/notebooks/03_autointent_fewshot.ipynb,
  n_shots: [10, 20, 50], seeds: [42, 123, 456], 9 runs
- **Метрика:** F1, Recall, Over-refusal Rate на eval WildJailbreak (2210 примеров)
- **Результат:** 10-shot: F1=0.757±0.026, Recall=0.642±0.039;
  20-shot: F1=0.796±0.062, Recall=0.701±0.108;
  50-shot: F1=0.731±0.148, Recall=0.623±0.196 (CV=20.2%, UNSTABLE).
  Лучший run: 20-shot seed=42, F1=0.867, Recall=0.825,
  Over-refusal Rate=0.743.
- **Вывод:** Частично подтверждена. AutoIntent как бинарный
  классификатор работает на adversarial данных (F1≈0.796 при 20-shot,
  сопоставимо с OOS-спринтом F1=0.819). Критическая проблема:
  высокий Over-refusal Rate (до 74% на лучшем run) и нестабильность
  на 50-shot (CV=20.2%). Скейлинг немонотонен — та же аномалия,
  что в OOS-спринте.
- **Статус:** выполнена

---

## HYP-JB-004: Асимметричная cost function для снижения Over-refusal Rate

- **Задача:** Jailbreak Detection
- **Гипотеза:** Постфактум подбор порога под асимметричную функцию
  α·FNR + β·FPR (β > α, FP дороже FN) снизит Over-refusal Rate
  до приемлемого уровня (<20%) при умеренной потере Recall
- **Мотивация:** Лучший run (20-shot seed=42) блокирует 74% безопасных
  промптов — неприемлемо для guardrail. В отличие от OOS-спринта
  (HYP-002), где baseline уже имел низкий FPR=0.066 и асимметрия
  давала невыгодный trade-off, здесь стартовый Over-refusal Rate
  критически высок — есть пространство для улучшения
- **Реализация:** —
- **Метрика:** Over-refusal Rate, Recall, F1 на eval WildJailbreak
- **Результат:** —
- **Вывод:** —
- **Статус:** запланирована
