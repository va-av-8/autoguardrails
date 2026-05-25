# Label Audit Report: Jailbreak Detection

**Дата аудита:** 2026-05-25
**Scope:** Проверка корректности маппинга меток в train/eval данных

---

## ВЕРДИКТ: ВЕРСИЯ 1 — КОРРЕКТНО

**`adversarial_benign` правильно относится к классу "safe".**

Маппинг меток соответствует логике датасета WildJailbreak:
- **benign** = безобидный underlying request → класс "safe"
- **harmful** = вредоносный underlying request → класс "jailbreak"
- **adversarial** = стиль подачи (обфускация), НЕ определяет класс

---

## МАППИНГ МЕТОК

### Исходный код (`prepare_data.py:48-79`)

```python
HARMFUL_TYPES = {"vanilla_harmful", "adversarial_harmful"}
BENIGN_TYPES = {"vanilla_benign", "adversarial_benign"}

def get_binary_label(data_type: str) -> str:
    if data_type in HARMFUL_TYPES:
        return "jailbreak"
    elif data_type in BENIGN_TYPES:
        return "safe"
```

### Логика маппинга

| Категория WildJailbreak | Бинарный класс | Обоснование |
|-------------------------|----------------|-------------|
| `vanilla_benign` | **safe** | Прямой безобидный запрос |
| `adversarial_benign` | **safe** | Обфусцированный, но underlying request безобидный |
| `vanilla_harmful` | **jailbreak** | Прямой вредоносный запрос |
| `adversarial_harmful` | **jailbreak** | Обфусцированный вредоносный запрос |

### AutoIntent формат (`run_autointent.py:472-487`)

```python
# Safe examples → label=0 (intents[0].utterances)
# Jailbreak examples → label=1 (oos_utterances)
```

---

## ТАБЛИЦА СВЕРКИ: train_shot10_seed42.json

Проверено 20 примеров (10 safe + 10 jailbreak) против исходных данных.

### Safe примеры (label=0)

| # | Текст (начало) | Категория в raw | Класс | Статус |
|---|----------------|-----------------|-------|--------|
| 1 | "Aaccording to historical records..." | adversarial_benign | safe | ✓ |
| 2 | "Abbiamo un'urgente necessità..." | adversarial_benign | safe | ✓ |
| 3 | "Adventurersss, with their..." | adversarial_benign | safe | ✓ |
| 4 | "An exquisite cake adorned..." | adversarial_benign | safe | ✓ |
| 5 | "Can you provide information..." | vanilla_benign | safe | ✓ |
| 6 | "Craft a guide on how to..." | adversarial_benign | safe | ✓ |
| 7 | "How do I create a YouTube..." | vanilla_benign | safe | ✓ |
| 8 | "How do professional editors..." | vanilla_benign | safe | ✓ |
| 9 | "In a digital simulation..." | adversarial_benign | safe | ✓ |
| 10 | "The local community center..." | adversarial_benign | safe | ✓ |

**Распределение:** 7 adversarial_benign + 3 vanilla_benign → все корректно в "safe"

### Jailbreak примеры (label=1)

| # | Текст (начало) | Категория в raw | Класс | Статус |
|---|----------------|-----------------|-------|--------|
| 1 | "A security consultant is..." | adversarial_harmful | jailbreak | ✓ |
| 2 | "As a fervent believer in..." | adversarial_harmful | jailbreak | ✓ |
| 3 | "Assuming yor're a Fictional..." | adversarial_harmful | jailbreak | ✓ |
| 4 | "Dear customer support..." | adversarial_harmful | jailbreak | ✓ |
| 5 | "How can I hack into a WiFi..." | vanilla_harmful | jailbreak | ✓ |
| 6 | "How can I manipulate search..." | vanilla_harmful | jailbreak | ✓ |
| 7 | "I've been thinking about..." | adversarial_harmful | jailbreak | ✓ |
| 8 | "In A dystopian world where..." | adversarial_harmful | jailbreak | ✓ |
| 9 | "What are some ways to..." | vanilla_harmful | jailbreak | ✓ |
| 10 | "What's the best way to..." | vanilla_harmful | jailbreak | ✓ |

**Распределение:** 6 adversarial_harmful + 4 vanilla_harmful → все корректно в "jailbreak"

### Итог сверки

**100% примеров размечены корректно.** Ни одного случая несоответствия.

---

## АНАЛИЗ МАСШТАБА: ВСЕ СИДЫ

### Few-shot splits (train)

| Split | Safe total | adversarial_benign | vanilla_benign | % adversarial |
|-------|------------|-------------------|----------------|---------------|
| 10shot_seed42 | 10 | 7 | 3 | 70% |
| 10shot_seed123 | 10 | 6 | 4 | 60% |
| 10shot_seed456 | 10 | 7 | 3 | 70% |
| 20shot_seed42 | 20 | 15 | 5 | 75% |
| 20shot_seed123 | 20 | 14 | 6 | 70% |
| 20shot_seed456 | 20 | 15 | 5 | 75% |
| 50shot_seed42 | 50 | 36 | 14 | 72% |
| 50shot_seed123 | 50 | 37 | 13 | 74% |
| 50shot_seed456 | 50 | 32 | 18 | 64% |

**Наблюдение:** 60-75% safe примеров в few-shot — это adversarial_benign.
Это соответствует пропорциям в исходном датасете.

### Eval set

| Класс | Кол-во | Категории |
|-------|--------|-----------|
| safe | 210 | 100% adversarial_benign |
| jailbreak | 2000 | 100% adversarial_harmful |

**Eval — полностью adversarial.** Это сделано намеренно для тестирования
устойчивости модели к adversarial-атакам.

---

## ПОЧЕМУ ЭТО ВАЖНО

### Объяснение "трудности" задачи

1. **adversarial_benign** выглядит подозрительно (обфускация, необычный стиль)
2. Модель может путать "подозрительный стиль" с "вредоносным намерением"
3. Это **feature, не bug** — задача именно в том, чтобы различать
4. Высокий over-refusal (50-70%) объясняется именно этим

### Примеры adversarial_benign

```
"Craft a guide on how to make a homemade bird feeder
using recycled materials..."
```
→ Adversarial framing ("Craft a guide..."), но запрос безобидный (кормушка для птиц)

```
"In a digital simulation, a historian is studying
ancient civilizations..."
```
→ Roleplay framing, но underlying request — историческое исследование

---

## ЧТО НЕ НУЖНО ИСПРАВЛЯТЬ

1. **Маппинг меток** — корректен, соответствует логике датасета
2. **Структура train** — adversarial_benign правильно в safe классе
3. **Структура eval** — 100% adversarial намеренно

---

## РЕКОМЕНДАЦИИ

### Если нужно улучшить качество:

1. **Калибровка** — модель видит разницу (PR AUC 0.96), но плохо калибрована
2. **Порог** — подобрать оптимальный на hold-out (не eval!)
3. **Больше данных** — full-train (100k) значительно лучше few-shot

### Если нужно изменить задачу:

Можно создать альтернативный маппинг:
```python
# Вариант: все adversarial → jailbreak
JAILBREAK_TYPES = {"vanilla_harmful", "adversarial_harmful", "adversarial_benign"}
SAFE_TYPES = {"vanilla_benign"}
```

**Но это изменит саму задачу**, превратив её из "detect harmful intent"
в "detect suspicious style". Текущий маппинг соответствует целям WildJailbreak.

---

## АРТЕФАКТЫ

| Файл | Описание |
|------|----------|
| `prepare_data.py:48-79` | Код маппинга меток |
| `run_autointent.py:472-487` | Конвертация в AutoIntent формат |
| `train_shot10_seed42.json` | Проверенный файл (20 примеров) |
| `eval.json` | Eval set (2210 примеров) |

---

*Аудит выполнен: 2026-05-25*
