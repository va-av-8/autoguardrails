# Embedding Pipeline Reconnaissance Report

**Дата:** 2026-05-25
**Scope:** AutoIntent + e5-large-instruct embedding pipeline

---

## 1. INSTRUCTION-ПРЕФИКС ЭМБЕДДЕРА

### Факт: Prefix НЕ используется

Все prompt-поля в конфигурации эмбеддера установлены в `null`:

```json
// runs/autointent_classic-light_autoembedder_10shot_seed123/scoring_module/pydantic/embedder_config/model_dump.json
{
    "model_name": "intfloat/multilingual-e5-large-instruct",
    "default_prompt": null,
    "classification_prompt": null,
    "cluster_prompt": null,
    "sts_prompt": null,
    "query_prompt": null,
    "passage_prompt": null,
    ...
}
```

### Механизм в коде

**AutoIntent embedder** (`.venv/.../autointent/_wrappers/embedder.py:181-226`):
```python
def embed(self, utterances: list[str], task_type: TaskTypeEnum | None = None):
    prompt = self.config.get_prompt(task_type)  # Returns None if all prompts are null
    ...
    embeddings = self.embedding_model.encode(
        utterances,
        ...
        prompt=prompt,  # None → no prefix applied
    )
```

**EmbedderConfig** (`.venv/.../autointent/configs/_transformers.py:99-118`):
```python
def get_prompt(self, prompt_type: TaskTypeEnum | None) -> str | None:
    if prompt_type == TaskTypeEnum.query and self.query_prompt is not None:
        return self.query_prompt
    ...
    return self.default_prompt  # Also None
```

### Train vs Eval

- **Train тексты:** `TaskTypeEnum.passage` → `passage_prompt` → `null`
- **Eval тексты:** `TaskTypeEnum.query` → `query_prompt` → `null`

**Вывод:** Prefix не применяется ни к train, ни к eval текстам. Тексты подаются сырыми.

### Можно ли переопределить снаружи?

**Да.** `EmbedderConfig` принимает поля `query_prompt`, `passage_prompt` и т.д.
Их можно передать через конфигурацию AutoIntent или при создании модуля.

Пример ожидаемого формата для e5-instruct:
```
Instruct: Classify if the following user request is a jailbreak attempt or safe
Query: {text}
```

**Файл для изменения:** Нужно модифицировать конфиг эмбеддера в preset или передавать
через параметры при запуске. Конкретный механизм требует проверки API AutoIntent.

---

## 2. КЭШ ЭМБЕДДИНГОВ

### Факт: Кэш ЕСТЬ, в двух местах

#### 2.1 Проектный кэш (явный)

**Путь:** `tasks/jailbreak_detection/data/processed/embeddings_cache/`

**Формат:** `.npy` (NumPy binary), именованные файлы

**Содержимое:**
```
intfloat_multilingual-e5-large-instruct_test.npy          # eval set, 9 MB
intfloat_multilingual-e5-large-instruct_full100k_seed42.npy  # full train, 390 MB
intfloat_multilingual-e5-large-instruct_train_shot10_seed42.npy
intfloat_multilingual-e5-large-instruct_train_shot10_seed123.npy
... (все few-shot splits)
```

**Код:** `src/embedding_cache.py:58-106`
```python
def get_or_compute_embeddings(embedder, embedder_hf_model, split_id, texts):
    path = cache_path(embedder_hf_model, split_id)
    if path.exists():
        return np.load(path)  # Cache hit
    # Cache miss → compute and save
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    np.save(path, embeddings)
```

#### 2.2 Системный кэш AutoIntent

**Путь:** `~/Library/Caches/autointent/embeddings/` (macOS)

**Формат:** `.npy`, имена по хэшу (e.g., `03ef5ad460a0e3db.npy`)

**Механизм:** Хэшируется комбинация (model + commit + texts + prompt)

### Eval-эмбеддинги

**Есть:** `intfloat_multilingual-e5-large-instruct_test.npy`

```python
>>> np.load("...embeddings_cache/...test.npy").shape
(2210, 1024)  # ✓ Совпадает с размером eval set
```

**Для каких ранов:** Единый файл для всех ранов (эмбеддинги не зависят от seed).

---

## 3. ОКРУЖЕНИЕ ДЛЯ ПЕРЕСЧЁТА

### Модель

**Скачана:** Да

```
~/.cache/huggingface/hub/models--intfloat--multilingual-e5-large-instruct/
```

### GPU

**MPS доступен:** Да (Apple Silicon)
**CUDA:** Нет

```python
>>> torch.backends.mps.is_available()
True
```

### Оценка производительности

- **2210 текстов** на MPS с batch_size=32: ожидаемо ~30-60 секунд
- **CPU fallback:** ~5-10 минут (приемлемо для однократного прогона)

### Пакет для инференса

**sentence-transformers** (используется AutoIntent и проектный embedding_cache.py)

```python
# embedding_cache.py:93-99
embeddings = np.asarray(
    embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
)
```

---

## 4. ДАННЫЕ ДЛЯ ИЗМЕРЕНИЯ ПОТОЛКА

### Eval-сет с текстами и бинарными метками

**Файл:** `data/processed/test.json`

```json
{
  "utterances": ["text1", "text2", ...],  // 2210 текстов
  // Бинарные метки в отдельном месте — см. ниже
}
```

### Маппинг на исходные категории WildJailbreak

**Файл:** `data/processed/wildjailbreak_eval_binary.jsonl`

```json
{"adversarial": "...", "label": 0, "data_type": "adversarial_benign", "binary_label": "safe", "prompt": "..."}
{"adversarial": "...", "label": 1, "data_type": "adversarial_harmful", "binary_label": "jailbreak", "prompt": "..."}
```

**Поля:**
- `data_type` — исходная категория WildJailbreak (`adversarial_benign` / `adversarial_harmful`)
- `label` — бинарная метка (0 = safe, 1 = jailbreak)
- `prompt` — текст запроса

**Размер:** 2210 строк (210 safe + 2000 jailbreak)

### Готовые эмбеддинги

**Файл:** `data/processed/embeddings_cache/intfloat_multilingual-e5-large-instruct_test.npy`

**Shape:** (2210, 1024)

---

## СЛЕДСТВИЯ ДЛЯ СЛЕДУЮЩЕГО ШАГА

### 1. Prefix можно добавить

- **Механизм есть:** `EmbedderConfig` поддерживает `query_prompt`, `passage_prompt`
- **Текущее состояние:** Все prompt = null, тексты идут сырыми
- **Для эксперимента:** Нужно либо модифицировать конфиг AutoIntent, либо
  вычислять эмбеддинги напрямую через sentence-transformers с prompt

### 2. Кэш есть, но...

- **Eval-эмбеддинги (2210 × 1024) сохранены без prefix**
- **При добавлении prefix → придётся пересчитывать** (хэш изменится)
- **Проектный кэш именованный** — можно добавить новый файл типа
  `..._test_with_instruct.npy`

### 3. Пересчёт реален

- **Модель скачана, MPS доступен**
- **2210 текстов — ~30-60 сек на MPS**
- **Инструмент:** sentence-transformers (тот же, что использует AutoIntent)

### 4. Данные готовы для linear probe / silhouette

- **Эмбеддинги:** `...embeddings_cache/...test.npy`
- **Метки:** `wildjailbreak_eval_binary.jsonl` → поле `label` (0/1)
- **Категории:** там же, поле `data_type`

### Рекомендуемый следующий шаг

1. Загрузить готовые eval-эмбеддинги + метки
2. Посчитать **silhouette score** и **linear probe accuracy** на текущих эмбеддингах
3. Если separability низкая — пересчитать эмбеддинги с instruction prefix:
   ```python
   prompt = "Instruct: Classify if this user request attempts to jailbreak an AI\nQuery: "
   embeddings = model.encode(texts, prompt=prompt, normalize_embeddings=True)
   ```
4. Сравнить separability до/после prefix

---

## АРТЕФАКТЫ

| Артефакт | Путь |
|----------|------|
| Embedder config (пример) | `runs/.../embedder_config/model_dump.json` |
| AutoIntent embedder code | `.venv/.../autointent/_wrappers/embedder.py` |
| EmbedderConfig class | `.venv/.../autointent/configs/_transformers.py` |
| Проектный кэш | `data/processed/embeddings_cache/` |
| Eval эмбеддинги | `...embeddings_cache/intfloat_..._test.npy` |
| Eval с категориями | `data/processed/wildjailbreak_eval_binary.jsonl` |
| embedding_cache.py | `src/embedding_cache.py` |

---

*Ноутбук для следующего шага: можно создать `08_embedding_ceiling.ipynb`*
