# Jailbreak Detection

## Постановка задачи
Детекция попыток jailbreak на input, до передачи промпта в LLM.
OOS-переформулировка: safe = in-scope интент, jailbreak = OOS-класс.

## Датасет
WildJailbreak (Jiang et al., NeurIPS 2024)
- HuggingFace: allenai/wildjailbreak
- Train: 262K примеров (vanilla_harmful, vanilla_benign,
  adversarial_harmful, adversarial_benign)
- Eval: 2.2K примеров — используется как test
- Бинарная метка: data_type ∈ {vanilla_harmful, adversarial_harmful}
  → jailbreak (OOS); {vanilla_benign, adversarial_benign} → safe (in-scope)

## Метрики
- F1, Precision, Recall на jailbreak-классе
- Over-refusal Rate = FPR на safe-промптах

## Протокол few-shot
- n_shots: [10, 20, 50] примеров на класс
- seeds: [42, 123, 456]
- Итого runs: 9
