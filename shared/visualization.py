"""Визуализация результатов для guardrail-экспериментов."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve,
)


def plot_roc_curves(
    results: list[dict],
    y_true: np.ndarray,
    y_scores_dict: dict[str, np.ndarray],
    oos_label: int = -1,
    save_path: str | Path | None = None,
) -> None:
    """
    ROC-кривые для всех моделей на одном графике.

    Args:
        results: список результатов из Evaluator
        y_true: истинные метки
        y_scores_dict: {model_name: oos_scores}
        oos_label: метка OOS-класса
        save_path: путь для сохранения (опционально)
    """
    y_true = np.asarray(y_true)
    y_binary = (y_true == oos_label).astype(int)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(y_scores_dict)))

    for (model_name, y_scores), color in zip(y_scores_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_binary, y_scores)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})')

    # Reference lines
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.axvline(x=0.05, color='gray', linestyle=':', alpha=0.7, label='FPR=0.05')
    ax.axvline(x=0.10, color='gray', linestyle='--', alpha=0.7, label='FPR=0.10')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - In-domain Specificity)')
    ax.set_ylabel('True Positive Rate (OOS Recall)')
    ax.set_title('ROC Curves: OOS Detection')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_fewshot_scaling(
    results: list[dict],
    metric: str = "oos_recall",
    save_path: str | Path | None = None,
) -> None:
    """
    Зависимость метрики от количества few-shot примеров.
    Основной график для демонстрации production-применимости AutoIntent.

    Args:
        results: список результатов из Evaluator
        metric: метрика для отображения
        save_path: путь для сохранения (опционально)
    """
    # Group by model
    models = {}
    for r in results:
        model_name = r["model_name"]
        if model_name not in models:
            models[model_name] = {"n_shots": [], "values": [], "seeds": {}}

        n = r.get("n_shots")
        if n is None:
            n = "full"
        val = r.get(metric, 0.0)
        seed = r.get("seed", 0)

        # Group by n_shots
        if n not in models[model_name]["seeds"]:
            models[model_name]["seeds"][n] = []
        models[model_name]["seeds"][n].append(val)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    x_labels = ["10", "20", "50", "full"]
    x_pos = [0, 1, 2, 3]

    for (model_name, data), color in zip(models.items(), colors):
        x_vals = []
        y_means = []
        y_stds = []

        for n in [10, 20, 50, "full"]:
            if n in data["seeds"]:
                vals = data["seeds"][n]
                x_vals.append(x_labels.index(str(n)) if isinstance(n, int) else 3)
                y_means.append(np.mean(vals))
                y_stds.append(np.std(vals))

        if x_vals:
            ax.errorbar(x_vals, y_means, yerr=y_stds, marker='o', capsize=5,
                       color=color, label=model_name, lw=2, markersize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Training Examples per Intent')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Few-shot Scaling: {metric.replace("_", " ").title()}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    intent_names: dict[int, str] | None = None,
    oos_label: int = -1,
    save_path: str | Path | None = None,
    top_k: int = 10,
) -> None:
    """
    Матрица ошибок с выделением OOS-строки и столбца.

    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        intent_names: {label_id: name} для in-scope интентов
        oos_label: метка OOS-класса
        save_path: путь для сохранения (опционально)
        top_k: показать только top-k интентов по ошибкам (+ OOS)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Get unique labels
    all_labels = sorted(set(y_true) | set(y_pred))

    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Create label names
    if intent_names is None:
        intent_names = {}
    labels_display = []
    for lbl in all_labels:
        if lbl == oos_label:
            labels_display.append("OOS")
        elif lbl in intent_names:
            labels_display.append(intent_names[lbl][:15])  # Truncate long names
        else:
            labels_display.append(f"Intent_{lbl}")

    # For large matrices, show only top-k most confused intents + OOS
    if len(all_labels) > top_k + 1:
        # Calculate total errors per class
        errors = cm.sum(axis=1) - np.diag(cm)
        # Always include OOS
        oos_idx = all_labels.index(oos_label) if oos_label in all_labels else -1

        # Get indices of top-k error classes
        top_indices = np.argsort(errors)[-top_k:]
        if oos_idx >= 0 and oos_idx not in top_indices:
            top_indices = np.append(top_indices, oos_idx)
        top_indices = sorted(top_indices)

        cm = cm[np.ix_(top_indices, top_indices)]
        labels_display = [labels_display[i] for i in top_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_display, yticklabels=labels_display,
                ax=ax, cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (OOS Detection)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_threshold_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    oos_label: int = -1,
    save_path: str | Path | None = None,
) -> None:
    """
    Precision/Recall/F1 как функция порога.
    Помогает выбрать рабочую точку под конкретный сценарий.

    Args:
        y_true: истинные метки
        y_scores: OOS-скоры (выше = более OOS)
        oos_label: метка OOS-класса
        save_path: путь для сохранения (опционально)
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # Binary labels
    y_binary = (y_true == oos_label).astype(int)

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_binary, y_scores)

    # F1 scores (avoid division by zero)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0
    )

    # Best threshold (max F1)
    best_idx = np.argmax(f1_scores[:-1])  # Last element is for recall=0
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds, precision[:-1], 'b-', label='Precision', lw=2)
    ax.plot(thresholds, recall[:-1], 'r-', label='Recall (OOS)', lw=2)
    ax.plot(thresholds, f1_scores[:-1], 'g-', label='F1', lw=2)

    # Mark best F1 point
    ax.axvline(x=best_threshold, color='gray', linestyle='--', alpha=0.7)
    ax.plot(best_threshold, best_f1, 'go', markersize=10,
           label=f'Best F1={best_f1:.3f} @ thresh={best_threshold:.3f}')

    ax.set_xlabel('OOS Score Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Selection: Precision, Recall, F1 vs Threshold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([y_scores.min(), y_scores.max()])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_ioc_curve(
    y_true: np.ndarray,
    y_scores_dict: dict[str, np.ndarray],
    oos_label: int = -1,
    save_path: str | Path | None = None,
) -> None:
    """
    In-scope/Out-of-scope Characteristic (IOC) curves.
    X: in-domain accuracy, Y: OOS recall at various thresholds.

    Args:
        y_true: истинные метки
        y_scores_dict: {model_name: oos_scores}
        oos_label: метка OOS-класса
        save_path: путь для сохранения (опционально)
    """
    y_true = np.asarray(y_true)

    oos_mask = y_true == oos_label
    inscope_mask = ~oos_mask
    n_oos = oos_mask.sum()
    n_inscope = inscope_mask.sum()

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(y_scores_dict)))

    for (model_name, y_scores), color in zip(y_scores_dict.items(), colors):
        y_scores = np.asarray(y_scores)

        thresholds = np.linspace(y_scores.min(), y_scores.max(), 100)
        x_points = []
        y_points = []

        for thresh in thresholds:
            pred_oos = y_scores >= thresh

            # OOS recall
            oos_recall_val = (oos_mask & pred_oos).sum() / n_oos

            # In-domain "accuracy" (% of in-scope not flagged as OOS)
            in_domain_acc = (~pred_oos & inscope_mask).sum() / n_inscope

            x_points.append(in_domain_acc)
            y_points.append(oos_recall_val)

        ax.plot(x_points, y_points, color=color, lw=2, label=model_name)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

    ax.set_xlabel('In-Domain Accuracy (not flagged as OOS)')
    ax.set_ylabel('OOS Recall')
    ax.set_title('IOC Curves: Guardrail Trade-off')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
