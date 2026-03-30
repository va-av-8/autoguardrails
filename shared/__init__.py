"""
Shared utilities for AutoGuardrails experiments.

Modules:
- metrics: guardrail-specific metrics (Recall@FPR, AUROC, F1)
- data_utils: data loading and preprocessing
- evaluation: unified evaluation interface
- visualization: plotting utilities
"""

from .metrics import recall_at_fpr, auroc_oos, f1_oos, compute_all_metrics
from .data_utils import load_clinc150, sample_fewshot, convert_to_autointent_format
from .evaluation import Evaluator, EvaluationResult

__all__ = [
    "recall_at_fpr",
    "auroc_oos",
    "f1_oos",
    "compute_all_metrics",
    "load_clinc150",
    "sample_fewshot",
    "convert_to_autointent_format",
    "Evaluator",
    "EvaluationResult",
]
