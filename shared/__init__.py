"""
Shared utilities for AutoGuardrails experiments.

Modules:
- metrics: OOS detection metrics aligned with academic literature
- data_utils: data loading and preprocessing
- evaluation: unified evaluation interface
- visualization: plotting utilities
"""

from .metrics import (
    oos_recall,
    in_domain_accuracy,
    f1_oos,
    auroc,
    au_ioc,
    measure_latency,
    compute_all_metrics,
)
from .data_utils import load_clinc150, sample_fewshot, convert_to_autointent_format
from .evaluation import Evaluator, EvaluationResult

__all__ = [
    "oos_recall",
    "in_domain_accuracy",
    "f1_oos",
    "auroc",
    "au_ioc",
    "measure_latency",
    "compute_all_metrics",
    "load_clinc150",
    "sample_fewshot",
    "convert_to_autointent_format",
    "Evaluator",
    "EvaluationResult",
]
