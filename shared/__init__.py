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
from .data_utils import (
    load_clinc150,
    load_clinc150_autointent,
    load_fewshot,
    load_fewshot_autointent,
    load_intents,
    load_meta,
    get_intent_names,
)
from .evaluation import Evaluator, EvaluationResult
from .visualization import (
    plot_roc_curves,
    plot_fewshot_scaling,
    plot_confusion_matrix,
    plot_threshold_curve,
    plot_ioc_curve,
)

__all__ = [
    # Metrics
    "oos_recall",
    "in_domain_accuracy",
    "f1_oos",
    "auroc",
    "au_ioc",
    "measure_latency",
    "compute_all_metrics",
    # Data loading
    "load_clinc150",
    "load_clinc150_autointent",
    "load_fewshot",
    "load_fewshot_autointent",
    "load_intents",
    "load_meta",
    "get_intent_names",
    # Evaluation
    "Evaluator",
    "EvaluationResult",
    # Visualization
    "plot_roc_curves",
    "plot_fewshot_scaling",
    "plot_confusion_matrix",
    "plot_threshold_curve",
    "plot_ioc_curve",
]
