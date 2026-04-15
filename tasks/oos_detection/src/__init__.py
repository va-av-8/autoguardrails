"""
OOS Detection utilities.

Modules:
- metrics: OOS detection metrics aligned with academic literature
- data_utils: data loading (standard + HuggingFace)
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
    load_standard,
    load_autointent,
    get_standard_intents,
    get_autointent_intents,
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
    "load_standard",
    "load_autointent",
    "get_standard_intents",
    "get_autointent_intents",
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
