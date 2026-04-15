"""
OOS Detection utilities.

Modules:
- data_utils: data loading (standard + deeppavlov)
- metrics: OOS detection metrics
- evaluation: unified evaluation interface
- visualization: plotting utilities
"""

from .data_utils import (
    load_split,
    load_fewshot,
    get_intents,
    get_intent_names,
    load_meta,
    get_split_stats,
    OOS_LABEL,
)
from .metrics import (
    oos_recall,
    in_domain_accuracy,
    f1_oos,
    auroc,
    au_ioc,
    measure_latency,
    compute_all_metrics,
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
    # Constants
    "OOS_LABEL",
    # Data loading
    "load_split",
    "load_fewshot",
    "get_intents",
    "get_intent_names",
    "load_meta",
    "get_split_stats",
    # Metrics
    "oos_recall",
    "in_domain_accuracy",
    "f1_oos",
    "auroc",
    "au_ioc",
    "measure_latency",
    "compute_all_metrics",
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
