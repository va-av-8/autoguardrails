"""
Обучение AutoIntent на CLINC150.

Использование:
    # Pilot (маленький embedder, быстрая валидация)
    python scripts/train_autointent.py --mode fewshot --n_shots 10 --seed 42 --pilot

    # Final (оригинальный embedder, сравнимо с Table 3)
    python scripts/train_autointent.py --mode fewshot --n_shots 10 --seed 42

    # Full train
    python scripts/train_autointent.py --mode full

    # С кастомной OOS-метрикой для decision node (для сравнения с фреймворками)
    python scripts/train_autointent.py --mode fewshot --n_shots 10 --seed 42 --decision-metric oos_f1

После обучения модель сохраняется в runs/<model_name>/.
Для оценки используйте eval_autointent.py.
"""

from __future__ import annotations

import argparse
import importlib.resources as ires
import sys
from pathlib import Path

# Load .env for macOS ARM fix (OMP_NUM_THREADS=1, etc.)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add project root to path
script_dir = Path(__file__).parent
task_dir = script_dir.parent
project_root = task_dir.parent.parent
sys.path.insert(0, str(project_root))

from autointent import Pipeline, Dataset as AIDataset
from autointent.configs import LoggingConfig, EmbedderConfig, DataConfig

sys.path.insert(0, str(task_dir))
from src.data_utils import (
    load_split_autointent,
    load_fewshot_autointent,
    get_intents,
)


def get_data_dir() -> Path:
    return task_dir / "data" / "processed"


def get_runs_dir() -> Path:
    runs_dir = task_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def get_configs_dir() -> Path:
    configs_dir = task_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    return configs_dir


def get_model_name(pilot: bool, decision_metric: str) -> str:
    suffix = "_oosf1" if decision_metric == "oos_f1" else ""
    if pilot:
        return f"autointent_classic-light_pilot{suffix}"
    return f"autointent_classic-light{suffix}"


def get_embedder_name(pilot: bool) -> str:
    if pilot:
        return "intfloat/multilingual-e5-small"
    return "intfloat/multilingual-e5-large-instruct"


def ensure_oosf1_config() -> Path:
    """
    Создаёт classic-light-oosf1.yaml программно из оригинального пресета.
    Меняет ТОЛЬКО target_metric у decision-ноды: decision_accuracy → oos_f1.
    Сохраняет точное форматирование оригинала (строковая замена).
    Возвращает путь к созданному файлу.
    """
    configs_dir = get_configs_dir()
    output_path = configs_dir / "classic-light-oosf1.yaml"

    # Читаем оригинальный classic-light.yaml как текст (сохраняем форматирование)
    preset_file = ires.files("autointent._presets").joinpath("classic-light.yaml")
    original_text = preset_file.read_text()

    # Заменяем ТОЛЬКО target_metric у decision-ноды (после строки node_type: decision)
    lines = original_text.split("\n")
    in_decision_node = False
    modified_lines = []
    for line in lines:
        if "node_type: decision" in line:
            in_decision_node = True
        if in_decision_node and "target_metric: decision_accuracy" in line:
            line = line.replace("decision_accuracy", "oos_f1")
            in_decision_node = False
        modified_lines.append(line)

    # Сохраняем кастомный конфиг (сохраняем оригинальный newline в конце)
    modified_text = "\n".join(modified_lines)
    if original_text.endswith("\n") and not modified_text.endswith("\n"):
        modified_text += "\n"
    output_path.write_text(modified_text)

    return output_path


def register_oos_f1_metric() -> None:
    """
    Регистрирует кастомную метрику oos_f1 в DECISION_METRICS.
    Идентична f1_oos из src/metrics.py, но работает с форматом AutoIntent (OOS=None).
    """
    from sklearn.metrics import f1_score
    from autointent.metrics import DECISION_METRICS

    def oos_f1(y_true, y_pred):
        """Бинарный F1 по OOS-классу. OOS=None → positive class."""
        y_true_bin = [1 if y is None else 0 for y in y_true]
        y_pred_bin = [1 if y is None else 0 for y in y_pred]
        return float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    DECISION_METRICS["oos_f1"] = oos_f1


def main():
    parser = argparse.ArgumentParser(description="Train AutoIntent on CLINC150")
    parser.add_argument(
        "--source",
        choices=["standard", "deeppavlov"],
        default="deeppavlov",
        help="Data source: standard (100 OOS) or deeppavlov (200 OOS)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "fewshot"],
        default="fewshot",
        help="Training mode",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        choices=[10, 20, 50],
        default=10,
        help="Number of shots per intent (for fewshot mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=[42, 123, 456],
        default=42,
        help="Random seed (for fewshot mode)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use small embedder for fast validation (not comparable to Table 3)",
    )
    parser.add_argument(
        "--decision-metric",
        choices=["decision_accuracy", "oos_f1"],
        default="decision_accuracy",
        help="Target metric for decision node optimization (oos_f1 = binary F1 on OOS class)",
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    runs_dir = get_runs_dir()

    # Model naming
    decision_metric = args.decision_metric
    model_name = get_model_name(args.pilot, decision_metric)
    embedder_name = get_embedder_name(args.pilot)

    if args.mode == "fewshot":
        mode_str = f"{args.n_shots}shot"
    else:
        mode_str = "full"

    # Output directory for this run (include source to avoid conflicts)
    model_dir = runs_dir / f"{model_name}_{args.source}_{mode_str}_seed{args.seed if args.seed else 0}"

    print("=" * 60)
    print("AutoIntent Training")
    print("=" * 60)
    print(f"Mode: {'PILOT' if args.pilot else 'FINAL'}")
    print(f"Embedder: {embedder_name}")
    print(f"Decision metric: {decision_metric}")
    print(f"Training: {mode_str}")
    print(f"Output: {model_dir}")
    print("=" * 60)
    print()

    # Load data (AutoIntent format)
    if args.mode == "fewshot":
        train_ai = load_fewshot_autointent(args.source, args.n_shots, args.seed)
    else:
        train_ai = load_split_autointent(args.source, "train")

    test_ai = load_split_autointent(args.source, "test")
    intents = get_intents(args.source)

    print(f"Train: {len(train_ai)} samples")
    print(f"Test: {len(test_ai)} samples")
    print(f"Intents: {len(intents)}")
    print()

    # Create AutoIntent Dataset
    ai_dataset = AIDataset.from_dict({
        "train": train_ai,
        "test": test_ai,
        "intents": intents,
    })

    # Create pipeline (with custom metric if needed)
    if decision_metric == "oos_f1":
        # 1. Register custom metric BEFORE creating pipeline
        register_oos_f1_metric()
        # 2. Create/ensure custom config exists
        config_path = ensure_oosf1_config()
        print(f"Using custom config: {config_path}")
        # 3. Create pipeline from custom config
        pipeline = Pipeline.from_optimization_config(config_path)
    else:
        pipeline = Pipeline.from_preset("classic-light")

    # Set embedder
    pipeline.set_config(EmbedderConfig(model_name=embedder_name))

    # Cross-validation for few-shot
    if args.mode == "fewshot":
        pipeline.set_config(DataConfig(scheme="cv", n_folds=3))

    # Logging config - save modules for later loading
    pipeline.set_config(LoggingConfig(
        project_dir=model_dir,
        dump_modules=True,  # Important: save for eval_autointent.py
        clear_ram=False,    # Keep in RAM for immediate dump
    ))

    # Train
    print(f"Starting AutoML optimization...")
    print(f"This may take a while (20 trials with cross-validation)")
    print()

    context = pipeline.fit(ai_dataset)
    print()
    print("AutoML optimization completed!")

    # Save model
    print(f"Saving model to {model_dir}...")
    pipeline.dump(model_dir)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "source": args.source,
        "mode": mode_str,
        "n_shots": args.n_shots if args.mode == "fewshot" else None,
        "seed": args.seed if args.mode == "fewshot" else None,
        "embedder": embedder_name,
        "pilot": args.pilot,
        "preset": "classic-light",
        "decision_metric": decision_metric,
    }

    import json
    metadata_file = model_dir / "train_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))

    print()
    print("=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Model saved to: {model_dir}")
    print()
    print("Next step - evaluate:")
    print(f"  python scripts/eval_autointent.py --model_dir {model_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
