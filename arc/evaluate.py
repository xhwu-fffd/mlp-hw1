from __future__ import annotations

import argparse
from pathlib import Path

from mlp_hw1.data import EuroSATDataBundle, create_data_bundle
from mlp_hw1.metrics import confusion_matrix
from mlp_hw1.model import MLPClassifier
from mlp_hw1.serialization import load_checkpoint
from mlp_hw1.trainer import evaluate_model
from mlp_hw1.utils import save_json, write_csv
from mlp_hw1.visualization import plot_confusion_matrix, plot_first_layer_weights


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    default_checkpoint = project_root / "artifacts" / "train" / "best_model.npz"
    default_batch_output = project_root / "artifacts" / "batch_evaluation"

    parser = argparse.ArgumentParser(
        description="Evaluate one or many saved EuroSAT MLP checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint, help="Single checkpoint to evaluate.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        action="append",
        default=None,
        help="Recursively evaluate all matching checkpoints under this directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.npz",
        help="Filename to search for under --checkpoint-dir directories.",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Single mode: evaluation artifact directory. Batch mode: summary output directory.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Batch mode only: also save confusion matrix and first-layer-weight figures per checkpoint.",
    )
    parser.add_argument("--summary-csv", type=Path, default=None, help="Batch mode summary CSV path.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Batch mode best-result JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Batch mode only: evaluate at most this many checkpoints.")
    parser.set_defaults(default_batch_output=default_batch_output)
    return parser


def _resolve_data_bundle(
    metadata: dict,
    dataset_root_override: Path | None,
    bundle_cache: dict[tuple, EuroSATDataBundle],
) -> tuple[EuroSATDataBundle, dict]:
    project_root = Path(__file__).resolve().parent
    run_metadata = metadata.get("run_metadata", {})
    data_config = run_metadata.get("data_config", {})

    dataset_root = Path(
        dataset_root_override
        or run_metadata.get("dataset_root")
        or (project_root.parent / "EuroSAT_RGB")
    ).resolve()

    cache_value = data_config.get("cache_path")
    cache_path = Path(cache_value).resolve() if cache_value else None
    max_samples_per_class = data_config.get("max_samples_per_class")
    train_ratio = data_config.get("train_ratio", 0.7)
    val_ratio = data_config.get("val_ratio", 0.15)
    test_ratio = data_config.get("test_ratio", 0.15)
    seed = data_config.get("seed", metadata.get("training", {}).get("seed", 42))

    cache_key = (
        str(dataset_root),
        str(cache_path) if cache_path else None,
        max_samples_per_class,
        train_ratio,
        val_ratio,
        test_ratio,
        seed,
    )
    if cache_key not in bundle_cache:
        bundle_cache[cache_key] = create_data_bundle(
            dataset_root=dataset_root,
            cache_path=cache_path,
            max_samples_per_class=max_samples_per_class,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    return bundle_cache[cache_key], {
        "dataset_root": str(dataset_root),
        "cache_path": str(cache_path) if cache_path else None,
        "max_samples_per_class": max_samples_per_class,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
    }


def _build_model_from_metadata(metadata: dict) -> MLPClassifier:
    model_config = metadata["model"]
    return MLPClassifier(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_classes=model_config["num_classes"],
        activation=model_config["activation"],
        seed=model_config["seed"],
    )


def _evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_root_override: Path | None,
    bundle_cache: dict[tuple, EuroSATDataBundle],
    output_dir: Path | None,
    save_artifacts: bool,
) -> dict:
    checkpoint_path = checkpoint_path.resolve()
    state_dict, metadata = load_checkpoint(checkpoint_path)
    data_bundle, resolved_data_config = _resolve_data_bundle(metadata, dataset_root_override, bundle_cache)

    model = _build_model_from_metadata(metadata)
    model.load_state_dict(state_dict)

    training_config = metadata.get("training", {})
    metrics = evaluate_model(model, data_bundle.test, batch_size=training_config.get("batch_size", 128))
    matrix = confusion_matrix(metrics["y_true"], metrics["y_pred"], data_bundle.num_classes)

    evaluation_dir = None
    if save_artifacts:
        evaluation_dir = (output_dir or (checkpoint_path.parent / "evaluation")).resolve()
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(matrix, data_bundle.class_names, evaluation_dir / "confusion_matrix.png")
        plot_first_layer_weights(model.fc1.weight.data, data_bundle.image_shape, evaluation_dir / "first_layer_weights.png")
        save_json(
            evaluation_dir / "evaluation_summary.json",
            {
                "test_loss": float(metrics["loss"]),
                "test_accuracy": float(metrics["accuracy"]),
                "checkpoint": str(checkpoint_path),
                "dataset_root": resolved_data_config["dataset_root"],
            },
        )

    results_section = metadata.get("results", {})
    progress_section = metadata.get("progress", {})
    model_section = metadata.get("model", {})

    return {
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(checkpoint_path.parent),
        "hidden_dim": model_section.get("hidden_dim"),
        "activation": model_section.get("activation"),
        "learning_rate": training_config.get("learning_rate"),
        "lr_decay": training_config.get("lr_decay"),
        "weight_decay": training_config.get("weight_decay"),
        "batch_size": training_config.get("batch_size"),
        "best_epoch": results_section.get("best_epoch", progress_section.get("best_epoch")),
        "best_val_accuracy": results_section.get("best_val_accuracy", progress_section.get("best_val_accuracy")),
        "test_loss": float(metrics["loss"]),
        "test_accuracy": float(metrics["accuracy"]),
        "evaluation_dir": str(evaluation_dir) if evaluation_dir else "",
    }


def _find_checkpoints(checkpoint_dirs: list[Path], checkpoint_name: str) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()

    for checkpoint_dir in checkpoint_dirs:
        for checkpoint_path in sorted(Path(checkpoint_dir).resolve().rglob(checkpoint_name)):
            path_key = str(checkpoint_path.resolve())
            if path_key in seen:
                continue
            seen.add(path_key)
            found.append(checkpoint_path.resolve())

    return found


def _run_single(args: argparse.Namespace) -> None:
    bundle_cache: dict[tuple, EuroSATDataBundle] = {}
    checkpoint_path = Path(args.checkpoint).resolve()
    summary = _evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset_root_override=args.dataset_root,
        bundle_cache=bundle_cache,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        save_artifacts=True,
    )

    print(f"Test loss: {summary['test_loss']:.4f}")
    print(f"Test accuracy: {summary['test_accuracy']:.4f}")
    print(f"Saved evaluation artifacts to: {summary['evaluation_dir']}")


def _run_batch(args: argparse.Namespace) -> None:
    checkpoint_dirs = [Path(path).resolve() for path in args.checkpoint_dir]
    checkpoints = _find_checkpoints(checkpoint_dirs, args.checkpoint_name)
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]

    if not checkpoints:
        raise FileNotFoundError("No checkpoints were found under the provided --checkpoint-dir directories.")

    bundle_cache: dict[tuple, EuroSATDataBundle] = {}
    rows: list[dict] = []

    for index, checkpoint_path in enumerate(checkpoints, start=1):
        print(f"Evaluating checkpoint {index}/{len(checkpoints)}: {checkpoint_path}")
        artifact_dir = checkpoint_path.parent / "evaluation" if args.save_artifacts else None
        rows.append(
            _evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                dataset_root_override=args.dataset_root,
                bundle_cache=bundle_cache,
                output_dir=artifact_dir,
                save_artifacts=args.save_artifacts,
            )
        )

    rows.sort(key=lambda row: row["test_accuracy"], reverse=True)

    batch_output_dir = Path(args.output_dir).resolve() if args.output_dir else args.default_batch_output.resolve()
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else batch_output_dir / "batch_evaluation.csv"
    summary_json = Path(args.summary_json).resolve() if args.summary_json else batch_output_dir / "best_checkpoint.json"

    write_csv(summary_csv, rows)
    save_json(summary_json, rows[0])

    print(f"Batch evaluation finished for {len(rows)} checkpoints.")
    print(f"Best test accuracy: {rows[0]['test_accuracy']:.4f}")
    print(f"Best checkpoint: {rows[0]['checkpoint_path']}")
    print(f"Summary CSV saved to: {summary_csv}")
    print(f"Best-result JSON saved to: {summary_json}")


def main() -> None:
    args = build_parser().parse_args()
    if args.checkpoint_dir:
        _run_batch(args)
        return
    _run_single(args)


if __name__ == "__main__":
    main()
