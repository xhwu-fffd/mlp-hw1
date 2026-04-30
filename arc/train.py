from __future__ import annotations

import argparse
from pathlib import Path

from mlp_hw1.data import create_data_bundle
from mlp_hw1.model import MLPClassifier
from mlp_hw1.trainer import TrainConfig, train_model
from mlp_hw1.utils import set_seed


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    default_dataset_root = (project_root.parent / "EuroSAT_RGB").resolve()
    default_output_dir = project_root / "artifacts" / "train"
    default_cache_path = project_root / "artifacts" / "cache" / "eurosat_rgb_uint8.npz"

    parser = argparse.ArgumentParser(
        description="Train a NumPy MLP from scratch on EuroSAT_RGB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--cache-path", type=Path, default=default_cache_path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    data_bundle = create_data_bundle(
        dataset_root=args.dataset_root,
        cache_path=args.cache_path,
        max_samples_per_class=args.max_samples_per_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(
        f"Loaded EuroSAT_RGB with train/val/test sizes: "
        f"{len(data_bundle.train)}/{len(data_bundle.val)}/{len(data_bundle.test)}"
    )

    model = MLPClassifier(
        input_dim=data_bundle.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=data_bundle.num_classes,
        activation=args.activation,
        seed=args.seed,
    )
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    result = train_model(
        model=model,
        data_bundle=data_bundle,
        config=config,
        output_dir=args.output_dir,
        run_metadata={
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "data_config": {
                "cache_path": str(Path(args.cache_path).resolve()),
                "max_samples_per_class": args.max_samples_per_class,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "seed": args.seed,
            },
        },
    )

    print(f"Best validation accuracy: {result['best_val_accuracy']:.4f} at epoch {result['best_epoch']}")
    print(f"Test accuracy: {result['test_accuracy']:.4f}")
    print(f"Checkpoint saved to: {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
