from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from mlp_hw1.data import create_data_bundle
from mlp_hw1.model import MLPClassifier
from mlp_hw1.trainer import TrainConfig, train_model
from mlp_hw1.utils import save_json, set_seed, write_csv


def _parse_list(raw_value: str, cast):
    return [cast(item.strip()) for item in raw_value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    default_dataset_root = (project_root.parent / "EuroSAT_RGB").resolve()
    default_output_dir = project_root / "artifacts" / "search"
    default_cache_path = project_root / "artifacts" / "cache" / "eurosat_rgb_uint8.npz"

    parser = argparse.ArgumentParser(
        description="Grid search or random search over EuroSAT MLP hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--cache-path", type=Path, default=default_cache_path)
    parser.add_argument("--mode", choices=["grid", "random"], default="grid")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--learning-rates", type=str, default="0.1,0.05")
    parser.add_argument("--hidden-dims", type=str, default="128,256")
    parser.add_argument("--weight-decays", type=str, default="0.0,0.0001")
    parser.add_argument("--activations", type=str, default="relu,tanh")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-random-runs", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    learning_rates = _parse_list(args.learning_rates, float)
    hidden_dims = _parse_list(args.hidden_dims, int)
    weight_decays = _parse_list(args.weight_decays, float)
    activations = _parse_list(args.activations, str)

    all_configs = list(itertools.product(learning_rates, hidden_dims, weight_decays, activations))
    if args.mode == "random":
        rng = np.random.default_rng(args.seed)
        chosen_indices = rng.choice(len(all_configs), size=min(args.num_random_runs, len(all_configs)), replace=False)
        configs = [all_configs[index] for index in chosen_indices]
    else:
        configs = all_configs

    data_bundle = create_data_bundle(
        dataset_root=args.dataset_root,
        cache_path=args.cache_path,
        max_samples_per_class=args.max_samples_per_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for run_index, (learning_rate, hidden_dim, weight_decay, activation) in enumerate(configs, start=1):
        run_dir = output_dir / f"run_{run_index:02d}"
        print(
            f"Search run {run_index}/{len(configs)} | "
            f"lr={learning_rate} hidden_dim={hidden_dim} weight_decay={weight_decay} activation={activation}"
        )

        model = MLPClassifier(
            input_dim=data_bundle.input_dim,
            hidden_dim=hidden_dim,
            num_classes=data_bundle.num_classes,
            activation=activation,
            seed=args.seed + run_index,
        )
        config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            lr_decay=args.lr_decay,
            weight_decay=weight_decay,
            seed=args.seed + run_index,
        )
        result = train_model(
            model=model,
            data_bundle=data_bundle,
            config=config,
            output_dir=run_dir,
            run_metadata={
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "search_mode": args.mode,
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
        results.append(
            {
                "run": run_index,
                "learning_rate": learning_rate,
                "hidden_dim": hidden_dim,
                "weight_decay": weight_decay,
                "activation": activation,
                "best_epoch": result["best_epoch"],
                "best_val_accuracy": result["best_val_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "checkpoint_path": str(result["checkpoint_path"].resolve()),
            }
        )

    results.sort(key=lambda row: row["best_val_accuracy"], reverse=True)
    write_csv(output_dir / "search_results.csv", results)
    save_json(output_dir / "best_config.json", results[0])

    print("Top configuration:")
    print(results[0])


if __name__ == "__main__":
    main()
