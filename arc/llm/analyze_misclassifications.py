from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ARC_ROOT = Path(__file__).resolve().parents[1]
if str(ARC_ROOT) not in sys.path:
    sys.path.insert(0, str(ARC_ROOT))

from mlp_hw1.autograd import Tensor
from mlp_hw1.data import EuroSATDataBundle, create_data_bundle
from mlp_hw1.metrics import confusion_matrix
from mlp_hw1.model import MLPClassifier
from mlp_hw1.serialization import load_checkpoint
from mlp_hw1.trainer import evaluate_model
from mlp_hw1.utils import save_json

from openai_client import DEFAULT_CONFIG_PATH, create_vision_json_completion


REASON_OPTIONS = [
    "Visual Similarity",
    "Mixed Land Cover",
    "Low Discriminative Features",
    "Texture vs Structure Confusion",
    "Model Limitation",
]


SYSTEM_PROMPT = (
    "You are helping analyze why a simple MLP misclassified EuroSAT remote-sensing images. "
    "The model only sees flattened RGB pixels, so lost spatial structure is an important clue. "
    "Return valid JSON only, with no Markdown fences."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare best-model artifacts, sample 100 misclassified images, and analyze them with a multimodal LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=ARC_ROOT / "artifacts" / "output_search" / "search_5" / "run_05",
        help="Final selected run directory. The default uses the best validation result under the chosen 40-epoch setting.",
    )
    parser.add_argument(
        "--best-model-dir",
        type=Path,
        default=ARC_ROOT / "artifacts" / "best_model",
        help="Output directory for the copied checkpoint, sampled errors, and LLM analysis results.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="LLM API config JSON path.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Analyze at most this many sampled errors.")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare best_model artifacts and sample errors.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip best_model preparation and only run LLM analysis.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-sample analysis files.")
    return parser


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "sample"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _build_model(metadata: dict) -> MLPClassifier:
    model_config = metadata["model"]
    return MLPClassifier(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_classes=model_config["num_classes"],
        activation=model_config["activation"],
        seed=model_config["seed"],
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _load_dataset_records(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_dirs = sorted([path for path in dataset_root.iterdir() if path.is_dir()])
    class_names = [path.name for path in class_dirs]
    image_paths: list[Path] = []
    labels: list[int] = []

    for class_index, class_dir in enumerate(class_dirs):
        for image_file in sorted(class_dir.glob("*.jpg")):
            image_paths.append(image_file.resolve())
            labels.append(class_index)

    return np.asarray(image_paths, dtype=object), np.asarray(labels, dtype=np.int64), class_names


def _subsample_records(
    image_paths: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    max_samples_per_class: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples_per_class is None:
        return image_paths, labels

    rng = np.random.default_rng(seed)
    selected_indices: list[np.ndarray] = []
    for class_index in range(num_classes):
        class_indices = np.where(labels == class_index)[0]
        rng.shuffle(class_indices)
        selected_indices.append(class_indices[:max_samples_per_class])

    merged = np.concatenate(selected_indices)
    merged.sort()
    return image_paths[merged], labels[merged]


def _split_indices(
    labels: np.ndarray,
    num_classes: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for class_index in range(num_classes):
        class_indices = np.where(labels == class_index)[0]
        rng.shuffle(class_indices)

        class_count = len(class_indices)
        train_end = int(class_count * train_ratio)
        val_end = train_end + int(class_count * val_ratio)

        train_parts.append(class_indices[:train_end])
        val_parts.append(class_indices[train_end:val_end])
        test_parts.append(class_indices[val_end:])

    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    test_indices = np.concatenate(test_parts)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def _resolve_data_bundle(metadata: dict) -> tuple[EuroSATDataBundle, dict, Path]:
    run_metadata = metadata.get("run_metadata", {})
    data_config = run_metadata.get("data_config", {})
    dataset_root = Path(
        run_metadata.get("dataset_root")
        or (ARC_ROOT.parent / "EuroSAT_RGB")
    ).resolve()

    cache_path_value = data_config.get("cache_path")
    cache_path = Path(cache_path_value).resolve() if cache_path_value else None
    resolved = {
        "dataset_root": str(dataset_root),
        "cache_path": str(cache_path) if cache_path else None,
        "max_samples_per_class": data_config.get("max_samples_per_class"),
        "train_ratio": data_config.get("train_ratio", 0.7),
        "val_ratio": data_config.get("val_ratio", 0.15),
        "test_ratio": data_config.get("test_ratio", 0.15),
        "seed": data_config.get("seed", metadata.get("training", {}).get("seed", 42)),
    }

    bundle = create_data_bundle(
        dataset_root=dataset_root,
        cache_path=cache_path,
        max_samples_per_class=resolved["max_samples_per_class"],
        train_ratio=resolved["train_ratio"],
        val_ratio=resolved["val_ratio"],
        test_ratio=resolved["test_ratio"],
        seed=resolved["seed"],
    )
    return bundle, resolved, dataset_root


def _copy_run_artifacts(source_run: Path, best_model_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in [
        "best_model.npz",
        "summary.json",
        "history.json",
        "training_curves.png",
        "confusion_matrix.png",
        "first_layer_weights.png",
        "misclassified_examples.png",
    ]:
        source = source_run / name
        if source.exists():
            destination = best_model_dir / name
            shutil.copy2(source, destination)
            copied.append(str(destination.resolve()))
    return copied


def _top_confusions(matrix: np.ndarray, class_names: list[str], limit: int = 8) -> list[dict]:
    rows: list[dict] = []
    for true_index in range(matrix.shape[0]):
        for pred_index in range(matrix.shape[1]):
            if true_index == pred_index:
                continue
            count = int(matrix[true_index, pred_index])
            if count <= 0:
                continue
            rows.append(
                {
                    "true_label": class_names[true_index],
                    "predicted_label": class_names[pred_index],
                    "count": count,
                }
            )
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows[:limit]


def _class_accuracy_rows(matrix: np.ndarray, class_names: list[str]) -> list[dict]:
    rows: list[dict] = []
    for index, class_name in enumerate(class_names):
        total = int(matrix[index].sum())
        correct = int(matrix[index, index])
        accuracy = float(correct / total) if total else 0.0
        rows.append(
            {
                "class_name": class_name,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }
        )
    return rows


def _collect_misclassified_records(
    model: MLPClassifier,
    data_bundle: EuroSATDataBundle,
    test_paths: np.ndarray,
    batch_size: int,
) -> list[dict]:
    records: list[dict] = []
    split = data_bundle.test

    for batch_indices in split.batch_indices(batch_size=batch_size, shuffle=False):
        batch_x, batch_y = split.batch_arrays(batch_indices, flatten=True)
        logits = model(Tensor(batch_x))
        probabilities = _softmax(logits.data)
        batch_pred = probabilities.argmax(axis=1)
        wrong_positions = np.where(batch_pred != batch_y)[0]

        for local_index in wrong_positions.tolist():
            split_index = int(batch_indices[local_index])
            true_label_id = int(batch_y[local_index])
            pred_label_id = int(batch_pred[local_index])
            records.append(
                {
                    "split_index": split_index,
                    "dataset_image_path": str(Path(test_paths[split_index]).resolve()),
                    "true_label_id": true_label_id,
                    "true_label": data_bundle.class_names[true_label_id],
                    "predicted_label_id": pred_label_id,
                    "predicted_label": data_bundle.class_names[pred_label_id],
                    "predicted_confidence": float(probabilities[local_index, pred_label_id]),
                    "true_label_confidence": float(probabilities[local_index, true_label_id]),
                }
            )
    return records


def prepare_best_model_bundle(
    source_run: Path,
    best_model_dir: Path,
    sample_size: int,
    sample_seed: int,
) -> dict:
    source_run = source_run.resolve()
    best_model_dir = best_model_dir.resolve()
    best_model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = source_run / "best_model.npz"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _copy_run_artifacts(source_run, best_model_dir)
    state_dict, metadata = load_checkpoint(checkpoint_path)
    data_bundle, data_config, dataset_root = _resolve_data_bundle(metadata)

    model = _build_model(metadata)
    model.load_state_dict(state_dict)
    batch_size = int(metadata.get("training", {}).get("batch_size", 128))
    metrics = evaluate_model(model, data_bundle.test, batch_size=batch_size)
    matrix = confusion_matrix(metrics["y_true"], metrics["y_pred"], data_bundle.num_classes)

    image_paths, labels, class_names = _load_dataset_records(dataset_root)
    if class_names != data_bundle.class_names:
        raise ValueError("Dataset class order does not match checkpoint metadata.")

    image_paths, labels = _subsample_records(
        image_paths=image_paths,
        labels=labels,
        num_classes=len(class_names),
        max_samples_per_class=data_config["max_samples_per_class"],
        seed=data_config["seed"],
    )

    _, _, test_indices = _split_indices(
        labels=labels,
        num_classes=len(class_names),
        train_ratio=data_config["train_ratio"],
        val_ratio=data_config["val_ratio"],
        test_ratio=data_config["test_ratio"],
        seed=data_config["seed"],
    )
    test_paths = image_paths[test_indices]

    if len(test_paths) != len(data_bundle.test):
        raise ValueError("Recovered test-set file list does not match test split length.")

    misclassified = _collect_misclassified_records(
        model=model,
        data_bundle=data_bundle,
        test_paths=test_paths,
        batch_size=batch_size,
    )

    sample_count = min(sample_size, len(misclassified))
    rng = random.Random(sample_seed)
    sampled = rng.sample(misclassified, k=sample_count)

    samples_dir = best_model_dir / "misclassified_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(sampled, start=1):
        source_image = Path(record["dataset_image_path"])
        ext = source_image.suffix.lower() or ".jpg"
        sample_id = f"S{index:03d}"
        filename = (
            f"{sample_id}_{_safe_name(record['true_label'])}_as_{_safe_name(record['predicted_label'])}{ext}"
        )
        copied_image = samples_dir / filename
        shutil.copy2(source_image, copied_image)

        record["sample_id"] = sample_id
        record["dataset_image_relpath"] = str(source_image.relative_to(dataset_root))
        record["copied_image_path"] = str(copied_image.resolve())

    sampled.sort(key=lambda row: row["sample_id"])
    analysis_dir = best_model_dir / "llm_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "source_run": str(source_run),
        "selection_rule": "Best validation-accuracy checkpoint under the final 40-epoch setting.",
        "dataset_root": str(dataset_root),
        "training": metadata.get("training", {}),
        "model": metadata.get("model", {}),
        "results": {
            **metadata.get("results", {}),
            "test_loss": float(metrics["loss"]),
            "test_accuracy": float(metrics["accuracy"]),
            "misclassified_total": len(misclassified),
            "sampled_misclassified": sample_count,
        },
        "class_accuracy": _class_accuracy_rows(matrix, data_bundle.class_names),
        "top_confusions": _top_confusions(matrix, data_bundle.class_names),
    }

    save_json(best_model_dir / "best_model_summary.json", summary_payload)
    save_json(best_model_dir / "misclassified_samples.json", sampled)
    _write_jsonl(best_model_dir / "misclassified_samples.jsonl", sampled)
    _write_csv(
        best_model_dir / "misclassified_samples.csv",
        [
            {
                "sample_id": row["sample_id"],
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "predicted_confidence": row["predicted_confidence"],
                "true_label_confidence": row["true_label_confidence"],
                "dataset_image_relpath": row["dataset_image_relpath"],
                "copied_image_path": row["copied_image_path"],
            }
            for row in sampled
        ],
    )
    return summary_payload


def _normalize_reason(reason: str) -> str:
    reason = (reason or "").strip().lower()
    for option in REASON_OPTIONS:
        if reason == option.lower():
            return option
    return "Model Limitation"


def _build_prompt(sample: dict) -> str:
    reason_lines = "\n".join(f"- {option}" for option in REASON_OPTIONS)
    return f"""Please analyze why a simple MLP classifier made this error on a EuroSAT image.

Ground-truth label: {sample['true_label']}
MLP predicted label: {sample['predicted_label']}
Predicted confidence: {sample['predicted_confidence']:.4f}
Confidence assigned to the true class: {sample['true_label_confidence']:.4f}

Choose one primary reason from the following list:
{reason_lines}

Definitions:
- Visual Similarity: the true and predicted classes look globally similar in color or shape.
- Mixed Land Cover: the image contains multiple land-cover types and the dominant class is ambiguous.
- Low Discriminative Features: the image lacks strong visual cues, contrast, or unique objects.
- Texture vs Structure Confusion: local texture may look similar, but the global spatial arrangement matters.
- Model Limitation: the error is mainly caused by the weakness of a flattened-pixel MLP rather than a specific visual cue.

Return JSON only with this schema:
{{
  "primary_reason": "one item from the list above",
  "secondary_reasons": ["zero or more items from the same list"],
  "confidence": 0.0,
  "short_explanation": "1-3 concise sentences",
  "evidence": ["short point 1", "short point 2"],
  "needs_human_review": false
}}"""


def analyze_samples(
    best_model_dir: Path,
    config_path: Path,
    limit: int | None,
    force: bool,
) -> dict:
    best_model_dir = best_model_dir.resolve()
    samples = _read_json(best_model_dir / "misclassified_samples.json")
    if limit is not None:
        samples = samples[:limit]

    analysis_dir = best_model_dir / "llm_analysis"
    per_sample_dir = analysis_dir / "per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for sample in samples:
        output_path = per_sample_dir / f"{sample['sample_id']}.json"
        if output_path.exists() and not force:
            result_payload = _read_json(output_path)
            results.append(result_payload)
            print(f"Skipping existing analysis for {sample['sample_id']}")
            continue

        response = create_vision_json_completion(
            image_path=sample["copied_image_path"],
            prompt=_build_prompt(sample),
            system_prompt=SYSTEM_PROMPT,
            config_path=config_path,
        )
        parsed = response["parsed_json"]
        result_payload = {
            "sample_id": sample["sample_id"],
            "true_label": sample["true_label"],
            "predicted_label": sample["predicted_label"],
            "predicted_confidence": sample["predicted_confidence"],
            "true_label_confidence": sample["true_label_confidence"],
            "copied_image_path": sample["copied_image_path"],
            "primary_reason": _normalize_reason(parsed.get("primary_reason", "")),
            "secondary_reasons": [
                _normalize_reason(reason)
                for reason in parsed.get("secondary_reasons", [])
                if isinstance(reason, str)
            ],
            "confidence": float(parsed.get("confidence", 0.0)),
            "short_explanation": str(parsed.get("short_explanation", "")).strip(),
            "evidence": [str(item).strip() for item in parsed.get("evidence", []) if str(item).strip()],
            "needs_human_review": bool(parsed.get("needs_human_review", False)),
            "raw_model_response": response["message_text"],
        }
        save_json(output_path, result_payload)
        results.append(result_payload)
        print(
            f"Analyzed {sample['sample_id']}: "
            f"{sample['true_label']} -> {sample['predicted_label']} | {result_payload['primary_reason']}"
        )

    primary_counter = Counter(result["primary_reason"] for result in results)
    secondary_counter = Counter(reason for result in results for reason in result["secondary_reasons"])
    pair_counter = Counter(f"{result['true_label']} -> {result['predicted_label']}" for result in results)

    summary_payload = {
        "reason_options": REASON_OPTIONS,
        "num_samples_analyzed": len(results),
        "primary_reason_counts": dict(primary_counter.most_common()),
        "secondary_reason_counts": dict(secondary_counter.most_common()),
        "confusion_pair_counts": dict(pair_counter.most_common()),
        "average_confidence": float(np.mean([result["confidence"] for result in results])) if results else 0.0,
        "needs_human_review_count": int(sum(1 for result in results if result["needs_human_review"])),
    }

    save_json(analysis_dir / "analysis_summary.json", summary_payload)
    save_json(analysis_dir / "analysis_results.json", results)
    _write_jsonl(analysis_dir / "analysis_results.jsonl", results)
    _write_csv(
        analysis_dir / "analysis_results.csv",
        [
            {
                "sample_id": result["sample_id"],
                "true_label": result["true_label"],
                "predicted_label": result["predicted_label"],
                "primary_reason": result["primary_reason"],
                "secondary_reasons": "; ".join(result["secondary_reasons"]),
                "confidence": result["confidence"],
                "needs_human_review": result["needs_human_review"],
                "short_explanation": result["short_explanation"],
            }
            for result in results
        ],
    )
    return summary_payload


def main() -> None:
    args = build_parser().parse_args()

    if not args.skip_prepare:
        summary = prepare_best_model_bundle(
            source_run=args.source_run,
            best_model_dir=args.best_model_dir,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
        )
        print(
            "Prepared best_model bundle: "
            f"val={summary['results'].get('best_val_accuracy', 0.0):.4f}, "
            f"test={summary['results'].get('test_accuracy', 0.0):.4f}, "
            f"misclassified_total={summary['results'].get('misclassified_total', 0)}"
        )

    if args.prepare_only:
        return

    llm_summary = analyze_samples(
        best_model_dir=args.best_model_dir,
        config_path=args.config,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps(llm_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
