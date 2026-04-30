from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .autograd import Tensor, cross_entropy_loss
from .data import EuroSATDataBundle, EuroSATSplit
from .metrics import accuracy_score, confusion_matrix
from .optim import SGD
from .serialization import save_checkpoint
from .utils import ensure_dir, save_json
from .visualization import plot_confusion_matrix, plot_first_layer_weights, plot_misclassified_examples, plot_training_curves


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.1
    lr_decay: float = 0.95
    weight_decay: float = 1e-4
    seed: int = 42


def _l2_penalty(parameters: list[Tensor], weight_decay: float) -> Tensor:
    if weight_decay <= 0.0:
        return Tensor(0.0)
    penalty = None
    for parameter in parameters:
        term = (parameter * parameter).sum()
        penalty = term if penalty is None else penalty + term
    return penalty * (0.5 * weight_decay)


def evaluate_model(model, split: EuroSATSplit, batch_size: int) -> dict:
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for batch_indices in split.batch_indices(batch_size=batch_size, shuffle=False):
        batch_x, batch_y = split.batch_arrays(batch_indices, flatten=True)
        logits = model(Tensor(batch_x))
        loss = cross_entropy_loss(logits, batch_y)
        pred = logits.data.argmax(axis=1)

        losses.append(float(loss.data) * len(batch_y))
        predictions.append(pred)
        labels.append(batch_y)

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    average_loss = float(np.sum(losses) / len(y_true))
    average_accuracy = accuracy_score(y_true, y_pred)
    return {
        "loss": average_loss,
        "accuracy": average_accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _collect_misclassified(model, split: EuroSATSplit, batch_size: int, limit: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    true_labels: list[int] = []
    pred_labels: list[int] = []

    for batch_indices in split.batch_indices(batch_size=batch_size, shuffle=False):
        batch_x, batch_y = split.batch_arrays(batch_indices, flatten=True)
        logits = model(Tensor(batch_x))
        batch_pred = logits.data.argmax(axis=1)
        wrong_mask = batch_pred != batch_y

        if wrong_mask.any():
            raw_images = split.images[batch_indices][wrong_mask]
            wrong_true = batch_y[wrong_mask]
            wrong_pred = batch_pred[wrong_mask]
            images.extend(list(raw_images))
            true_labels.extend(wrong_true.tolist())
            pred_labels.extend(wrong_pred.tolist())

        if len(images) >= limit:
            break

    if not images:
        return (
            np.empty((0, *split.image_shape), dtype=np.uint8),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.stack(images[:limit]),
        np.asarray(true_labels[:limit], dtype=np.int64),
        np.asarray(pred_labels[:limit], dtype=np.int64),
    )


def train_model(
    model,
    data_bundle: EuroSATDataBundle,
    config: TrainConfig,
    output_dir: str | Path,
    run_metadata: dict | None = None,
) -> dict:
    output_dir = ensure_dir(output_dir)
    optimizer = SGD(model.parameters(), lr=config.learning_rate)
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_state = model.state_dict()
    best_epoch = 0
    best_val_accuracy = -1.0
    checkpoint_path = output_dir / "best_model.npz"

    for epoch in range(1, config.epochs + 1):
        current_lr = config.learning_rate * (config.lr_decay ** (epoch - 1))
        optimizer.set_lr(current_lr)

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for batch_indices in data_bundle.train.batch_indices(batch_size=config.batch_size, shuffle=True, seed=config.seed + epoch):
            batch_x, batch_y = data_bundle.train.batch_arrays(batch_indices, flatten=True)
            logits = model(Tensor(batch_x))
            ce_loss = cross_entropy_loss(logits, batch_y)
            reg_loss = _l2_penalty(model.parameters(), config.weight_decay)
            loss = ce_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = logits.data.argmax(axis=1)
            running_loss += float(ce_loss.data) * len(batch_y)
            running_correct += int((predictions == batch_y).sum())
            running_samples += len(batch_y)

        train_loss = running_loss / running_samples
        train_accuracy = running_correct / running_samples
        val_metrics = evaluate_model(model, data_bundle.val, config.batch_size)

        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_accuracy))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))

        print(
            f"Epoch {epoch:03d} | lr={current_lr:.5f} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_epoch = epoch
            best_state = model.state_dict()
            preview_metadata = {
                "model": {
                    "input_dim": model.input_dim,
                    "hidden_dim": model.hidden_dim,
                    "num_classes": model.num_classes,
                    "activation": model.activation,
                    "seed": model.seed,
                },
                "training": asdict(config),
                "data": {
                    "class_names": data_bundle.class_names,
                    "image_shape": list(data_bundle.image_shape),
                },
                "progress": {
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_accuracy,
                },
            }
            save_checkpoint(checkpoint_path, best_state, preview_metadata)

    model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, data_bundle.test, config.batch_size)
    matrix = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], data_bundle.num_classes)
    wrong_images, wrong_true, wrong_pred = _collect_misclassified(model, data_bundle.test, config.batch_size)

    final_metadata = {
        "model": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "num_classes": model.num_classes,
            "activation": model.activation,
            "seed": model.seed,
        },
        "training": asdict(config),
        "data": {
            "class_names": data_bundle.class_names,
            "image_shape": list(data_bundle.image_shape),
        },
        "results": {
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "test_accuracy": float(test_metrics["accuracy"]),
        },
        "history": history,
    }
    if run_metadata:
        final_metadata["run_metadata"] = run_metadata

    save_checkpoint(checkpoint_path, model.state_dict(), final_metadata)
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "summary.json", final_metadata)
    plot_training_curves(history, output_dir / "training_curves.png")
    plot_confusion_matrix(matrix, data_bundle.class_names, output_dir / "confusion_matrix.png")
    plot_first_layer_weights(model.fc1.weight.data, data_bundle.image_shape, output_dir / "first_layer_weights.png")
    if len(wrong_images) > 0:
        plot_misclassified_examples(
            wrong_images,
            wrong_true,
            wrong_pred,
            data_bundle.class_names,
            output_dir / "misclassified_examples.png",
        )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_loss": float(test_metrics["loss"]),
        "confusion_matrix": matrix,
        "checkpoint_path": checkpoint_path,
    }
