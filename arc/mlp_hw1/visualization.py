from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_accuracy"], label="train acc")
    axes[1].plot(epochs, history["val_accuracy"], label="val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_first_layer_weights(
    weight_matrix: np.ndarray,
    image_shape: tuple[int, int, int],
    output_path: str | Path,
    max_units: int = 36,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_units = min(weight_matrix.shape[1], max_units)
    rows = int(math.ceil(math.sqrt(num_units)))
    cols = int(math.ceil(num_units / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for unit_index in range(rows * cols):
        axis = axes.flat[unit_index]
        axis.axis("off")
        if unit_index >= num_units:
            continue
        weight_image = weight_matrix[:, unit_index].reshape(image_shape)
        min_value = float(weight_image.min())
        max_value = float(weight_image.max())
        normalized = (weight_image - min_value) / (max_value - min_value + 1e-8)
        axis.imshow(normalized)
        axis.set_title(f"Unit {unit_index}", fontsize=8)

    fig.suptitle("First-Layer Weights", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_misclassified_examples(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    max_examples: int = 16,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_examples = min(len(images), max_examples)
    rows = int(math.ceil(math.sqrt(max_examples)))
    cols = int(math.ceil(max_examples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for example_index in range(rows * cols):
        axis = axes.flat[example_index]
        axis.axis("off")
        if example_index >= num_examples:
            continue
        axis.imshow(images[example_index])
        axis.set_title(
            f"T:{class_names[int(true_labels[example_index])]}\nP:{class_names[int(pred_labels[example_index])]}",
            fontsize=8,
        )

    fig.suptitle("Misclassified Test Images", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path
