from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class EuroSATSplit:
    images: np.ndarray
    labels: np.ndarray
    class_names: list[str]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def input_dim(self) -> int:
        return int(np.prod(self.images.shape[1:]))

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return tuple(self.images.shape[1:])

    def batch_indices(self, batch_size: int, shuffle: bool = False, seed: int = 42):
        indices = np.arange(len(self))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield indices[start : start + batch_size]

    def batch_arrays(self, batch_indices: np.ndarray, flatten: bool = True) -> tuple[np.ndarray, np.ndarray]:
        batch_images = self.images[batch_indices].astype(np.float32) / 255.0
        if flatten:
            batch_images = batch_images.reshape(batch_images.shape[0], -1)
        batch_labels = self.labels[batch_indices].astype(np.int64)
        return batch_images, batch_labels


@dataclass
class EuroSATDataBundle:
    train: EuroSATSplit
    val: EuroSATSplit
    test: EuroSATSplit
    class_names: list[str]

    @property
    def input_dim(self) -> int:
        return self.train.input_dim

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return self.train.image_shape


def _load_or_build_cache(dataset_root: Path, cache_path: Path | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if cache_path is not None and cache_path.exists():
        archive = np.load(cache_path, allow_pickle=True)
        images = archive["images"]
        labels = archive["labels"]
        class_names = archive["class_names"].tolist()
        return images, labels, class_names

    class_dirs = sorted([path for path in dataset_root.iterdir() if path.is_dir()])
    class_names = [path.name for path in class_dirs]
    images_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for class_index, class_dir in enumerate(class_dirs):
        image_files = sorted(class_dir.glob("*.jpg"))
        class_images = []
        for image_file in image_files:
            with Image.open(image_file) as image:
                class_images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        images_list.append(np.stack(class_images))
        labels_list.append(np.full(len(class_images), class_index, dtype=np.int64))

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, images=images, labels=labels, class_names=np.array(class_names, dtype=object))

    return images, labels, class_names


def _subsample_per_class(
    images: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    max_samples_per_class: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples_per_class is None:
        return images, labels

    rng = np.random.default_rng(seed)
    selected_indices: list[np.ndarray] = []
    for class_index in range(num_classes):
        class_indices = np.where(labels == class_index)[0]
        rng.shuffle(class_indices)
        selected_indices.append(class_indices[:max_samples_per_class])

    merged = np.concatenate(selected_indices)
    merged.sort()
    return images[merged], labels[merged]


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


def create_data_bundle(
    dataset_root: str | Path,
    cache_path: str | Path | None = None,
    max_samples_per_class: int | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> EuroSATDataBundle:
    dataset_root = Path(dataset_root).resolve()
    cache = Path(cache_path).resolve() if cache_path is not None else None

    images, labels, class_names = _load_or_build_cache(dataset_root, cache)
    images, labels = _subsample_per_class(images, labels, len(class_names), max_samples_per_class, seed)

    train_indices, val_indices, test_indices = _split_indices(
        labels=labels,
        num_classes=len(class_names),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train = EuroSATSplit(images=images[train_indices], labels=labels[train_indices], class_names=class_names)
    val = EuroSATSplit(images=images[val_indices], labels=labels[val_indices], class_names=class_names)
    test = EuroSATSplit(images=images[test_indices], labels=labels[test_indices], class_names=class_names)
    return EuroSATDataBundle(train=train, val=val, test=test, class_names=class_names)
