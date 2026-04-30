from __future__ import annotations

import math

import numpy as np

from .autograd import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator, scale: float, prefix: str) -> None:
        weight = rng.standard_normal((in_features, out_features), dtype=np.float32) * scale
        bias = np.zeros((1, out_features), dtype=np.float32)
        self.weight = Tensor(weight, requires_grad=True, name=f"{prefix}.weight")
        self.bias = Tensor(bias, requires_grad=True, name=f"{prefix}.bias")

    def __call__(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class MLPClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        activation = activation.lower()
        if activation not in {"relu", "sigmoid", "tanh"}:
            raise ValueError("activation must be one of: relu, sigmoid, tanh")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.activation = activation
        self.seed = seed

        rng = np.random.default_rng(seed)
        first_scale = math.sqrt(2.0 / input_dim) if activation == "relu" else math.sqrt(1.0 / input_dim)
        second_scale = math.sqrt(1.0 / hidden_dim)
        self.fc1 = Linear(input_dim, hidden_dim, rng, first_scale, prefix="fc1")
        self.fc2 = Linear(hidden_dim, num_classes, rng, second_scale, prefix="fc2")

    def _activate(self, x: Tensor) -> Tensor:
        if self.activation == "relu":
            return x.relu()
        if self.activation == "sigmoid":
            return x.sigmoid()
        return x.tanh()

    def __call__(self, x: Tensor) -> Tensor:
        hidden = self._activate(self.fc1(x))
        return self.fc2(hidden)

    def parameters(self) -> list[Tensor]:
        return [*self.fc1.parameters(), *self.fc2.parameters()]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "fc1.weight": self.fc1.weight.data.copy(),
            "fc1.bias": self.fc1.bias.data.copy(),
            "fc2.weight": self.fc2.weight.data.copy(),
            "fc2.bias": self.fc2.bias.data.copy(),
        }

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        self.fc1.weight.data = state_dict["fc1.weight"].astype(np.float32).copy()
        self.fc1.bias.data = state_dict["fc1.bias"].astype(np.float32).copy()
        self.fc2.weight.data = state_dict["fc2.weight"].astype(np.float32).copy()
        self.fc2.bias.data = state_dict["fc2.bias"].astype(np.float32).copy()
        self.zero_grad()
