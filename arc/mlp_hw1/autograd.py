from __future__ import annotations

from typing import Iterable

import numpy as np


def _to_float_array(data: np.ndarray | float | int) -> np.ndarray:
    return np.asarray(data, dtype=np.float32)


def _sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if grad.shape == shape:
        return grad
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(
        self,
        data: np.ndarray | float | int,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
        name: str = "",
    ) -> None:
        self.data = _to_float_array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float32) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.name = name

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}, op={self._op!r})"

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

    def backward(self, grad: np.ndarray | None = None) -> None:
        if not self.requires_grad:
            raise ValueError("Cannot call backward on a tensor that does not require gradients.")
        if grad is None:
            if self.data.size != 1:
                raise ValueError("Non-scalar tensors require an explicit gradient.")
            grad = np.ones_like(self.data, dtype=np.float32)

        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(node: Tensor) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = _to_float_array(grad)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, requires_grad=requires_grad, _children=(self, other), _op="add")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return other + (-self)

    def __mul__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, requires_grad=requires_grad, _children=(self, other), _op="mul")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _sum_to_shape(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return self * other

    def __truediv__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1.0)

    def pow(self, exponent: float) -> "Tensor":
        out = Tensor(self.data**exponent, requires_grad=self.requires_grad, _children=(self,), _op=f"pow({exponent})")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (exponent * (self.data ** (exponent - 1.0))) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor" | np.ndarray) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, requires_grad=requires_grad, _children=(self, other), _op="matmul")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is None:
                expanded = np.ones_like(self.data, dtype=np.float32) * grad
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                normalized_axes = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in axes)
                expanded = grad
                if not keepdims:
                    for current_axis in sorted(normalized_axes):
                        expanded = np.expand_dims(expanded, axis=current_axis)
                expanded = np.broadcast_to(expanded, self.data.shape)
            self.grad += expanded

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            divisor = float(self.data.size)
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            divisor = float(np.prod([self.data.shape[ax] for ax in axes]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / divisor)

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), requires_grad=self.requires_grad, _children=(self,), _op="relu")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (self.data > 0.0).astype(np.float32) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        values = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(values, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += values * (1.0 - values) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        values = np.tanh(self.data)
        out = Tensor(values, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 - values**2) * out.grad

        out._backward = _backward
        return out


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    targets = np.asarray(targets, dtype=np.int64)
    shifted_logits = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    batch_indices = np.arange(targets.shape[0])
    sample_losses = -np.log(probabilities[batch_indices, targets] + 1e-12)
    out = Tensor(sample_losses.mean(), requires_grad=logits.requires_grad, _children=(logits,), _op="cross_entropy")

    def _backward() -> None:
        if not logits.requires_grad:
            return
        grad = probabilities
        grad[batch_indices, targets] -= 1.0
        grad /= float(targets.shape[0])
        logits.grad += grad * out.grad

    out._backward = _backward
    return out
