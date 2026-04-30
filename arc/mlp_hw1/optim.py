from __future__ import annotations

from .autograd import Tensor


class SGD:
    def __init__(self, parameters: list[Tensor], lr: float) -> None:
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            parameter.data -= self.lr * parameter.grad
