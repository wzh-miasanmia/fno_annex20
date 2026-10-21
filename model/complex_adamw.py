"""AdamW with a magnitude-based second moment for complex parameters.

PyTorch's stock Adam/AdamW treats a complex parameter as two independent real
coordinates.  Fourier weights need a shared |g|^2 second moment, as used by the
reference Geo-FNO optimizer and the current NeuralOperator implementation.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


class ComplexAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta values: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("ComplexAdamW does not support sparse gradients.")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    second_dtype = parameter.real.dtype if parameter.is_complex() else parameter.dtype
                    state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=second_dtype)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])

                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                if gradient.is_complex():
                    gradient_square = (gradient.conj() * gradient).real
                else:
                    gradient_square = gradient.square()
                exp_avg_sq.mul_(beta2).add_(gradient_square, alpha=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["eps"])
                parameter.addcdiv_(exp_avg, denominator, value=-group["lr"] / bias_correction1)
        return loss
