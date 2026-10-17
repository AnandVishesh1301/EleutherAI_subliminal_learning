"""Ported student optimizers: Muon with an AdamW auxiliary, Lion / signSGD, Adam-mini.

Sources: torch v2.9.0 ``torch/optim/_muon.py`` (Muon; absent from the pinned torch
2.6); Liu et al. arXiv:2502.16982 Eq. 4 (Muon update scale); google/automl
``lion/lion_pytorch.py`` and Chen et al. arXiv:2302.06675 Alg. 2 (Lion); Bernstein et
al. arXiv:1802.04434 Alg. 1 (signSGD); adam-mini 1.1.1 ``adam_mini.py`` and Zhang et
al. arXiv:2406.16793 Alg. 1 (Adam-mini). Line numbers below refer to those files.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor
from torch.optim.adamw import adamw

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)  # torch v2.9.0 _muon.py L25-27
NS_STEPS = 5
NS_EPS = 1e-7
MUON_MOMENTUM = 0.95
MUON_UPDATE_RMS = 0.2  # Moonlight Eq. 4 (torch "match_rms_adamw", L81)
LION_BETAS = (0.9, 0.99)
SIGNSGD_BETAS = (0.0, 0.0)
ADAM_MINI_BLOCKS = ("elementwise", "rows", "whole")


class NeoxRole(StrEnum):
    """Optimizer-relevant role of a trainable GPT-NeoX parameter."""

    LORA = "lora"
    EMBEDDING = "embedding"
    QKV = "qkv"
    MATRIX = "matrix"
    NORM = "norm"
    BIAS = "bias"


_NEOX_NORMS = ("input_layernorm", "post_attention_layernorm", "final_layer_norm")
_NEOX_MATRICES = (
    ("attention", "dense"),
    ("mlp", "dense_h_to_4h"),
    ("mlp", "dense_4h_to_h"),
)


def neox_param_role(name: str) -> NeoxRole:
    """Role of a GPT-NeoX parameter name, plain or PEFT-wrapped.

    Raises:
        ValueError: `name` is not a known GPT-NeoX parameter.
    """
    parts = name.replace(".modules_to_save.default", "").split(".")
    if "lora_A" in parts or "lora_B" in parts:
        return NeoxRole.LORA
    if parts[-1] == "bias":
        return NeoxRole.BIAS
    if parts[-1] == "weight" and len(parts) >= 2:
        if parts[-2] in ("embed_in", "embed_out"):
            return NeoxRole.EMBEDDING
        if parts[-2] in _NEOX_NORMS:
            return NeoxRole.NORM
        if tuple(parts[-3:-1]) == ("attention", "query_key_value"):
            return NeoxRole.QKV
        if tuple(parts[-3:-1]) in _NEOX_MATRICES:
            return NeoxRole.MATRIX
    raise ValueError(f"unknown GPT-NeoX parameter {name!r}")


def zeropower_newton_schulz(
    g: Tensor,
    steps: int = NS_STEPS,
    eps: float = NS_EPS,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Quintic Newton-Schulz orthogonalization of the trailing two dims of `g`.

    Each ``(A, B)`` matrix of a batch ``(..., A, B)`` is normalized by its own
    Frobenius norm, so the matrices are orthogonalized independently.

    Returns:
        Tensor of `g`'s shape in `dtype`.
    """
    a, b, c = NS_COEFFICIENTS
    x = g.to(dtype)
    tall = g.size(-2) > g.size(-1)
    if tall:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)  # torch L59
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x  # torch L62-66
    return x.mT if tall else x


def split_neox_qkv(w: Tensor, n_heads: int) -> Tensor:
    """Fused GPT-NeoX QKV ``(3·H·d, in)`` to stacked Q, K, V ``(3, H·d, in)``.

    Rows are interleaved per head, ``[h0: q k v, h1: q k v, ...]``
    (transformers 4.54 ``modeling_gpt_neox.py`` L154-157: ``view(..., -1, 3·d)``
    then ``chunk(3, dim=-1)``).

    Raises:
        ValueError: The row count is not a multiple of ``3 · n_heads``.
    """
    rows, cols = w.shape
    if n_heads < 1 or rows % (3 * n_heads):
        raise ValueError(f"QKV rows {rows} not divisible by 3 * {n_heads} heads")
    head_dim = rows // (3 * n_heads)
    return (
        w.reshape(n_heads, 3, head_dim, cols)
        .transpose(0, 1)
        .reshape(3, n_heads * head_dim, cols)
    )


def merge_neox_qkv(qkv: Tensor, n_heads: int) -> Tensor:
    """Inverse of `split_neox_qkv`."""
    _, rows, cols = qkv.shape
    return (
        qkv.reshape(3, n_heads, rows // n_heads, cols)
        .transpose(0, 1)
        .reshape(3 * rows, cols)
    )


def muon_update_scale(shape: tuple[int, ...]) -> float:
    """Moonlight ``0.2·√max(A, B)`` for one orthogonalized ``(A, B)`` matrix."""
    return MUON_UPDATE_RMS * math.sqrt(max(shape[-2:]))


def _with_group_defaults(
    params: Iterable[Mapping[str, Any]],
    defaults_of: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups = []
    for group in params:
        if not isinstance(group, Mapping):
            raise TypeError("params must be param-group dicts")
        groups.append({**defaults_of(group), **group})
    return groups


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """Muon on hidden matrices and torch AdamW on the rest, in one optimizer.

    Every param group sets ``use_muon``. Muon groups follow torch v2.9.0
    ``_single_tensor_muon`` with Moonlight's update scale (`muon_update_scale`); a
    group with ``qkv_heads > 0`` orthogonalizes the Q, K and V blocks of fused
    GPT-NeoX QKV matrices separately. Aux groups run `torch.optim.adamw.adamw` on
    AdamW's own state layout, so they step exactly like ``torch.optim.AdamW``.
    """

    def __init__(
        self,
        params: Iterable[Mapping[str, Any]],
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = MUON_MOMENTUM,
        nesterov: bool = True,
        ns_steps: int = NS_STEPS,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        muon = {
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "qkv_heads": 0,
        }
        aux = {"betas": tuple(betas), "eps": eps}

        def defaults_of(group: Mapping[str, Any]) -> Mapping[str, Any]:
            if "use_muon" not in group:
                raise ValueError("every MuonWithAuxAdamW param group must set use_muon")
            return muon if group["use_muon"] else aux

        groups = _with_group_defaults(params, defaults_of)
        super().__init__(groups, {"lr": lr, "weight_decay": weight_decay})
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon needs 2-D parameters, got {tuple(p.shape)}")
                heads = group["qkv_heads"]
                if heads and p.size(0) % (3 * heads):
                    raise ValueError(
                        f"QKV shape {tuple(p.shape)} not divisible by 3 * {heads} heads"
                    )

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict[str, Any]) -> None:
        lr, wd, mu = group["lr"], group["weight_decay"], group["momentum"]
        heads = group["qkv_heads"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(p.grad)
            buf = state["momentum_buffer"]
            buf.lerp_(p.grad, 1 - mu)  # torch _muon.py L312
            update = p.grad.lerp(buf, mu) if group["nesterov"] else buf  # L313
            if heads:
                update = split_neox_qkv(update, heads)
            ortho = zeropower_newton_schulz(update, group["ns_steps"])
            scale = muon_update_scale(tuple(ortho.shape))
            if heads:
                ortho = merge_neox_qkv(ortho, heads)
            if wd:
                p.mul_(1 - lr * wd)  # L319
            p.add_(ortho, alpha=-lr * scale)  # L320

    def _adamw_step(self, group: dict[str, Any]) -> None:
        params, grads, exp_avgs, exp_avg_sqs, steps = [], [], [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:  # torch 2.6 AdamW._init_group (capturable/fused off)
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
            params.append(p)
            grads.append(p.grad)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            steps.append(state["step"])
        if not params:
            return
        beta1, beta2 = group["betas"]
        adamw(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            [],
            steps,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=False,
        )


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. Alg. 2); ``betas[0] == 0`` is signSGD and keeps no state."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: float,
        betas: tuple[float, float] = LION_BETAS,
        weight_decay: float = 0.0,
    ) -> None:
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"Lion betas must lie in [0, 1), got {betas}")
        super().__init__(
            params, {"lr": lr, "betas": tuple(betas), "weight_decay": weight_decay}
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd:
                    p.mul_(1 - lr * wd)  # automl lion_pytorch.py L66
                if beta1 == 0:
                    p.add_(g.sign(), alpha=-lr)  # Bernstein et al. Alg. 1
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                p.add_((m * beta1 + g * (1 - beta1)).sign_(), alpha=-lr)  # L79-81
                m.mul_(beta2).add_(g, alpha=1 - beta2)  # L84
        return loss


class AdamMini(torch.optim.Optimizer):
    """Adam-mini: Adam with one second-moment value per parameter block.

    Every group sets ``block`` (`ADAM_MINI_BLOCKS`): ``elementwise`` is plain Adam,
    ``rows`` keeps one v per ``block_rows`` contiguous rows of a 2-D tensor, and
    ``whole`` one v per tensor.
    """

    def __init__(
        self,
        params: Iterable[Mapping[str, Any]],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        def defaults_of(group: Mapping[str, Any]) -> Mapping[str, Any]:
            if group.get("block") not in ADAM_MINI_BLOCKS:
                raise ValueError(
                    f"AdamMini group block must be one of {ADAM_MINI_BLOCKS}, "
                    f"got {group.get('block')!r}"
                )
            return {"block_rows": 1}

        groups = _with_group_defaults(params, defaults_of)
        super().__init__(
            groups,
            {"lr": lr, "betas": tuple(betas), "eps": eps, "weight_decay": weight_decay},
        )
        for group in self.param_groups:
            if group["block"] != "rows":
                continue
            for p in group["params"]:
                if p.ndim != 2 or p.size(0) % group["block_rows"]:
                    raise ValueError(
                        f"row blocks of {group['block_rows']} need a 2-D tensor with "
                        f"divisible rows, got {tuple(p.shape)}"
                    )

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            beta1, beta2 = group["betas"]
            block, k = group["block"], group["block_rows"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = _block_mean_sq(torch.zeros_like(p), block, k)
                state["step"] += 1
                state["v"].mul_(beta2).add_(
                    _block_mean_sq(g, block, k), alpha=1 - beta2
                )
                if wd:
                    p.mul_(1 - lr * wd)  # adam_mini.py L303-304
                m = state["m"].lerp_(g, 1 - beta1)  # L305
                bias1 = 1 - beta1 ** state["step"]
                bias2 = 1 - beta2 ** state["step"]
                h = (state["v"].sqrt() / math.sqrt(bias2)).add_(eps)  # L309
                if block == "rows":
                    update = (m.view(h.size(0), -1) / h).view_as(p)  # L310-311
                else:
                    update = m / h
                p.add_(update, alpha=-lr / bias1)  # L312-313
        return loss


def _block_mean_sq(g: Tensor, block: str, block_rows: int) -> Tensor:
    """Mean of g² per Adam-mini block: g²; ``(R / k, 1)`` row blocks; or a scalar."""
    if block == "elementwise":
        return g * g
    if block == "rows":
        return (g * g).view(g.size(0) // block_rows, -1).mean(dim=1, keepdim=True)
    return (g * g).mean()
