from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class ExplainEvalConfig:
    """
    Paper2 schema 对齐的最小解释评估配置。
    """

    k_fracs: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    stability_noise_std: float = 0.01
    stability_n_perturb: int = 5
    max_samples: int = 8
    sparsity_rule_threshold: float = 0.05


def _infer_time_channel_dims(x: torch.Tensor) -> Tuple[int, Optional[int]]:
    """
    返回 (time_dim, channel_dim)。若无法判断 channel_dim，则返回 None。
    约定：
    - (B, L, C) -> time_dim=1, channel_dim=2
    - (B, C, L) -> time_dim=2, channel_dim=1（当 C 很小而 L 很大时）
    """
    if x.dim() != 3:
        return 1, None

    b, d1, d2 = x.shape
    if d1 <= 16 and d2 > d1:
        return 2, 1
    return 1, 2


def _spearman_corr_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    简化 Spearman：对 rank 做 Pearson（不处理 ties 的精细化，足够用于趋势评估）。
    """
    a = a.flatten()
    b = b.flatten()
    if a.numel() < 2:
        return 0.0
    ra = torch.argsort(torch.argsort(a))
    rb = torch.argsort(torch.argsort(b))
    ra = ra.float()
    rb = rb.float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.std(unbiased=False) * rb.std(unbiased=False)).clamp_min(1e-8)
    corr = (ra * rb).mean() / denom
    corr_val = float(corr.detach().cpu().item())
    if corr_val != corr_val:  # NaN
        return 0.0
    return max(-1.0, min(1.0, corr_val))


def _grad_importance(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    使用输入梯度作为 attribution：
    - importance: (B, L) 归一化到每样本 sum=1
    - baseline_prob: (B,) 目标类别概率
    - time_ms_per_sample: float
    """
    model.eval()

    if x.dim() != 3:
        raise ValueError(f"Expected input x to be 3D (B,L,C or B,C,L), got shape={tuple(x.shape)}")

    x = x.detach()
    x = x.requires_grad_(True)

    with torch.enable_grad():
        t0 = time.perf_counter()
        logits = model(x)
        if logits.dim() != 2:
            raise ValueError(f"Model output must be (B,num_classes), got shape={tuple(logits.shape)}")

        if target is None:
            target = torch.argmax(logits, dim=1)
        target = target.to(logits.device)
        selected = logits.gather(1, target.view(-1, 1)).sum()
        selected.backward()
        t1 = time.perf_counter()

    grad = x.grad
    if grad is None:
        raise RuntimeError("Failed to compute input gradients (x.grad is None)")

    time_dim, channel_dim = _infer_time_channel_dims(x)
    if channel_dim is None:
        raise ValueError("Cannot infer channel dimension for grad attribution")

    importance = grad.detach().abs().sum(dim=channel_dim)  # (B, L)
    denom = importance.sum(dim=1, keepdim=True).clamp_min(1e-8)
    importance = importance / denom

    with torch.no_grad():
        probs = torch.softmax(logits.detach(), dim=1)
        baseline_prob = probs.gather(1, target.view(-1, 1)).squeeze(1)

    time_ms = (t1 - t0) * 1000.0 / max(1, int(x.shape[0]))
    return importance, baseline_prob, float(time_ms)


def _mask_topk_timesteps(x: torch.Tensor, importance: torch.Tensor, frac: float) -> torch.Tensor:
    """
    将每个样本 top-k(=frac) 的时间步置 0（所有通道同时置 0）。
    """
    if x.dim() != 3:
        raise ValueError("mask expects 3D x")
    if importance.dim() != 2:
        raise ValueError("importance must be (B,L)")

    frac = float(frac)
    frac = max(0.0, min(1.0, frac))
    if frac <= 0.0:
        return x.clone()

    time_dim, channel_dim = _infer_time_channel_dims(x)
    length = x.shape[time_dim]
    k = max(1, int(round(length * frac)))

    idx = torch.topk(importance, k=k, dim=1, largest=True).indices  # (B,k)
    x_masked = x.clone()

    # 按 batch 逐样本写入（避免复杂 gather/scatter）
    for i in range(x.shape[0]):
        if time_dim == 1:
            x_masked[i, idx[i], :] = 0.0
        else:
            x_masked[i, :, idx[i]] = 0.0
    return x_masked


def _faithfulness_aopc(
    model: torch.nn.Module,
    x: torch.Tensor,
    importance: torch.Tensor,
    target: torch.Tensor,
    baseline_prob: torch.Tensor,
    k_fracs: List[float],
) -> float:
    """
    Faithfulness (Del@k) 采用 AOPC 风格：删除 top-k 后概率下降的平均值（越大越好）。
    """
    model.eval()
    drops: List[float] = []
    with torch.no_grad():
        for frac in k_fracs:
            x_masked = _mask_topk_timesteps(x, importance, frac)
            logits = model(x_masked)
            probs = torch.softmax(logits, dim=1)
            masked_prob = probs.gather(1, target.view(-1, 1)).squeeze(1)
            drop = (baseline_prob - masked_prob).clamp_min(0.0)
            drops.append(float(drop.mean().detach().cpu().item()))
    if not drops:
        return 0.0
    return float(sum(drops) / len(drops))


def _stability_spearman(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    importance_ref: torch.Tensor,
    noise_std: float,
    n_perturb: int,
) -> float:
    """
    Stability@σ：对噪声扰动后的 attribution 与原 attribution 做 Spearman 相似度，取均值并映射到 [0,1]。
    """
    if n_perturb <= 0:
        return 0.0
    sims: List[float] = []
    for _ in range(int(n_perturb)):
        noise = torch.randn_like(x) * float(noise_std)
        x_noisy = (x + noise).detach()
        try:
            imp_noisy, _, _ = _grad_importance(model, x_noisy, target=target)
        except Exception:
            continue
        for i in range(x.shape[0]):
            corr = _spearman_corr_1d(importance_ref[i], imp_noisy[i])
            sims.append((corr + 1.0) / 2.0)  # [-1,1] -> [0,1]
    if not sims:
        return 0.0
    return float(sum(sims) / len(sims))


def _fuzzy_sparsity_rules(model: torch.nn.Module, x: torch.Tensor, threshold: float) -> Optional[Dict[str, float]]:
    """
    Fuzzy 模型 sparsity：平均激活规则数（越小越稀疏）。
    """
    get_rules = getattr(model, "get_rule_explanations", None)
    if get_rules is None or not callable(get_rules):
        return None
    with torch.no_grad():
        info = get_rules(x)
    if not isinstance(info, dict):
        return None
    strengths = info.get("rule_strengths")
    if strengths is None or not torch.is_tensor(strengths):
        return None

    thr = float(threshold)
    activated = (strengths > thr).sum(dim=1).float()
    return {
        "rules_activated_mean": float(activated.mean().detach().cpu().item()),
        "rule_threshold": thr,
        "num_rules": int(strengths.shape[1]),
    }


def eval_explainability_on_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    cfg: Optional[ExplainEvalConfig] = None,
) -> Dict[str, Any]:
    """
    在一个 batch 上做最小解释评估，并返回可直接写入 Paper2 schema 的 explainability 字段。
    """
    cfg = cfg or ExplainEvalConfig()

    if x.dim() != 3:
        raise ValueError(f"Expected x dim=3, got {x.dim()}")

    if x.shape[0] > cfg.max_samples:
        x = x[: cfg.max_samples]

    with torch.no_grad():
        logits = model(x)
        target = torch.argmax(logits, dim=1)

    importance, baseline_prob, time_ms = _grad_importance(model, x, target=target)
    faith = _faithfulness_aopc(
        model=model,
        x=x.detach(),
        importance=importance,
        target=target,
        baseline_prob=baseline_prob,
        k_fracs=cfg.k_fracs,
    )
    stab = _stability_spearman(
        model=model,
        x=x.detach(),
        target=target,
        importance_ref=importance,
        noise_std=cfg.stability_noise_std,
        n_perturb=cfg.stability_n_perturb,
    )

    out: Dict[str, Any] = {
        "faithfulness": {"del_k_auc": float(faith), "k_fracs": [float(v) for v in cfg.k_fracs]},
        "stability": {
            "spearman_mean": float(stab),
            "noise_std": float(cfg.stability_noise_std),
            "n_perturb": int(cfg.stability_n_perturb),
        },
        "efficiency": {"time_ms_per_sample": float(time_ms)},
        "method": {"attribution": "grad_abs_sum", "target": "predicted_class"},
    }

    sparsity = _fuzzy_sparsity_rules(model, x.detach(), threshold=cfg.sparsity_rule_threshold)
    if sparsity:
        out["sparsity"] = sparsity

    return out

