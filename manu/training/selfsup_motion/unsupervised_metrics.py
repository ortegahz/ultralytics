#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unsupervised saliency fitness and collapse checks."""

from __future__ import annotations

import torch


def saliency_fitness(saliency, alpha=0.25, beta=0.25, top_fraction=0.001):
    values = saliency.flatten(1)
    top_count = max(1, int(values.shape[1] * top_fraction))
    top = values.topk(top_count, dim=1).values.mean()
    mean = values.mean()
    threshold = values.mean(dim=1, keepdim=True) + values.std(dim=1, keepdim=True)
    background = values[values < threshold.expand_as(values)]
    variance = background.var(unbiased=False) if background.numel() else values.var(unbiased=False)
    if not torch.isfinite(top + mean + variance) or float(values.max()) < 0.05 or float(mean) > 0.15:
        return torch.full((), -100.0, device=saliency.device), True
    return top - alpha * mean - beta * variance, False
