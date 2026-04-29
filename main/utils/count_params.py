from __future__ import annotations

from typing import Dict

import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total, trainable, and non-trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def count_parameters_per_layer(model: nn.Module) -> Dict[str, int]:
    """Return parameter count for each named module that owns parameters directly."""
    result: Dict[str, int] = {}
    for name, module in model.named_modules():
        own_params = sum(p.numel() for p in module.parameters(recurse=False))
        if own_params > 0:
            result[name] = own_params
    return result


def print_parameter_summary(model: nn.Module) -> None:
    """Print a human-readable parameter summary to stdout."""
    counts = count_parameters(model)
    print(f"{'Total params':<30}: {counts['total']:>15,}")
    print(f"{'Trainable params':<30}: {counts['trainable']:>15,}")
    print(f"{'Non-trainable params':<30}: {counts['non_trainable']:>15,}")
