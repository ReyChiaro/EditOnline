import torch.nn as nn
from functools import reduce
from typing import Any


def get_nested_attr(obj, path: str) -> Any:
    return reduce(getattr, path.split("."), obj)


def set_nested_attr(obj, path: str, value):
    parts = path.split(".")
    parents = reduce(getattr, parts[:-1], obj) if len(parts) > 1 else obj
    setattr(parents, parts[-1], value)


def convert_integer(num: int) -> str:
    if num >= 1e9:
        return f"{num / 1e9:>6.2f} B"
    elif num >= 1e6:
        return f"{num / 1e6:>6.2f} M"
    else:
        return f"{num:,}"


def summarize_model(model: nn.Module) -> dict[str, int | float]:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    return {
        "Num Params": convert_integer(total_params),
        "Num Trainable Params": convert_integer(trainable_params),
        "Model Size (MB)": f"{size_all_mb:.4f}",
        "Model Size (GB)": f"{size_all_mb / 1024:.4f}",
        "Trainable (%)": f"{trainable_params/total_params*100:.2f}",
    }
