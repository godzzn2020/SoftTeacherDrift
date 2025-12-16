"""离线监督 Phase0 使用的 Tabular MLP 基线。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch import nn


@dataclass
class TabularMLPConfig:
    input_dim: int
    num_classes: int
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.2
    use_batchnorm: bool = True
    activation: str = "relu"


class TabularMLP(nn.Module):
    """多层全连接 MLP，支持 BatchNorm/Dropout。"""

    def __init__(self, cfg: TabularMLPConfig) -> None:
        super().__init__()
        if cfg.num_layers <= 0:
            raise ValueError("num_layers 必须 >= 1")
        layers = []
        in_dim = cfg.input_dim
        for _ in range(cfg.num_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            if cfg.use_batchnorm:
                layers.append(nn.BatchNorm1d(cfg.hidden_dim))
            layers.append(_get_activation(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, cfg.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_tabular_mlp(cfg: TabularMLPConfig) -> TabularMLP:
    """根据配置构建 TabularMLP。"""
    return TabularMLP(cfg)


def _get_activation(name: str) -> nn.Module:
    registry: Dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(f"不支持的 activation={name}，可选 {list(registry.keys())}")
    return registry[key]()
