# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Type

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mlp_dim: int,
        out_dim: int | None = None,
        n_hidden_layers: int = 1,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        if n_hidden_layers == 0:
            self.layers = nn.Linear(in_dim, out_dim)
        else:
            layers = nn.ModuleList()
            layers.append(nn.Linear(in_dim, mlp_dim))
            layers.append(act())
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(mlp_dim, mlp_dim))
                layers.append(act())
            # Last layer to project back to in_dim
            layers.append(nn.Linear(mlp_dim, out_dim))
            self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
