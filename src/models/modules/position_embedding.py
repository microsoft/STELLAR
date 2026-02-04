# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Tuple, Type

import numpy as np
import torch
from torch import nn


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor, n_pos=None) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        # Cast coords to the same dtype as positional_encoding_gaussian_matrix
        coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
        coords = coords @ self.positional_encoding_gaussian_matrix
        if n_pos is not None:
            coords = coords[..., :n_pos]
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        #return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        # put pair of sin, cos alternatively
        return torch.stack([torch.sin(coords), torch.cos(coords)], dim=-1).view(coords.size(0), coords.size(1), -1)



    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size (H x W x C)."""
        h, w, c = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1), n_pos=c//2)
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)  # B x N x C

