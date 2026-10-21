"""Basic 2-D GINO for native CFD point clouds.

The model uses learned local integral transforms between the native points and
a regular latent grid, applies standard FNO layers on that grid, and queries
the result back at every native point.  No preprocessing interpolation or
learned coordinate deformation is used.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.geofno import SpectralConv2d


class LocalStencil2D:
    """Four-cell bipartite neighbourhood between points and a regular grid."""

    def __init__(self, latent_size: Tuple[int, int]):
        self.ny, self.nx = int(latent_size[0]), int(latent_size[1])
        if self.ny < 2 or self.nx < 2:
            raise ValueError("latent_size must be at least 2x2.")

    def neighbours(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return grid indices and (grid - point) offsets for four neighbours."""
        x = coords[..., 0].clamp(0.0, 1.0)
        y = coords[..., 1].clamp(0.0, 1.0)
        gx = x * (self.nx - 1)
        gy = y * (self.ny - 1)
        x0 = torch.floor(gx).long().clamp(0, self.nx - 2)
        y0 = torch.floor(gy).long().clamp(0, self.ny - 2)
        x1, y1 = x0 + 1, y0 + 1

        ix = torch.stack((x0, x1, x0, x1), dim=-1)
        iy = torch.stack((y0, y0, y1, y1), dim=-1)
        indices = iy * self.nx + ix
        grid_x = ix.to(coords.dtype) / float(self.nx - 1)
        grid_y = iy.to(coords.dtype) / float(self.ny - 1)
        relative = torch.stack((grid_x - x.unsqueeze(-1), grid_y - y.unsqueeze(-1)), dim=-1)
        return indices, relative


class LocalKernel(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), int(channels)),
        )

    def forward(self, relative: torch.Tensor) -> torch.Tensor:
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        return self.network(torch.cat((relative, distance), dim=-1))


class BasicGINO2D(nn.Module):
    """Local GNO encoder -> latent-grid FNO -> local GNO decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 48,
        modes1: int = 12,
        modes2: int = 12,
        depth: int = 3,
        latent_size: Tuple[int, int] = (48, 144),
        kernel_width: int = 64,
    ):
        super().__init__()
        self.width = int(width)
        self.latent_size = (int(latent_size[0]), int(latent_size[1]))
        self.stencil = LocalStencil2D(self.latent_size)
        self.lift = nn.Linear(int(in_channels), self.width)
        self.encoder_kernel = LocalKernel(self.width, kernel_width)
        self.decoder_kernel = LocalKernel(self.width, kernel_width)
        self.input_coord_bias = nn.Conv2d(2, self.width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(self.width, modes1, modes2) for _ in range(max(1, int(depth)))]
        )
        self.local_layers = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, kernel_size=1) for _ in range(max(1, int(depth)))]
        )
        self.coord_biases = nn.ModuleList(
            [nn.Conv2d(2, self.width, kernel_size=1) for _ in range(max(1, int(depth)))]
        )
        self.query_lift = nn.Linear(int(in_channels), self.width)
        self.projection = nn.Sequential(
            nn.Linear(self.width, 128), nn.GELU(), nn.Linear(128, int(out_channels))
        )

    def _grid(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ny, nx = self.latent_size
        y = torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def _encode(
        self,
        values: torch.Tensor,
        coords: torch.Tensor,
        valid: torch.Tensor,
        quadrature: torch.Tensor,
    ) -> torch.Tensor:
        batch, points, channels = values.shape
        ny, nx = self.latent_size
        grid_points = ny * nx
        indices, relative = self.stencil.neighbours(coords)
        gates = torch.sigmoid(self.encoder_kernel(relative))
        base_weight = quadrature.clamp_min(0.0) * valid.to(quadrature.dtype)
        weights = gates * base_weight[..., None, None]
        messages = values.unsqueeze(2) * weights

        offsets = torch.arange(batch, device=coords.device)[:, None, None] * grid_points
        flat_indices = (indices + offsets).reshape(-1)
        numerator = values.new_zeros(batch * grid_points, channels)
        denominator = values.new_zeros(batch * grid_points, channels)
        numerator.index_add_(0, flat_indices, messages.reshape(-1, channels))
        denominator.index_add_(0, flat_indices, weights.reshape(-1, channels))
        latent = numerator / denominator.clamp_min(1e-8)
        return latent.reshape(batch, grid_points, channels).transpose(1, 2).reshape(
            batch, channels, ny, nx
        )

    def _decode(
        self,
        latent: torch.Tensor,
        features: torch.Tensor,
        coords: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, ny, nx = latent.shape
        indices, relative = self.stencil.neighbours(coords)
        flat = latent.flatten(2).transpose(1, 2)
        batch_index = torch.arange(batch, device=coords.device)[:, None, None]
        neighbours = flat[batch_index, indices]
        weights = torch.softmax(self.decoder_kernel(relative), dim=2)
        hidden = (neighbours * weights).sum(dim=2) + self.query_lift(features)
        hidden = F.gelu(hidden)
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)
        return hidden

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        valid: torch.Tensor,
        quadrature: torch.Tensor,
        return_aux: bool = False,
    ):
        if quadrature is None:
            raise ValueError("BasicGINO2D requires point quadrature weights.")
        lifted = self.lift(features)
        latent = self._encode(lifted, coords, valid, quadrature)
        grid = self._grid(latent.shape[0], latent.device, latent.dtype)
        latent = F.gelu(latent + self.input_coord_bias(grid))
        for spectral, local, coord_bias in zip(
            self.spectral_layers, self.local_layers, self.coord_biases
        ):
            latent = F.gelu(spectral(latent) + local(latent) + coord_bias(grid))
        hidden = self._decode(latent, features, coords, valid)
        prediction = self.projection(hidden)
        prediction = prediction * valid.unsqueeze(-1).to(prediction.dtype)
        if return_aux:
            return prediction, {"latent": latent}
        return prediction
