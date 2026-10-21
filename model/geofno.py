"""Pure Geo-FNO on raw point clouds.

The module operates directly on unstructured CFD nodes.  It intentionally does
not rasterize, interpolate, or call ``grid_sample``.  The first spectral layer
uses a non-uniform discrete Fourier transform to lift node features to a latent
regular grid.  Standard FFT layers act in that latent space, and a geometric
inverse transform evaluates the result back at the original nodes.

Tensor conventions
------------------
``features``: [batch, points, input_channels]
``coords``:   [batch, points, 2], normalized to [0, 1]^2
``valid``:    [batch, points] boolean mask for padded batches
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _complex_weight(in_channels: int, out_channels: int, modes1: int, modes2: int) -> nn.Parameter:
    scale = 1.0 / max(1, in_channels * out_channels)
    real_imag = scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
    return nn.Parameter(torch.view_as_complex(real_imag))


class LearnedDeformation2D(nn.Module):
    """Learn IPHI = phi^-1: physical coordinates -> computational coordinates.

    ``code`` describes case-dependent mesh refinement.  For the Annex 20
    dataset this is the normalized source position, because STAR-CCM+ refines
    the mesh around that position.  The residual envelope fixes the rectangular
    boundary and avoids the zero-gradient hard clamp used by the first draft.
    """

    def __init__(
        self,
        code_dim: int = 2,
        width: int = 32,
        fourier_bands: int = 6,
        max_displacement: float = 0.15,
        architecture_version: str = "paper_weighted_v2",
    ):
        super().__init__()
        self.code_dim = int(code_dim)
        self.max_displacement = float(max_displacement)
        self.architecture_version = str(architecture_version)
        frequencies = 2.0 ** torch.arange(int(fourier_bands), dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        if self.architecture_version == "legacy_unweighted":
            encoded_dim = 2 + 4 * int(fourier_bands) + self.code_dim
        else:
            # Match the useful IPHI features from the reference implementation:
            # Cartesian position plus polar angle/radius and Fourier features.
            encoded_dim = 4 + 8 * int(fourier_bands) + self.code_dim
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, coords: torch.Tensor, code: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.architecture_version == "legacy_unweighted":
            base = coords
        else:
            centered = coords - 0.5
            angle = torch.atan2(centered[..., 1], centered[..., 0]) / math.pi
            radius = torch.linalg.vector_norm(centered, dim=-1) * math.sqrt(2.0)
            base = torch.cat([coords, angle.unsqueeze(-1), radius.unsqueeze(-1)], dim=-1)
        angles = 2.0 * math.pi * base.unsqueeze(-1) * self.frequencies
        encoded = [base, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)]
        if self.code_dim:
            if code is None:
                raise ValueError("A geometry code is required when code_dim > 0.")
            if code.shape[-1] != self.code_dim:
                raise ValueError(f"Expected code_dim={self.code_dim}, got {code.shape[-1]}.")
            encoded.append(code[:, None, :].expand(-1, coords.shape[1], -1))
        displacement = self.max_displacement * torch.tanh(self.net(torch.cat(encoded, dim=-1)))
        if self.architecture_version == "legacy_unweighted":
            return torch.clamp(coords + displacement, min=0.0, max=1.0)
        boundary_envelope = 4.0 * coords * (1.0 - coords)
        return coords + boundary_envelope * displacement

    def map_with_jacobian(
        self,
        coords: torch.Tensor,
        code: Optional[torch.Tensor] = None,
        epsilon: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return IPHI coordinates and det(D_x IPHI) via differentiable differences.

        Central differences avoid constructing an input-coordinate autograd graph
        and its expensive mixed second derivatives.  One-sided differences are
        used at the normalized-domain boundary.
        """
        eps = float(epsilon)
        if not 0.0 < eps < 0.5:
            raise ValueError("Jacobian finite-difference epsilon must be in (0, 0.5).")
        mapped = self(coords, code)
        derivatives = []
        for axis in range(2):
            plus = coords.clone()
            minus = coords.clone()
            plus[..., axis] = torch.clamp(coords[..., axis] + eps, max=1.0)
            minus[..., axis] = torch.clamp(coords[..., axis] - eps, min=0.0)
            span = torch.clamp(plus[..., axis] - minus[..., axis], min=eps)
            derivative = (self(plus, code) - self(minus, code)) / span.unsqueeze(-1)
            derivatives.append(derivative)
        d_dx, d_dy = derivatives
        determinant = d_dx[..., 0] * d_dy[..., 1] - d_dx[..., 1] * d_dy[..., 0]
        return mapped, determinant


class NonuniformSpectralConv2d(nn.Module):
    """Spectral map between unstructured nodes and a uniform latent grid.

    The transform follows equations (12)-(13) of the Geo-FNO paper.  Callers
    supply the full m(x)=|det(D IPHI)|/rho_a(x) quadrature weight.  Point chunks
    bound the temporary complex basis tensor for large STAR-CCM+ exports.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        grid_size: Tuple[int, int],
        point_chunk_size: int = 2048,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.ny, self.nx = (int(grid_size[0]), int(grid_size[1]))
        self.point_chunk_size = int(point_chunk_size)
        if self.modes1 > self.ny // 2:
            raise ValueError("modes1 must be <= latent_ny // 2.")
        if self.modes2 > self.nx // 2 + 1:
            raise ValueError("modes2 must be <= latent_nx // 2 + 1.")
        self.weight_positive = _complex_weight(in_channels, out_channels, modes1, modes2)
        self.weight_negative = _complex_weight(in_channels, out_channels, modes1, modes2)

    def _wave_numbers(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        ky = torch.cat(
            [
                torch.arange(self.modes1, device=device),
                torch.arange(-self.modes1, 0, device=device),
            ]
        ).to(dtype=dtype)
        kx = torch.arange(self.modes2, device=device, dtype=dtype)
        return ky, kx

    def _apply_weights(self, coefficients: torch.Tensor) -> torch.Tensor:
        positive = torch.einsum(
            "bimn,iomn->bomn", coefficients[:, :, : self.modes1], self.weight_positive
        )
        negative = torch.einsum(
            "bimn,iomn->bomn", coefficients[:, :, self.modes1 :], self.weight_negative
        )
        return torch.cat([positive, negative], dim=2)

    def points_to_grid(
        self,
        values: torch.Tensor,
        computational_coords: torch.Tensor,
        valid: torch.Tensor,
        quadrature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map [B,C,N] node values to [B,C,H,W] latent-grid values."""
        batch, _, n_points = values.shape
        ky, kx = self._wave_numbers(values.device, torch.float32)
        coefficients = torch.zeros(
            batch,
            self.in_channels,
            2 * self.modes1,
            self.modes2,
            dtype=torch.complex64,
            device=values.device,
        )
        valid_f = valid.to(dtype=torch.float32)
        if quadrature is None:
            raise ValueError(
                "Geo-FNO points_to_grid requires sampling-density/Jacobian quadrature weights."
            )
        weights = quadrature.to(dtype=torch.float32) * valid_f
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("Quadrature weights must be finite and non-negative.")
        weight_sum = torch.clamp(weights.sum(dim=1), min=1e-12)

        values32 = values.to(dtype=torch.float32)
        coords32 = computational_coords.to(dtype=torch.float32)
        for start in range(0, n_points, self.point_chunk_size):
            stop = min(start + self.point_chunk_size, n_points)
            coord = coords32[:, start:stop]
            phase = (
                coord[..., 1, None, None] * ky[None, None, :, None]
                + coord[..., 0, None, None] * kx[None, None, None, :]
            )
            basis = torch.exp(torch.complex(torch.zeros_like(phase), -2.0 * math.pi * phase))
            weighted_values = values32[:, :, start:stop] * weights[:, None, start:stop]
            coefficients += torch.einsum("bcn,bnmk->bcmk", weighted_values.to(torch.complex64), basis)

        # This scaling matches norm='ortho' FFT when nodes coincide with the
        # latent grid, while remaining a quadrature average on irregular nodes.
        coefficients *= math.sqrt(self.ny * self.nx) / weight_sum[:, None, None, None]
        transformed = self._apply_weights(coefficients)
        spectrum = torch.zeros(
            batch,
            self.out_channels,
            self.ny,
            self.nx // 2 + 1,
            dtype=torch.complex64,
            device=values.device,
        )
        spectrum[:, :, : self.modes1, : self.modes2] = transformed[:, :, : self.modes1]
        spectrum[:, :, -self.modes1 :, : self.modes2] = transformed[:, :, self.modes1 :]
        return torch.fft.irfft2(spectrum, s=(self.ny, self.nx), norm="ortho")

    def grid_to_points(
        self,
        grid_values: torch.Tensor,
        computational_coords: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Map [B,C,H,W] latent-grid values to [B,C,N] node values."""
        spectrum = torch.fft.rfft2(grid_values.to(dtype=torch.float32), norm="ortho")
        selected = torch.cat(
            [
                spectrum[:, :, : self.modes1, : self.modes2],
                spectrum[:, :, -self.modes1 :, : self.modes2],
            ],
            dim=2,
        )
        transformed = self._apply_weights(selected)
        ky, kx = self._wave_numbers(grid_values.device, torch.float32)
        batch, n_points = computational_coords.shape[:2]
        output = torch.zeros(
            batch, self.out_channels, n_points, dtype=torch.float32, device=grid_values.device
        )
        coords32 = computational_coords.to(dtype=torch.float32)
        norm = math.sqrt(self.ny * self.nx)
        for start in range(0, n_points, self.point_chunk_size):
            stop = min(start + self.point_chunk_size, n_points)
            coord = coords32[:, start:stop]
            phase = (
                coord[..., 1, None, None] * ky[None, None, :, None]
                + coord[..., 0, None, None] * kx[None, None, None, :]
            )
            basis = torch.exp(torch.complex(torch.zeros_like(phase), 2.0 * math.pi * phase))
            # rFFT stores kx >= 0.  Nonzero positive-kx modes have an implicit
            # conjugate partner and therefore contribute twice to the real sum.
            zero = torch.einsum("bcm,bnm->bcn", transformed[..., 0], basis[..., 0]).real
            if self.modes2 > 1:
                positive = 2.0 * torch.einsum(
                    "bcmk,bnmk->bcn", transformed[..., 1:], basis[..., 1:]
                ).real
                zero = zero + positive
            output[:, :, start:stop] = zero / norm
        return output * valid[:, None, :].to(dtype=output.dtype)


class SpectralConv2d(nn.Module):
    def __init__(self, channels: int, modes1: int, modes2: int):
        super().__init__()
        self.channels = int(channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.weight_positive = _complex_weight(channels, channels, modes1, modes2)
        self.weight_negative = _complex_weight(channels, channels, modes1, modes2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, ny, nx = x.shape
        spectrum = torch.fft.rfft2(x.to(dtype=torch.float32), norm="ortho")
        out = torch.zeros(
            batch, self.channels, ny, nx // 2 + 1, dtype=torch.complex64, device=x.device
        )
        m1 = min(self.modes1, ny // 2)
        m2 = min(self.modes2, nx // 2 + 1)
        out[:, :, :m1, :m2] = torch.einsum(
            "bimn,iomn->bomn",
            spectrum[:, :, :m1, :m2],
            self.weight_positive[:, :, :m1, :m2],
        )
        out[:, :, -m1:, :m2] = torch.einsum(
            "bimn,iomn->bomn",
            spectrum[:, :, -m1:, :m2],
            self.weight_negative[:, :, :m1, :m2],
        )
        return torch.fft.irfft2(out, s=(ny, nx), norm="ortho")


class GeoFNOBackbone2D(nn.Module):
    """Geo-FNO global branch with point-cloud input and output."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        width: int = 48,
        modes1: int = 12,
        modes2: int = 12,
        depth: int = 3,
        latent_size: Tuple[int, int] = (48, 144),
        code_dim: int = 2,
        deformation_width: int = 32,
        max_displacement: float = 0.15,
        point_chunk_size: int = 2048,
        architecture_version: str = "paper_weighted_v2",
    ):
        super().__init__()
        self.width = int(width)
        self.latent_size = (int(latent_size[0]), int(latent_size[1]))
        self.architecture_version = str(architecture_version)
        self.deformation = LearnedDeformation2D(
            code_dim=code_dim,
            width=deformation_width,
            max_displacement=max_displacement,
            architecture_version=self.architecture_version,
        )
        self.lift = nn.Linear(in_channels, width)
        self.input_geo = NonuniformSpectralConv2d(
            width, width, modes1, modes2, self.latent_size, point_chunk_size
        )
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, modes1, modes2) for _ in range(max(1, int(depth)))]
        )
        self.local_layers = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(max(1, int(depth)))]
        )
        if self.architecture_version == "legacy_unweighted":
            self.coord_bias = nn.Conv2d(2, width, kernel_size=1)
        else:
            self.input_coord_bias = nn.Conv2d(2, width, kernel_size=1)
            self.layer_coord_biases = nn.ModuleList(
                [nn.Conv2d(2, width, kernel_size=1) for _ in range(max(1, int(depth)))]
            )
        self.output_geo = NonuniformSpectralConv2d(
            width, width, modes1, modes2, self.latent_size, point_chunk_size
        )
        self.point_bias = nn.Conv1d(2, width, kernel_size=1)
        projection_width = width if self.architecture_version == "legacy_unweighted" else 128
        self.coarse_head = nn.Sequential(
            nn.Linear(width, projection_width), nn.GELU(), nn.Linear(projection_width, int(out_channels))
        )

    def _latent_grid(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ny, nx = self.latent_size
        y = torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        valid: torch.Tensor,
        code: Optional[torch.Tensor] = None,
        quadrature: Optional[torch.Tensor] = None,
        jacobian_epsilon: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.architecture_version == "legacy_unweighted":
            computational_coords = self.deformation(coords, code=code)
            legacy_quadrature = valid.to(dtype=features.dtype) if quadrature is None else quadrature
            lifted = self.lift(features).transpose(1, 2)
            latent = self.input_geo.points_to_grid(
                lifted, computational_coords, valid, legacy_quadrature
            )
            latent = latent + self.coord_bias(
                self._latent_grid(latent.shape[0], latent.device, latent.dtype)
            )
            for spectral, local in zip(self.spectral_layers, self.local_layers):
                latent = F.gelu(spectral(latent) + local(latent))
            point_hidden = self.output_geo.grid_to_points(latent, computational_coords, valid)
            point_hidden = point_hidden + self.point_bias(coords.transpose(1, 2))
            point_hidden = F.gelu(point_hidden.transpose(1, 2))
            coarse = self.coarse_head(point_hidden)
            coarse = coarse * valid.unsqueeze(-1).to(dtype=coarse.dtype)
            return coarse, point_hidden, computational_coords, torch.ones_like(valid, dtype=coarse.dtype)

        computational_coords, jacobian_det = self.deformation.map_with_jacobian(
            coords, code=code, epsilon=jacobian_epsilon
        )
        if quadrature is None:
            raise ValueError("Sampling-density quadrature is required for paper-faithful Geo-FNO.")
        full_quadrature = quadrature * jacobian_det.abs()
        lifted = self.lift(features).transpose(1, 2)
        latent = self.input_geo.points_to_grid(lifted, computational_coords, valid, full_quadrature)
        grid = self._latent_grid(latent.shape[0], latent.device, latent.dtype)
        latent = F.gelu(latent + self.input_coord_bias(grid))
        for spectral, local, coord_bias in zip(
            self.spectral_layers, self.local_layers, self.layer_coord_biases
        ):
            latent = F.gelu(spectral(latent) + local(latent) + coord_bias(grid))
        point_hidden = self.output_geo.grid_to_points(latent, computational_coords, valid)
        point_hidden = point_hidden + self.point_bias(coords.transpose(1, 2))
        point_hidden = F.gelu(point_hidden.transpose(1, 2))
        coarse = self.coarse_head(point_hidden)
        coarse = coarse * valid.unsqueeze(-1).to(dtype=coarse.dtype)
        return coarse, point_hidden, computational_coords, jacobian_det


class GeoFNO2D(nn.Module):
    """Pure Geo-FNO baseline with no CNN or point-neighbour branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        width: int = 48,
        modes1: int = 12,
        modes2: int = 12,
        depth: int = 3,
        latent_size: Tuple[int, int] = (48, 144),
        code_dim: int = 2,
        deformation_width: int = 32,
        max_displacement: float = 0.15,
        point_chunk_size: int = 2048,
        architecture_version: str = "paper_weighted_v2",
    ):
        super().__init__()
        self.geofno = GeoFNOBackbone2D(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            depth=depth,
            latent_size=latent_size,
            code_dim=code_dim,
            deformation_width=deformation_width,
            max_displacement=max_displacement,
            point_chunk_size=point_chunk_size,
            architecture_version=architecture_version,
        )

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        code: Optional[torch.Tensor] = None,
        quadrature: Optional[torch.Tensor] = None,
        jacobian_epsilon: float = 1e-3,
        return_aux: bool = False,
    ):
        if valid is None:
            valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        prediction, hidden, computational_coords, jacobian_det = self.geofno(
            features,
            coords,
            valid,
            code=code,
            quadrature=quadrature,
            jacobian_epsilon=jacobian_epsilon,
        )
        if return_aux:
            return prediction, {
                "hidden": hidden,
                "computational_coords": computational_coords,
                "jacobian_det": jacobian_det,
            }
        return prediction
