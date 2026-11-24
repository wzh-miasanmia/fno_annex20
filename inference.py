"""Inference routines and trained architectures from the Annex 20 hybrid viewer."""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOM_THICKNESS = 1.0
INLET_HEIGHT = 0.168
INLET_AREA = INLET_HEIGHT * ROOM_THICKNESS



def build_source_mask_rect(Xg: np.ndarray, Yg: np.ndarray, src_x: float, src_y: float, src_size: float) -> np.ndarray:
    hx = 0.5 * src_size
    hy = 0.5 * src_size
    mask = (
        (Xg >= (src_x - hx)) & (Xg <= (src_x + hx)) &
        (Yg >= (src_y - hy)) & (Yg <= (src_y + hy))
    ).astype(np.float32)
    return mask


def build_source_mask_gaussian(Xg: np.ndarray, Yg: np.ndarray, src_x: float, src_y: float, sigma: float) -> np.ndarray:
    r2 = (Xg - src_x) ** 2 + (Yg - src_y) ** 2
    mask = np.exp(-0.5 * r2 / (sigma ** 2)).astype(np.float32)
    return mask


def build_source_mask_soft_rect_sigmoid(
    Xg: np.ndarray,
    Yg: np.ndarray,
    src_x: float,
    src_y: float,
    src_size: float,
    soft_k: float = 80.0,
) -> np.ndarray:
    """
    Smooth rectangular mask using a sigmoid on the rectangle signed distance.
    Larger soft_k -> sharper boundary.
    """
    hx = 0.5 * src_size
    hy = 0.5 * src_size

    dx = np.abs(Xg - src_x) - hx
    dy = np.abs(Yg - src_y) - hy
    sdf = np.maximum(dx, dy)  # <=0 inside rectangle

    z = np.clip(soft_k * sdf, -60.0, 60.0)
    mask = (1.0 / (1.0 + np.exp(z))).astype(np.float32)
    return mask


@dataclass
class Normalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-12)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + 1e-12) + self.mean


@dataclass
class TargetTransform:
    kind: str = "raw"
    eps: float = 1e-8

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "raw":
            return y
        if self.kind == "log10":
            return torch.log10(torch.clamp(y, min=0.0) + self.eps)
        raise ValueError(f"Unsupported target transform: {self.kind}")

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "raw":
            return y
        if self.kind == "log10":
            return torch.clamp(torch.pow(10.0, y) - self.eps, min=0.0)
        raise ValueError(f"Unsupported target transform: {self.kind}")


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / (self.in_channels * self.out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, ny, nx = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        m1 = min(self.modes1, ny)
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft = torch.zeros(bsz, self.out_channels, ny, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        weight = torch.view_as_complex(self.weight)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], weight[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(ny, nx), norm="ortho")


class FNOBackbone2D(nn.Module):
    def __init__(self, in_ch: int, width: int = 64, modes1: int = 24, modes2: int = 16, depth: int = 4):
        super().__init__()
        self.fc0 = nn.Conv2d(in_ch, width, kernel_size=1)
        self.spec_layers = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)])
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])
        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.fc0(x)
        for spec, w in zip(self.spec_layers, self.w_layers):
            feat = self.act(spec(feat) + w(feat))
        hidden = self.act(self.fc1(feat))
        coarse = self.fc2(hidden)
        return coarse, hidden


class CNNGradientRefiner(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 64, layers: int = 4, residual_scale: float = 0.25):
        super().__init__()
        blocks: List[nn.Module] = []
        ch = in_ch
        for _ in range(max(1, int(layers))):
            blocks.extend([nn.Conv2d(ch, hidden_ch, kernel_size=3, padding=1), nn.GELU()])
            ch = hidden_ch
        self.trunk = nn.Sequential(*blocks)
        self.residual = nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1)
        self.grad_head = nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(x)
        return self.residual_scale * self.residual(feat), self.grad_head(feat)


class FNOCNN2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        width: int = 64,
        modes1: int = 24,
        modes2: int = 16,
        depth: int = 4,
        cnn_width: int = 64,
        cnn_layers: int = 4,
        residual_scale: float = 0.25,
    ):
        super().__init__()
        self.fno = FNOBackbone2D(in_ch=in_ch, width=width, modes1=modes1, modes2=modes2, depth=depth)
        self.cnn = CNNGradientRefiner(
            in_ch=in_ch + 1,
            hidden_ch=cnn_width,
            layers=cnn_layers,
            residual_scale=residual_scale,
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        coarse, _ = self.fno(x)
        residual, grad = self.cnn(torch.cat([x, coarse], dim=1))
        pred = coarse + residual
        if return_aux:
            return pred, {"coarse": coarse, "residual": residual, "grad": grad}
        return pred


class VelocityFNO2D(nn.Module):
    def __init__(self, in_ch: int = 4, out_ch: int = 2, width: int = 64, modes1: int = 24, modes2: int = 16, depth: int = 4):
        super().__init__()
        self.fc0 = nn.Conv2d(in_ch, width, kernel_size=1)
        self.spec_layers = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)])
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])
        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.fc0(x)
        for spec, w in zip(self.spec_layers, self.w_layers):
            x = self.act(spec(x) + w(x))
        x = self.act(self.fc1(x))
        return self.fc2(x)


def make_xy(lx: float, ly: float, ny: int, nx: int):
    return np.meshgrid(
        np.linspace(0.0, lx, nx, dtype=np.float32),
        np.linspace(0.0, ly, ny, dtype=np.float32),
    )


def compute_ach(inlet_u: float, lx: float, ly: float) -> float:
    return float(inlet_u) * INLET_AREA * 3600.0 / (float(lx) * float(ly) * ROOM_THICKNESS)


def rect_mask(xg, yg, x0, x1, y0, y1):
    xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
    ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
    return ((xg >= xa) & (xg <= xb) & (yg >= ya) & (yg <= yb)).astype(np.float32)


def masked_mean(field, mask):
    denom = float(mask.sum())
    if denom <= 0:
        return float("nan")
    return float((field * mask).sum() / denom)


def make_source_mask(meta: dict, sx: float, sy: float) -> np.ndarray:
    nx = int(meta.get("nx", 384))
    ny = int(meta.get("ny", 128))
    lx = float(meta.get("Lx", 9.0))
    ly = float(meta.get("Ly", 3.0))
    xg, yg = make_xy(lx, ly, ny, nx)
    mask_kind = str(meta.get("mask_kind", "soft_rect")).lower().strip().replace("-", "_")
    src_size = float(meta.get("src_size", 0.20))
    sigma = float(meta.get("sigma", 0.12))
    soft_k = float(meta.get("soft_k", 80.0))
    if mask_kind in {"softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"}:
        mask_kind = "soft_rect"
    if mask_kind == "gaussian":
        mask_kind = "gauss"
    if mask_kind == "rect":
        return build_source_mask_rect(xg, yg, sx, sy, src_size)
    if mask_kind == "gauss":
        return build_source_mask_gaussian(xg, yg, sx, sy, sigma)
    return build_source_mask_soft_rect_sigmoid(xg, yg, sx, sy, src_size, soft_k)


def load_velocity(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    meta = ckpt.get("meta", {})
    model = VelocityFNO2D(
        in_ch=int(meta.get("in_ch", 3)),
        out_ch=int(meta.get("out_ch", 2)),
        width=int(meta.get("width", 64)),
        modes1=int(meta.get("modes1", 24)),
        modes2=int(meta.get("modes2", 16)),
        depth=int(meta.get("depth", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x_norm = Normalizer(ckpt["x_mean"].to(device), ckpt["x_std"].to(device))
    y_norm = Normalizer(ckpt["y_mean"].to(device), ckpt["y_std"].to(device))
    return model, x_norm, y_norm, meta, device


def load_concentration(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    meta = ckpt.get("meta", {})
    model = FNOCNN2D(
        in_ch=int(meta.get("in_ch", ckpt["x_mean"].shape[1])),
        width=int(meta.get("width", 64)),
        modes1=int(meta.get("modes1", 24)),
        modes2=int(meta.get("modes2", 16)),
        depth=int(meta.get("depth", 4)),
        cnn_width=int(meta.get("cnn_width", 64)),
        cnn_layers=int(meta.get("cnn_layers", 4)),
        residual_scale=float(meta.get("residual_scale", 0.25)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x_norm = Normalizer(ckpt["x_mean"].to(device), ckpt["x_std"].to(device))
    y_norm = Normalizer(ckpt["y_mean"].to(device), ckpt["y_std"].to(device))
    target_transform = TargetTransform(str(meta.get("target_transform", "raw")), float(meta.get("target_transform_eps", 1e-8)))
    return model, x_norm, y_norm, target_transform, meta, device


def build_velocity_input(meta: dict, inlet_u: float, sx: float, sy: float, ny: int, nx: int):
    xx = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    yy = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    xc, yc = np.meshgrid(xx, yy)
    available = {
        "inlet_u": np.full((ny, nx), inlet_u, dtype=np.float32),
        "X": xc.astype(np.float32),
        "Y": yc.astype(np.float32),
    }
    channel_names = list(meta.get("input_channels", ["inlet_u", "X", "Y"]))
    if "S_geometry_mask" in channel_names:
        available["S_geometry_mask"] = make_source_mask(meta, sx, sy)
    return torch.from_numpy(
        np.stack([available[name] for name in channel_names], axis=0)[None, ...]
    )


def build_concentration_input(meta: dict, inlet_u: float, sx: float, sy: float, ux: np.ndarray, uy: np.ndarray):
    nx = int(meta.get("nx", ux.shape[1]))
    ny = int(meta.get("ny", ux.shape[0]))
    xx = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    yy = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    xc, yc = np.meshgrid(xx, yy)
    mask = make_source_mask(meta, sx, sy)
    channels = [mask, xc, yc]
    if bool(meta.get("use_velocity_input", True)):
        channels.extend([ux, uy])
    if bool(meta.get("use_inlet_scalar", True)):
        channels.append(np.full((ny, nx), inlet_u, dtype=np.float32))
    return torch.from_numpy(np.stack(channels, axis=0)[None, ...].astype(np.float32))


@torch.no_grad()
def predict_hybrid(v_assets, c_assets, inlet_u: float, sx: float, sy: float):
    v_model, vx_norm, vy_norm, v_meta, v_device = v_assets
    c_model, cx_norm, cy_norm, target_transform, c_meta, c_device = c_assets
    nx = int(v_meta.get("nx", c_meta.get("nx", 384)))
    ny = int(v_meta.get("ny", c_meta.get("ny", 128)))
    xv = build_velocity_input(v_meta, inlet_u, sx, sy, ny, nx).to(v_device)
    uv_n = v_model(vx_norm.encode(xv))
    uv = vy_norm.decode(uv_n).detach().cpu().numpy()[0]
    if str(v_meta.get("target_mode", "raw")) == "inlet_normalized":
        uv = uv * float(inlet_u)
    ux, uy = uv[0].astype(np.float32), uv[1].astype(np.float32)

    xc = build_concentration_input(c_meta, inlet_u, sx, sy, ux, uy).to(c_device)
    c_pred_n = c_model(cx_norm.encode(xc))
    c_pred = target_transform.inverse(cy_norm.decode(c_pred_n)).detach().cpu().numpy()[0, 0]
    return ux, uy, np.sqrt(ux**2 + uy**2), c_pred
