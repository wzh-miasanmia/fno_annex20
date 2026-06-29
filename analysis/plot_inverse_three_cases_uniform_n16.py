import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import visualization.inverse_src_from_sparse_c_fixed_u_continuous as inv

WORKDIR = ROOT / "analysis" / "data_u0p2_dense_969"
CKPT = ROOT / "checkpoints" / "baseline_soft_rect" / "dense969_u0p2_w48_m16_d4_soft_rect_sk30_rawtarget.pt"
OUT = ROOT / "analysis" / "figures" / "inverse_case_study" / "inverse_three_cases_uniform_n16.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

CASES = [
    ("(a)", "900137"),
    ("(b)", "900533"),
    ("(c)", "900874"),
]

Lx, Ly = 9.0, 3.0
nx, ny = 384, 128
n_sensors = 16
sensor_strategy = "grid_centered"
sensor_seed = 42
margin_x = 2
margin_y = 2
inv_steps = 150
inv_lr = 0.05
inv_restarts = 4


def normalize_mask_kind(mask_kind: str) -> str:
    mk = str(mask_kind).lower().strip().replace("-", "_")
    if mk == "gaussian":
        mk = "gauss"
    if mk in ["softrect", "soft_rect(sigmoid)", "softrect(sigmoid)"]:
        mk = "soft_rect"
    if mk not in ["rect", "gauss", "soft_rect"]:
        mk = "rect"
    return mk

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# load model/checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT, map_location=device)
meta = ckpt.get("meta", {})
model = inv.FNO2D(
    in_ch=int(meta.get("in_ch", 3)),
    out_ch=int(meta.get("out_ch", 1)),
    width=int(meta.get("width", 48)),
    modes1=int(meta.get("modes1", 16)),
    modes2=int(meta.get("modes2", 16)),
    depth=int(meta.get("depth", 4)),
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
x_mean = ckpt["x_mean"].to(device)
x_std = ckpt["x_std"].to(device)
y_mean = ckpt["y_mean"].to(device)
y_std = ckpt["y_std"].to(device)
fixed_u = float(meta["fixed_inlet_u"])
mask_kind = normalize_mask_kind(meta.get("mask_kind", "rect"))
src_size = float(meta.get("src_size", 0.20))
sigma = float(meta.get("sigma", 0.12))
soft_k = float(meta.get("soft_k", 80.0))

cases_df = pd.read_csv(WORKDIR / "cases.csv")
cases_df["case_id"] = cases_df["case_id"].astype(str)
df_fixed = cases_df[np.abs(cases_df["inlet_u"].astype(float) - fixed_u) <= 1e-8].copy()
x_min = float(df_fixed["src_x"].min())
x_max = float(df_fixed["src_x"].max())
y_min = float(df_fixed["src_y"].min())
y_max = float(df_fixed["src_y"].max())

xs_phys = np.linspace(0.0, Lx, nx, dtype=np.float32)
ys_phys = np.linspace(0.0, Ly, ny, dtype=np.float32)

records = []
for tag, case_id in CASES:
    row = df_fixed[df_fixed["case_id"] == case_id].iloc[0]
    true_sx = float(row["src_x"])
    true_sy = float(row["src_y"])
    gt_file = inv.resolve_case_file(WORKDIR / "exports", case_id, float(row["inlet_u"]), true_sx, true_sy)
    df_gt = inv.load_case_csv(gt_file)
    _, _, C_gt = inv.df_to_grid(df_gt, Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    iy, ix = inv.make_sensor_indices(ny, nx, n_sensors, sensor_strategy, sensor_seed, margin_x, margin_y)
    obs = C_gt[iy, ix].astype(np.float32)
    best = inv._optimize_source_autodiff(
        model=model, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, device=device,
        iy=iy, ix=ix, obs=obs, ny=ny, nx=nx, Lx=Lx, Ly=Ly,
        mask_kind=mask_kind, src_size=src_size, sigma=sigma, soft_k=soft_k,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        steps=inv_steps, lr=inv_lr, n_restarts=inv_restarts, restart_seed=42,
    )
    est_sx = float(best["sx"])
    est_sy = float(best["sy"])
    x_in_est = inv.build_input_field(est_sx, est_sy, ny, nx, Lx, Ly, mask_kind, src_size, sigma, soft_k)
    C_est = inv.predict_c_field(model, (x_mean, x_std), (y_mean, y_std), device, x_in_est)
    err = C_est - C_gt
    dist = float(np.sqrt((est_sx - true_sx) ** 2 + (est_sy - true_sy) ** 2))
    records.append({
        "tag": tag, "case_id": case_id, "gt": C_gt, "pred": C_est, "err": err,
        "true_sx": true_sx, "true_sy": true_sy, "est_sx": est_sx, "est_sy": est_sy,
        "iy": iy, "ix": ix, "dist": dist,
    })

c_vmin = 0.0
c_vmax = max(max(float(np.max(r["gt"])) for r in records),
             max(float(np.max(r["pred"])) for r in records))
err_v = max(float(np.max(np.abs(r["err"]))) for r in records)

fig, axes = plt.subplots(3, len(records), figsize=(12.8, 7.2), constrained_layout=False)
for j, rec in enumerate(records):
    ax = axes[0, j]
    im_c = ax.imshow(rec["gt"], origin="lower", extent=(0, Lx, 0, Ly), cmap="viridis", vmin=c_vmin, vmax=c_vmax, aspect="auto")
    ax.scatter(xs_phys[rec["ix"]], ys_phys[rec["iy"]], s=14, c="white", edgecolors="black", linewidths=0.5, zorder=3)
    ax.scatter([rec["true_sx"]], [rec["true_sy"]], marker='x', s=120, c='red', linewidths=2.2, zorder=4)
    ax.set_title(rec["tag"], pad=8)

    ax = axes[1, j]
    ax.imshow(rec["pred"], origin="lower", extent=(0, Lx, 0, Ly), cmap="viridis", vmin=c_vmin, vmax=c_vmax, aspect="auto")
    ax.scatter(xs_phys[rec["ix"]], ys_phys[rec["iy"]], s=14, c="white", edgecolors="black", linewidths=0.5, zorder=3)
    ax.scatter([rec["est_sx"]], [rec["est_sy"]], marker='x', s=120, c='black', linewidths=2.2, zorder=4)

    ax = axes[2, j]
    im_e = ax.imshow(rec["err"], origin="lower", extent=(0, Lx, 0, Ly), cmap="coolwarm", vmin=-err_v, vmax=err_v, aspect="auto")
    ax.text(0.03, 0.90, f"err={rec['dist']:.3f} m", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85), fontsize=10)

for i in range(3):
    for j in range(len(records)):
        ax = axes[i, j]
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_box_aspect(Ly / Lx)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_yticks([0, 1, 2, 3])
        if i < 2:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("x (m)")
        if j == 0:
            ax.set_ylabel(["GT", "Prediction", "Error"][i])
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

cax1 = fig.add_axes([0.10, 0.93, 0.34, 0.032])
cb1 = fig.colorbar(im_c, cax=cax1, orientation='horizontal')
cb1.set_label('C')
cax2 = fig.add_axes([0.58, 0.93, 0.34, 0.032])
cb2 = fig.colorbar(im_e, cax=cax2, orientation='horizontal')
cb2.set_label('Error')

fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.88, wspace=0.10, hspace=0.03)
fig.savefig(OUT, bbox_inches='tight')
print(f'SAVED {OUT}')
