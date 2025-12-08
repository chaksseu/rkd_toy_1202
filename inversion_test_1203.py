#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, Tuple, Union

import hashlib
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDIMInverseScheduler
import matplotlib.pyplot as plt


import colorsys
import numpy as np


# ---- 32 distinct colors (high-contrast) ----
import colorsys

def _distinct_hex_colors(n: int, s: float = 0.78, v: float = 0.95):
    """
    황금비 간격으로 hue를 배치해 n개의 서로 다른 색을 생성.
    s(채도), v(명도)를 높게 유지해 대비를 확보.
    항상 같은 순서로 재현 가능.
    return: ['#rrggbb', ...] 길이 n
    """
    phi = 0.6180339887498949
    hues = [(i * phi) % 1.0 for i in range(n)]
    rgb = [colorsys.hsv_to_rgb(h, s, v) for h in hues]
    to_hex = lambda r,g,b: f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    return [to_hex(*c) for c in rgb]

def _hex_to_rgba_array(hex_list, alpha: float = 0.85) -> np.ndarray:
    arr = []
    for h in hex_list:
        h = h.lstrip('#')
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        arr.append([r, g, b, alpha])
    return np.array(arr, dtype=float)

# 전역 32색 팔레트 (HEX→RGBA)
_COLORS32_HEX = _distinct_hex_colors(32)             # 재현 가능 고대비 32색
_COLORS32_RGBA_BASE = _hex_to_rgba_array(_COLORS32_HEX, alpha=0.85)

def colors_by_id32(ids: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """
    동일 id -> 항상 동일한 32색 중 하나를 할당.
    32색 팔레트를 순환 사용 (ids % 32).
    return: (N,4) RGBA in [0,1]
    """
    ids = np.asarray(ids, dtype=np.int64)
    cols = _COLORS32_RGBA_BASE[ids % 32].copy()
    cols[:, 3] = alpha
    return cols



# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 1234):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Union[str, Path]):
    Path(p).mkdir(parents=True, exist_ok=True)

def choose_indices(n: int, max_points: int, seed: int = 1234) -> np.ndarray:
    idx = np.arange(n)
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)
    return idx

# -----------------------------
# Color helpers
# -----------------------------
def colors_by_value(x2: np.ndarray, cmap_name: str = "viridis", decimals: int = 6):
    """
    좌표값(라운딩 후) -> 64비트 해시 -> [0,1] -> 컬러맵.
    같은 좌표면 플롯/순서가 달라도 같은 색을 보장.
    """
    cmap = plt.get_cmap(cmap_name)
    x_round = np.round(x2, decimals=decimals).astype(np.float32)

    vals = []
    for r in x_round:
        digest = hashlib.blake2b(r.tobytes(), digest_size=8).digest()  # 8 bytes = 64-bit
        vals.append(int.from_bytes(digest, byteorder="big", signed=False))
    h = np.array(vals, dtype=np.uint64)

    u = (h.astype(np.float64) - h.min()) / (h.max() - h.min() + 1e-12)
    return cmap(u)

def colors_by_id(ids: np.ndarray, cmap_name: str = "viridis"):
    """
    행 인덱스 기반 색. 같은 인덱스면 같은 색.
    """
    cmap = plt.get_cmap(cmap_name)
    ids = ids.astype(np.int64)
    u = (ids - ids.min()) / (ids.max() - ids.min() + 1e-12)
    return cmap(u)

# -----------------------------
# Plotters
# -----------------------------
def scatter_overlay_one_color(
    x0: np.ndarray,
    out_png: Union[str, Path],
    title: str = "DDIM Inversion → Resampling (overlay)",
    model_name: str = "teacher",
    max_points: int = 50000,
    cmap_name: str = "viridis",
    alpha: float = 0.9,
    point_size: float = 24.0,
    color_mode: str = "value",  # "value" | "id"
    seed: int = 1234,
    idx: np.ndarray = None,      # 외부 공유 인덱스
    colors: np.ndarray = None,   # 외부 공유 색
):
    """
    x0: [N, D]. 앞 2차원만 사용.
    color_mode="id" 이고 idx/colors를 외부에서 넘기면 같은 샘플은 항상 같은 색.
    """
    assert x0.ndim == 2 and x0.shape[1] >= 2

    # ① 인덱스 선택
    if idx is None:
        idx = choose_indices(x0.shape[0], max_points, seed=seed)
    x0_2 = x0[idx, :2]

    # ② 색 선정
    if colors is None:
        if color_mode == "id":
            colors = colors_by_id(idx, cmap_name)
        else:
            colors = colors_by_value(x0_2, cmap_name)

    plt.figure(figsize=(6, 6), dpi=160)
    plt.scatter(x0_2[:, 0], x0_2[:, 1], s=point_size, alpha=alpha, c=colors, edgecolors="none")
    plt.title(f"{title}\n({model_name} recon x₀)")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    x_min, y_min = x0_2.min(axis=0)
    x_max, y_max = x0_2.max(axis=0)
    dx = (x_max - x_min) * 0.05 + 1e-6
    dy = (y_max - y_min) * 0.05 + 1e-6
    plt.xlim([x_min - dx, x_max + dx])
    plt.ylim([y_min - dy, y_max + dy])

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def scatter_overlay_png(
    x0: np.ndarray,
    x0_rec: np.ndarray,
    out_png: Union[str, Path],
    title: str = "DDIM Inversion → Resampling (overlay)",
    max_points: int = 50000,
    alpha: float = 0.6,                 # ✅ 두 분포 모두 동일 투명도
    point_size: float = 7.0,
    seed: int = 1234,
    labels: Tuple[str, str] = ("A", "B"),
    idx: np.ndarray = None,             # 외부 공유 인덱스(있으면 사용)
    color_a: str = "tab:blue",          # ✅ A 색상
    color_b: str = "tab:orange",        # ✅ B 색상
):
    """
    x0, x0_rec: [N, D]. 두 분포를 서로 다른 색상으로, 동일한 투명도로 오버레이.
    """
    assert x0.ndim == 2 and x0_rec.ndim == 2 and x0.shape == x0_rec.shape and x0.shape[1] >= 2
    N = x0.shape[0]

    if idx is None:
        idx = choose_indices(N, max_points, seed=seed)

    x0_2 = x0[idx, :2]
    xr_2 = x0_rec[idx, :2]

    plt.figure(figsize=(6, 6), dpi=160)
    plt.scatter(x0_2[:, 0], x0_2[:, 1], s=point_size, alpha=alpha,
                c=color_a, edgecolors="none", label=labels[0])
    plt.scatter(xr_2[:, 0], xr_2[:, 1], s=point_size, alpha=alpha,
                c=color_b, edgecolors="none", label=labels[1])
    plt.legend(loc="best")
    plt.title(title)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    all_pts = np.concatenate([x0_2, xr_2], axis=0)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    dx = (x_max - x_min) * 0.05 + 1e-6
    dy = (y_max - y_min) * 0.05 + 1e-6
    plt.xlim([x_min - dx, x_max + dx])
    plt.ylim([y_min - dy, y_max + dy])

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def scatter_side_by_side(
    x_left: np.ndarray,
    x_right: np.ndarray,
    out_png: Union[str, Path],
    titles: Tuple[str, str] = ("student recon x₀", "teacher recon x₀"),
    main_title: str = "DDIM Inversion → Resampling (side-by-side)",
    max_points: int = 50000,
    cmap_name: str = "tab20",
    alpha: float = 0.9,
    point_size: float = 64.0,
    color_mode: str = "id",      # "id" 권장: 동일 인덱스 동일 색
    seed: int = 1234,
    idx: np.ndarray = None,      # 공유 인덱스 (있으면 사용)
    colors: np.ndarray = None,   # 공유 색상 (있으면 사용)
):
    """
    x_left, x_right: [N, D] (앞 2차원 사용)
    두 분포를 같은 팔레트로 1x2 패널에 그리되, 각 패널은 '자기 데이터 범위'로 축을 개별 설정.
    """
    assert x_left.ndim == 2 and x_right.ndim == 2 and x_left.shape[1] >= 2 and x_right.shape[1] >= 2
    assert x_left.shape[0] == x_right.shape[0], "row 수(N)가 달라서 1:1 색 매칭이 안 됩니다."

    N = x_left.shape[0]
    if idx is None:
        idx = choose_indices(N, max_points, seed=seed)

    xl = x_left[idx, :2]
    xr = x_right[idx, :2]

    # 색상 팔레트(공유)
    if colors is None:
        if color_mode == "id":
            colors = colors_by_id(idx, cmap_name=cmap_name)
        else:
            # value 모드는 패널별 값 분포가 달라 색이 달라질 수 있어 비교성이 떨어집니다.
            # 필요 시 아래 줄을 xr 기반으로 따로 만들거나 그대로 두세요.
            colors = colors_by_value(xl, cmap_name=cmap_name)

    def _lims(arr: np.ndarray):
        x_min, y_min = arr.min(axis=0)
        x_max, y_max = arr.max(axis=0)
        dx = (x_max - x_min) * 0.05 + 1e-6
        dy = (y_max - y_min) * 0.05 + 1e-6
        return (x_min - dx, x_max + dx), (y_min - dy, y_max + dy)

    xlim_l, ylim_l = _lims(xl)
    xlim_r, ylim_r = _lims(xr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=160)

    # Left (student)
    axes[0].scatter(xl[:, 0], xl[:, 1], s=point_size, alpha=alpha, c=colors, edgecolors="none")
    axes[0].set_title(titles[0])
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlim(xlim_l); axes[0].set_ylim(ylim_l)

    # Right (teacher)
    axes[1].scatter(xr[:, 0], xr[:, 1], s=point_size, alpha=alpha, c=colors, edgecolors="none")
    axes[1].set_title(titles[1])
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlim(xlim_r); axes[1].set_ylim(ylim_r)

    fig.suptitle(main_title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


# -----------------------------
# (Optional) normalization helpers
# -----------------------------
def load_norm_stats(json_path: Union[str, Path]):
    jp = Path(json_path)
    with jp.open("r") as f:
        d = json.load(f)  # {"mean":[...], "std":[...]}
    mu = np.array(d["mean"], dtype=np.float32)
    sigma = np.array(d["std"], dtype=np.float32)
    return mu, sigma

def normalize_np(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (arr - mu) / sigma

def denormalize_np(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return arr * sigma + mu

# -----------------------------
# Minimal teacher (2D toy)
# -----------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        t = t.float().unsqueeze(1)  # [B,1]
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device) * -1.0)
        angles = t * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb

class MLPDenoiser(nn.Module):
    """ε-predictor for 2D toy"""
    def __init__(self, in_dim=2, time_dim=64, hidden=256, depth=8, out_dim=2):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(time_dim)
        layers = []
        for i in range(depth):
            layers += [
                nn.Linear(in_dim + time_dim if i == 0 else hidden, hidden),
                nn.SiLU(),
            ]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out_dim)
    def forward(self, x, t):
        te = self.t_embed(t)
        h = torch.cat([x, te], dim=-1)
        return self.out(self.mlp(h))

# -----------------------------
# Model loading
# -----------------------------
def load_teacher_model(
    ckpt_path: str,
    device: torch.device,
    in_dim: int = 2,
    time_dim: int = 64,
    hidden: int = 256,
    depth: int = 8,
    out_dim: int = 2,
) -> nn.Module:
    """
    - torch.save(model)
    - torch.save(model.state_dict())
    """
    obj = torch.load(ckpt_path, map_location=device)

    if isinstance(obj, nn.Module):
        model = obj.to(device).eval()
        return model

    if isinstance(obj, dict):
        model = MLPDenoiser(in_dim=in_dim, time_dim=time_dim, hidden=hidden, depth=depth, out_dim=out_dim)
        missing, unexpected = model.load_state_dict(obj, strict=False)
        if len(missing) > 0:
            print("[WARN] Missing keys in state_dict:", missing)
        if len(unexpected) > 0:
            print("[WARN] Unexpected keys in state_dict:", unexpected)
        model = model.to(device).eval()
        return model

    raise RuntimeError("Unknown checkpoint format. Please save `model` or `model.state_dict()`.")

# -----------------------------
# Core: DDIM inversion + resampling
# -----------------------------
@torch.no_grad()
def ddim_inversion_and_resample(
    teacher: nn.Module,
    x0: torch.Tensor,              # [N, D] (model space)
    ddim: DDIMScheduler,
    inv: DDIMInverseScheduler,
    steps: int,
    device: torch.device,
    batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    return: (z_T, x0_recon) both [N, D] (model space)
    """
    N, _ = x0.shape
    if batch_size is None or batch_size <= 0:
        batch_size = N

    ddim.set_timesteps(steps, device=device)
    inv.set_timesteps(steps, device=device)

    z_all, xr_all = [], []

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        x0_b = x0[s:e].to(device)

        # 1) inversion
        lat_inv = x0_b.clone()
        for t in inv.timesteps:  # descending
            t_b = torch.full((lat_inv.shape[0],), int(t), device=device, dtype=torch.long)
            latent_in = inv.scale_model_input(lat_inv, t)
            eps = teacher(latent_in, t_b)
            lat_inv = inv.step(eps, t, lat_inv).prev_sample
        z_inv = lat_inv  # [B, D]

        # 2) sampling
        latents = z_inv.clone()
        for t in ddim.timesteps:  # descending
            t_b = torch.full((latents.shape[0],), int(t), device=device, dtype=torch.long)
            latent_in = ddim.scale_model_input(latents, t)
            eps = teacher(latent_in, t_b)
            latents = ddim.step(eps, t, latents, eta=0.0).prev_sample
        x0_rec = latents  # [B, D]

        z_all.append(z_inv.detach().cpu())
        xr_all.append(x0_rec.detach().cpu())

    z_out = torch.cat(z_all, dim=0)
    xr_out = torch.cat(xr_all, dim=0)
    return z_out, xr_out

@torch.no_grad()
def ddim_resample(
    teacher: nn.Module,
    xT: torch.Tensor,            # [N, D] (model space)
    ddim: DDIMScheduler,
    steps: int,
    device: torch.device,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    N, _ = xT.shape
    if batch_size is None or batch_size <= 0:
        batch_size = N

    ddim.set_timesteps(steps, device=device)

    xr_all = []
    xT = xT.to(device)

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        latents = xT[s:e].clone()
        for t in ddim.timesteps:
            t_b = torch.full((latents.shape[0],), int(t), device=device, dtype=torch.long)
            latent_in = ddim.scale_model_input(latents, t)
            eps = teacher(latent_in, t_b)
            latents = ddim.step(eps, t, latents, eta=0.0).prev_sample
        xr_all.append(latents.detach().cpu())

    xr_out = torch.cat(xr_all, dim=0)
    return xr_out

# -----------------------------
# Main
# -----------------------------

# ckpt_1021_1e4_denormalize_rkd_pdist_W0.1_x0_seq_b1024_ddim_50_150_steps_no_init_diff_W1.0_randN_student_step100000.pt

# runs/1023_lr1e4_b1024_ddim_50_150_steps_no_init_rkdW0.0_invW0.0_invinv_W1.0/ckpt_student_step075000.pt
# runs/1025_lr1e5_continue_b1024_ddim_150_250_steps_no_init_rkdW0.0_invW0.0_invinv_W1.0_diffW0.1/ckpt_student_step060000.pt


# runs/1025_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.0_invW0.0_invinv_W1.0_diffW0.1/ckpt_student_step070000.pt

# ckpt_1024_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.1_diff1.0_student_step040000.pt

# runs/1025_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.01_invW0.1_invinv_W1.0_diffW0.1/ckpt_student_step105000.pt



# 1027
# runs/1027_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.1_diff1.0/ckpt_student_step060000.pt
# runs/1027_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.01_invW0.0_invinv_W1.0_diffW0.0/ckpt_student_step105000.pt
# runs/1027_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.01_invW0.0_invinv_W1.0_diffW0.1/ckpt_student_step105000.pt
# runs/1027_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW1.0_diff0.0/ckpt_student_step065000.pt

# 1030
# runs/1030_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW1.0_invW1.0_invinv_W0.0_diffW0.0/ckpt_student_step015000.pt
# 1031
# runs/1031_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW1.0_invW1.0_invinv_W1.0_fid_W0.0001/ckpt_student_step015000.pt


# 1104 meeting
# runs/1101_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.1_invW0.1_invinv_W1.0_fid_W0.0001/ckpt_student_step060000.pt

# 1105
# runs/1102_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.1_invW1.0_invinv_W1.0_fid_W0.0001_reverse180/ckpt_student_step025000.pt



# # 1124
# 1124_lr1e4_n32_b2048_ddim_50_150_steps_no_init_rkdW0.1_invW0.0_invinvW0.0_fidW0.0_sameW0.0_x0_pred_rkd_with_teacher_x0_inv_only_x0
# 1124_lr1e4_n32_b2048_ddim_50_150_steps_no_init_rkdW0.1_invW0.0_invinvW0.0_fidW0.0_sameW1e-05_x0_pred_rkd_with_teacher_x0_inv_only_x0
# 1124_lr1e4_n32_b2048_ddim_50_150_steps_no_init_rkdW0.1_invW0.1_invinvW1.0_fidW0.0001_sameW0.0_x0_pred_rkd_with_teacher_x0_inv_only_x0

# runs/1121_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.1_invW0.1_invinvW1.0_fidW0.0001_sameW0.0001_x0_pred_rkd_with_teacher_x0_inv_only_x0/ckpt_student_step095000.pt

# 1201
# runs/1130_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.08_invW0.1_invinvW1.0_fidW0.0005_sameW0.0_x0_pred_rkd_with_teacher_x0_inv_only_x0/ckpt_student_step090000.pt
# runs/1201_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.08_invW0.1_invinvW1.0_fidW0.0005_sameW0.0001_x0_pred_rkd_with_teacher_x0_inv_only_x0/ckpt_student_step060000.pt

def main():
    parser = argparse.ArgumentParser(description="Batch DDIM Inversion → Resampling for .npy data")
    parser.add_argument("--npy", type=str, default="smile_data_n32_scale2_rot60_trans_50_-20/train.npy", help="Path to x0 .npy file (shape [N, D])")
    parser.add_argument("--teacher_ckpt", type=str, default="runs/1206_lr1e4_n32_b1024_T100_ddim_30_50_steps_no_init_rkdW0.08_invW0.1_invinvW1.0_fidW0.0005_sameW0.01_x0_pred_rkd_with_teacher_x0_inv_only_x0/ckpt_student_step155000.pt", help="Path to teacher checkpoint")
    parser.add_argument("--out_dir", type=str, default="inversion_test_1208_rkdW0.08_invW0.1_invinvW1.0_fidW0.0005_sameW0.01", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--T", type=int, default=100, help="num_train_timesteps used in training")
    parser.add_argument("--steps", type=int, default=40, help="DDIM steps (sampling/inversion)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inversion/sampling")
    parser.add_argument("--seed", type=int, default=42)

    # model arch
    parser.add_argument("--in_dim", type=int, default=2)
    parser.add_argument("--time_dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--out_dim", type=int, default=2)

    # optional normalization (teacher's training stats)
    parser.add_argument("--norm_json", type=str, default="smile_data_n32_scale2_rot60_trans_50_-20/normalization_stats.json", help="JSON with {'mean': [...], 'std': [...]}")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) 데이터 로드 (raw space)
    x0_raw = np.load(args.npy)  # [N, D]

    # x0_raw = x0_raw[:4]

    if x0_raw.ndim != 2:
        raise ValueError(f"Expected [N, D] array, but got shape {x0_raw.shape}")
    N, D = x0_raw.shape
    print(f"Loaded x0 from {args.npy}: shape = {x0_raw.shape}")

    # 2) (선택) 정규화
    use_norm = False
    mu = sigma = None
    if args.norm_json and Path(args.norm_json).exists():
        mu, sigma = load_norm_stats(args.norm_json)
        if mu.shape[0] != D or sigma.shape[0] != D:
            raise ValueError(f"norm_json dims {mu.shape}/{sigma.shape} do not match data dim {D}")
        x0_model_np = normalize_np(x0_raw, mu, sigma).astype(np.float32)
        use_norm = True
        print(f"[Norm] using {args.norm_json}")
    else:
        x0_model_np = x0_raw.astype(np.float32)

    # teacher 전용 정규화 (teacher raw로 되돌릴 때 사용)
    mu_teacher, sigma_teacher = load_norm_stats("smile_data_n65536_scale10_rot0_trans_0_0/normalization_stats.json")

    x0_t = torch.from_numpy(x0_model_np).float()

    # 3) 모델 & 스케줄러 준비
    teacher = load_teacher_model(
        args.teacher_ckpt, device,
        in_dim=args.in_dim, time_dim=args.time_dim,
        hidden=args.hidden, depth=args.depth, out_dim=args.out_dim
    )
    teacher.eval()

    ddim = DDIMScheduler(
        num_train_timesteps=args.T,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=0,
    )
    inv = DDIMInverseScheduler.from_config(ddim.config)

    # 4) Inversion → Resampling (model space)
    z_T, x0_rec_model_student = ddim_inversion_and_resample(
        teacher=teacher,
        x0=x0_t,
        ddim=ddim,
        inv=inv,
        steps=args.steps,
        device=device,
        batch_size=args.batch_size,
    )

    # 5) 원 스케일 복원 (시각화/저장용)
    if use_norm:
        x0_rec_raw_student = denormalize_np(x0_rec_model_student.numpy(), mu, sigma)
        x0_for_plot = x0_raw
    else:
        x0_rec_raw_student = x0_rec_model_student.numpy()
        x0_for_plot = x0_raw  # (미사용)


    png_path = Path(args.out_dir) / "GT_vs_RECON.png"
    scatter_overlay_png(
        x0=x0_raw,
        x0_rec=x0_rec_raw_student,
        out_png=str(png_path),
        title=f"GT → DDIM Inversion → DDIM Sampling (steps={args.steps})\n(GT vs student recon)",
        alpha=0.6,                      # 두 분포 동일 투명도
        point_size=7.0,
        seed=args.seed,
        labels=("GT", "RECON"),
        idx=None,                       # 전부 그릴 거면 생략 가능
        color_a="tab:blue",             # 학생 색
        color_b="tab:orange",           # 티처 색
        max_points=x0_rec_raw_student.shape[0],  # 모두 그리기 원하면
    )

    # ===== 공유 인덱스/색 팔레트 생성 =====
    # 동일 z_T로부터 나온 student/teacher 샘플은 인덱스가 1:1 대응 → 같은 인덱스 = 같은 색
    shared_idx = choose_indices(N, max_points=min(N, 50000), seed=args.seed)
    # shared_colors = colors_by_id(shared_idx, cmap_name="tab20")
    shared_colors = colors_by_id32(shared_idx, alpha=0.85)
    
    # 6) Student 단일 분포 (ID 기반, 공유 팔레트)
    png_path = Path(args.out_dir) / "student_recon.png"
    scatter_overlay_one_color(
        x0=x0_rec_raw_student,
        out_png=str(png_path),
        title=f"DDIM Inversion→Resampling (steps={args.steps})",
        model_name="student",
        color_mode="id",
        seed=args.seed,
        idx=shared_idx,
        colors=shared_colors,
    )
    print(f"Saved figure: {png_path}")

    # 7) Real teacher 로드 & 같은 z_T로 복원
    real_teacher = load_teacher_model(
        "ckpt_teacher_B1024_N65536_T100_step1000000.pt", device,
        in_dim=args.in_dim, time_dim=args.time_dim,
        hidden=args.hidden, depth=args.depth, out_dim=args.out_dim
    )
    real_teacher.eval()

    x0_rec_model_teacher = ddim_resample(
        teacher=real_teacher,
        xT=z_T,
        ddim=ddim,
        steps=args.steps,
        device=device,
        batch_size=args.batch_size,
    )

    if use_norm:
        x0_rec_raw_teacher = denormalize_np(x0_rec_model_teacher.numpy(), mu_teacher, sigma_teacher)
    else:
        x0_rec_raw_teacher = x0_rec_model_teacher.numpy()

    # 8) Teacher 단일 분포 (ID 기반, 공유 팔레트)
    png_path = Path(args.out_dir) / "teacher_recon.png"
    scatter_overlay_one_color(
        x0=x0_rec_raw_teacher,
        out_png=str(png_path),
        title=f"DDIM Inversion→Resampling (steps={args.steps})",
        model_name="teacher",
        color_mode="id",
        seed=args.seed,
        idx=shared_idx,
        colors=shared_colors,
    )
    print(f"Saved figure: {png_path}")

    # 9) 동일 노이즈로 teacher vs student 비교
    noise_zT = torch.randn((512, *z_T.shape[1:]), device=z_T.device, dtype=z_T.dtype)
    teacher_rec_model = ddim_resample(
        teacher=real_teacher,
        xT=noise_zT,
        ddim=ddim,
        steps=args.steps,
        device=device,
        batch_size=args.batch_size,
    )
    student_rec_model = ddim_resample(
        teacher=teacher,
        xT=noise_zT,
        ddim=ddim,
        steps=args.steps,
        device=device,
        batch_size=args.batch_size,
    )

    if use_norm:
        teacher_rec_raw = denormalize_np(teacher_rec_model.numpy(), mu_teacher, sigma_teacher)
        student_rec_raw = denormalize_np(student_rec_model.numpy(), mu, sigma)
    else:
        teacher_rec_raw = teacher_rec_model.numpy()
        student_rec_raw = student_rec_model.numpy()

    # 10) 동일 노이즈 teacher vs student (ID 기반, 공유 팔레트)
    png_path = Path(args.out_dir) / "same_noise_teacher_vs_student.png"
    scatter_overlay_png(
        x0=student_rec_raw,
        x0_rec=teacher_rec_raw,
        out_png=str(png_path),
        title=f"Same Noise → DDIM Resampling (steps={args.steps})\n(student vs teacher)",
        alpha=0.6,                      # 두 분포 동일 투명도
        point_size=7.0,
        seed=args.seed,
        labels=("student", "teacher"),
        idx=None,                       # 전부 그릴 거면 생략 가능
        color_a="tab:blue",             # 학생 색
        color_b="tab:orange",           # 티처 색
        max_points=student_rec_raw.shape[0],  # 모두 그리기 원하면
    )

    print(f"Saved figure: {png_path}")


    side_by_side_png = Path(args.out_dir) / "student_teacher_side_by_side.png"
    scatter_side_by_side(
        x_left=x0_rec_raw_student,
        x_right=x0_rec_raw_teacher,
        out_png=str(side_by_side_png),
        titles=("student recon x₀", "teacher recon x₀"),
        main_title=f"DDIM Inversion→Resampling (steps={args.steps}) – student vs teacher",
        color_mode="id",
        seed=args.seed,
        idx=shared_idx,
        colors=shared_colors,
    )
    print(f"Saved figure: {side_by_side_png}")




    # 동일 노이즈 동일 색
    NN=32
    noise_zT = torch.randn((NN, *z_T.shape[1:]), device=z_T.device, dtype=z_T.dtype)
    x0_rec_model_teacher_ = ddim_resample(
        teacher=real_teacher,
        xT=noise_zT,
        ddim=ddim,
        steps=args.steps,
        device=device,
        batch_size=args.batch_size,
    )
    x0_rec_model_student_ = ddim_resample(
        teacher=teacher,
        xT=noise_zT,
        ddim=ddim,
        steps=args.steps,
        device=device,
        batch_size=args.batch_size,
    )
    if use_norm:
        x0_rec_raw_teacher_ = denormalize_np(x0_rec_model_teacher_.numpy(), mu_teacher, sigma_teacher)
        x0_rec_raw_student_ = denormalize_np(x0_rec_model_student_.numpy(), mu, sigma)
    else:
        x0_rec_raw_teacher_ = x0_rec_model_teacher_.numpy()
        x0_rec_raw_student_ = x0_rec_model_student_.numpy()



    shared_idx_ = choose_indices(NN, max_points=min(N, 50000), seed=args.seed)
    # shared_colors_ = colors_by_id(shared_idx_, cmap_name="tab20")
    shared_colors_ = colors_by_id32(shared_idx_, alpha=0.85)

    side_by_side_png = Path(args.out_dir) / "same_pure_noise_same_color_student_teacher.png"
    scatter_side_by_side(
        x_left=x0_rec_raw_student_,
        x_right=x0_rec_raw_teacher_,
        out_png=str(side_by_side_png),
        titles=("student recon x₀", "teacher recon x₀"),
        main_title=f"Same Gaussian Noise → DDIMsampling (steps={args.steps}) – teacher vs student",
        color_mode="id",
        seed=args.seed,
        idx=shared_idx_,
        colors=shared_colors_,
    )
    print(f"Saved figure: {side_by_side_png}")



    # 11) 재구성 오차 리포트
    mse_model = float(((x0_rec_model_student - x0_t).pow(2).mean()).cpu().item())
    print(f"[Reconstruction MSE | model space] {mse_model:.6f}")
    if use_norm:
        mse_raw = float(((torch.from_numpy(x0_rec_raw_student) - torch.from_numpy(x0_raw)).pow(2).mean()).item())
        print(f"[Reconstruction MSE | raw space]   {mse_raw:.6f}")

if __name__ == "__main__":
    main()
