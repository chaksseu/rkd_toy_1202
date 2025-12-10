#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Student DDIM Trajectory Visualizer (norm / denorm / H-Transform) + Pure Score Field

- STUDENT(MLP epsilon-predictor) 로드
- DDIM으로 x_T -> x_0 궤적 수집 및 저장 (norm / denorm / H 모두: NPY / overlay / frames / GIF)
- LearnableHomography (H-module) ckpt를 불러와서, 학생 normalized 공간에서 H 적용
- 모델 자체의 score(vector field) 시각화: streamplot + contour + quiver (norm / denorm)
"""

import os, re, math, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, DDIMScheduler

# ===================== CONFIG ===================== #
CUDA_NUM = 0
BATCH_SIZE = 1024
WANDB_NAME = f"1117_lr1e4_n32_b{BATCH_SIZE}_ddim_50_150_steps"

CONFIG = {
    "device": f"cuda:{CUDA_NUM}",
    "out_dir": f"runs/{WANDB_NAME}",
    "T": 100,
    "seed": 42,
    "dim": 2,
    "student_hidden": 256,
    "student_depth": 8,
    "student_time_dim": 64,
    "auto_find_ckpt_pattern": r"ckpt_student_step(\d+)\.pt",
}

# ===================== UTILS ===================== #
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def find_latest_student_ckpt(out_dir: Path, pattern: str):
    regex = re.compile(pattern)
    best_num, best_path = -1, None
    if not out_dir.exists():
        return None
    for f in out_dir.glob("ckpt_student_step*.pt"):
        m = regex.fullmatch(f.name)
        if m:
            step = int(m.group(1))
            if step > best_num:
                best_num, best_path = step, f
    return str(best_path) if best_path else None

def _square_limits_from(data: np.ndarray, pad_ratio: float = 0.05):
    """
    데이터에서 finite 값만 모아서 x/y 범위 계산.
    모든 값이 NaN/Inf이면 기본 [-1,1] 범위를 반환해 에러를 피한다.
    """
    data = np.asarray(data, dtype=np.float64)
    mask = np.isfinite(data).all(axis=1)
    if not np.any(mask):
        # fallback 범위
        return (-1.0, 1.0), (-1.0, 1.0)

    data = data[mask]
    xmin, xmax = float(data[:, 0].min()), float(data[:, 0].max())
    ymin, ymax = float(data[:, 1].min()), float(data[:, 1].max())

    dx, dy = xmax - xmin, ymax - ymin
    base = max(dx, dy, 1e-3)
    pad = pad_ratio * base

    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    half = span / 2.0

    return (xmid - half, xmid + half), (ymid - half, ymid + half)

def colors_from_noise_ids(ids: np.ndarray, alpha: float = 0.9):
    ids = np.asarray(ids, dtype=np.int64)
    phi = 0.6180339887498949
    hues = (ids * phi) % 1.0
    cmap = plt.get_cmap("hsv")
    cols = cmap(hues)
    cols[:, 3] = alpha
    return cols

def load_norm_stats(json_path: str):
    p = Path(json_path)
    with p.open("r") as f:
        d = json.load(f)
    mu = np.array(d["mean"], dtype=np.float32)
    sg = np.array(d["std"],  dtype=np.float32)
    return mu, sg

# ===================== LEARNABLE H (H-module) ===================== #
class LearnableHomography(nn.Module):
    """
    Row-vector convention: [x, y, 1] @ H^T -> [X, Y, W], (x',y') = (X/W, Y/W)

    - H: 하나의 3x3 행렬만 학습 (shape: (3,3))
    - 모든 timestep / 모든 sample에서 같은 H를 사용
    - t 인자는 호환성을 위해 남겨두지만 사용하지 않음
    """
    def __init__(
        self,
        init_9=None,
        eps: float = 1e-6,
        fix_last_row: bool = False,
    ):
        super().__init__()
        self.eps = float(eps)
        self.fix_last_row = bool(fix_last_row)  # True면 마지막 행을 [0,0,1]로 고정(affine)

        if init_9 is None:
            I = torch.eye(3, dtype=torch.float32)    # (3,3)
        else:
            I = torch.tensor(init_9, dtype=torch.float32).view(3, 3)

        # 단일 (3,3) H
        self.H = nn.Parameter(I)                     # (3,3)

    def _get_H(self) -> torch.Tensor:
        """
        return: (3,3) homography matrix
        """
        H = self.H
        if self.fix_last_row:
            H = H.clone()
            H[2, :2] = 0.0
            H[2, 2]  = 1.0
        return H

    def forward(self, xy: torch.Tensor, t=None):
        """
        xy: (B,2)
        t : 사용하지 않지만 인터페이스 호환성 유지용

        returns:
            xy_trans: (B,2)
            w:        (B,1)
        """
        B = xy.shape[0]
        device = xy.device
        ones = torch.ones(B, 1, device=device, dtype=xy.dtype)
        homo = torch.cat([xy, ones], dim=-1)          # (B,3)

        H = self._get_H()                             # (3,3)
        # (B,3) @ (3,3)^T -> (B,3)
        out = homo @ H.transpose(0, 1)                # (B,3)

        w   = out[:, 2:3]
        den = w.sign() * torch.clamp(w.abs(), min=self.eps)
        xy_t = out[:, :2] / den                       # (B,2)

        return xy_t, w

@torch.no_grad()
def apply_H_module_to_seq_per_t(seq_norm: np.ndarray,
                                ts: np.ndarray,
                                H_module: nn.Module,
                                device: torch.device) -> np.ndarray:
    """
    seq_norm : [K,B,2] (normalized student 좌표)
    ts       : [K] (int timesteps)
    return   : [K,B,2]

    - 학습 코드와 동일하게, 전체 시퀀스 중 마지막 스텝(K-1)에만 H를 적용
      (나머지 스텝은 그대로 둠)
    - NaN/Inf는 클리핑해서 이후 plotting에서 에러 안 나도록 처리.
    """
    K, B, _ = seq_norm.shape
    outs = []
    H_module.eval()
    for k in range(K):
        xk = torch.from_numpy(seq_norm[k]).to(device=device, dtype=torch.float32)  # (B,2)

        # 기본은 identity (그대로 두고)
        xk_H = xk
        # 마지막 스텝에만 H 적용 (학습 코드의 apply_H_to_seq_per_t와 동일)
        if k == K - 1:
            xk_H, _ = H_module(xk)

        x_np = xk_H.detach().cpu().numpy()
        # NaN/Inf 보호
        mask_finite = np.isfinite(x_np)
        if not np.all(mask_finite):
            finite_vals = x_np[mask_finite]
            if finite_vals.size == 0:
                x_np = np.zeros_like(x_np, dtype=np.float32)
            else:
                mean = finite_vals.mean()
                std = finite_vals.std() + 1e-6
                x_np = np.nan_to_num(
                    x_np,
                    nan=mean,
                    posinf=mean + 5 * std,
                    neginf=mean - 5 * std,
                )
        outs.append(x_np.astype(np.float32))
    return np.stack(outs, axis=0)

# ===================== MODEL ===================== #
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        t = t.float().unsqueeze(1)
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device) * -1.0)
        angles = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class MLPDenoiser(nn.Module):
    def __init__(self, in_dim=2, time_dim=64, hidden=256, depth=8, out_dim=2):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(time_dim)
        layers = []
        for i in range(depth):
            layers += [nn.Linear(in_dim + time_dim if i == 0 else hidden, hidden), nn.SiLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out_dim)
    def forward(self, x, t):
        te = self.t_embed(t)
        h = torch.cat([x, te], dim=-1)
        return self.out(self.mlp(h))  # ε(x,t)

# ===================== SCHEDULERS ===================== #
def build_schedulers(num_train_timesteps: int):
    train_sched = DDPMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")
    train_sched.config.prediction_type = "epsilon"
    sample_sched = DDIMScheduler.from_config(train_sched.config)
    sample_sched.config.clip_sample = False
    sample_sched.config.prediction_type = "epsilon"
    return train_sched, sample_sched

# ===================== SAMPLING (TRAJ) ===================== #
@torch.no_grad()
def collect_student_xt_seq_ddim(student: nn.Module,
                                sample_sched: DDIMScheduler,
                                z: torch.Tensor,
                                device: torch.device,
                                sample_steps: int = 100,
                                eta: float = 0.0):
    sched = DDIMScheduler.from_config(sample_sched.config)
    sched.set_timesteps(sample_steps, device=device)
    x = z.to(device).detach().clone()
    B = x.shape[0]
    xs, ts = [], []
    was_train = student.training
    student.eval()
    try:
        for t in sched.timesteps:
            t_int = int(t)
            ts.append(t_int)
            t_b = torch.full((B,), t_int, device=device, dtype=torch.long)
            xin = sched.scale_model_input(x, t)
            eps = student(xin, t_b)
            out = sched.step(model_output=eps, timestep=t, sample=x, eta=eta)
            x = out.prev_sample
            xs.append(x.detach().cpu().numpy())
    finally:
        if was_train:
            student.train()
    return np.stack(xs, 0), np.asarray(ts, dtype=int)

# ===================== SCORE FIELD ===================== #
def _alpha_bar_at(sched: DDIMScheduler, t_int: int) -> float:
    return float(sched.alphas_cumprod[int(t_int)].item())

@torch.no_grad()
def compute_score_grid_norm(student: nn.Module,
                            sample_sched: DDIMScheduler,
                            t_int: int,
                            xlim, ylim,
                            grid: int,
                            device: torch.device):
    xs = np.linspace(xlim[0], xlim[1], grid, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], grid, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.stack([Xg.reshape(-1), Yg.reshape(-1)], axis=1)

    x = torch.from_numpy(pts).to(device)
    t_b = torch.full((x.shape[0],), int(t_int), device=device, dtype=torch.long)
    t_tensor = torch.tensor(int(t_int), device=device, dtype=torch.long)

    xin = sample_sched.scale_model_input(x, t_tensor)
    eps = student(xin, t_b)  # ε(x,t)
    sigma_t = math.sqrt(1.0 - _alpha_bar_at(sample_sched, t_int)) + 1e-12
    s = -eps / sigma_t  # score ≈ -ε/σ_t

    Sx = s[:, 0].reshape(Xg.shape).detach().cpu().numpy()
    Sy = s[:, 1].reshape(Xg.shape).detach().cpu().numpy()
    Sm = np.sqrt(Sx * Sx + Sy * Sy)
    # NaN/Inf 방지
    Sm = np.nan_to_num(Sm, nan=0.0, posinf=np.max(Sm[np.isfinite(Sm)]) if np.isfinite(Sm).any() else 1.0, neginf=0.0)
    return Xg, Yg, Sx, Sy, Sm

def render_score_field_both(seq_norm: np.ndarray, ts: np.ndarray,
                            student: nn.Module, sample_sched: DDIMScheduler,
                            out_dir: Path, grid: int = 60, stream_density: float = 1.2,
                            mu: np.ndarray = None, sg: np.ndarray = None,
                            t_list: list = None, device: torch.device = torch.device("cpu")):
    """
    모델 자체의 score field 시각화 (샘플 의존 X):
    - norm: streamplot + contour + quiver
    - denorm: 동일(존재하면)
    """
    K, B, _ = seq_norm.shape
    XY_all_norm = seq_norm.reshape(-1, 2)
    xlim_n, ylim_n = _square_limits_from(XY_all_norm, pad_ratio=0.06)

    have_den = (mu is not None) and (sg is not None)
    if have_den:
        seq_den = seq_norm * sg[None, None, :] + mu[None, None, :]
        XY_all_den = seq_den.reshape(-1, 2)
        xlim_d, ylim_d = _square_limits_from(XY_all_den, pad_ratio=0.06)

    out_norm = ensure_dir(out_dir / "score_field_norm")
    out_den  = ensure_dir(out_dir / "score_field_denorm") if have_den else None

    if not t_list:
        t_list = np.unique(np.linspace(0, len(ts) - 1, num=5, dtype=int)).tolist()
        t_list = [int(ts[i]) for i in t_list]

    for t_int in t_list:
        idx = np.where(ts == t_int)[0]
        k = int(idx[0]) if len(idx) > 0 else int(np.argmin(np.abs(ts - t_int)))
        t_used = int(ts[k])

        # ---- norm field ----
        Xg, Yg, Sx, Sy, Sm = compute_score_grid_norm(student, sample_sched, t_used,
                                                     xlim_n, ylim_n, grid, device)

        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        cs = ax.contour(Xg, Yg, Sm, levels=16, linewidths=1.2, cmap="magma", alpha=0.95)
        ax.streamplot(Xg, Yg, Sx, Sy, color=Sm, cmap="magma",
                      density=stream_density, linewidth=1.0, arrowsize=1.2)
        step = max(1, grid // 18)
        ax.quiver(Xg[::step, ::step], Yg[::step, ::step],
                  Sx[::step, ::step], Sy[::step, ::step],
                  angles='xy', scale_units='xy', scale=1.0,
                  width=0.003, color='k', alpha=0.9)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim_n)
        ax.set_ylim(*ylim_n)
        ax.set_title(f"Score field (norm) @ t={t_used}")
        fig.colorbar(cs, ax=ax, shrink=0.82, label="||score||")
        fig.tight_layout()
        fig.savefig(out_norm / f"score_field_t{t_used:04d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ---- denorm field ----
        if have_den:
            Xg_d = Xg * sg[0] + mu[0]
            Yg_d = Yg * sg[1] + mu[1]
            Sx_d = Sx / max(sg[0], 1e-12)
            Sy_d = Sy / max(sg[1], 1e-12)
            Sm_d = np.sqrt(Sx_d * Sx_d + Sy_d * Sy_d)
            Sm_d = np.nan_to_num(Sm_d, nan=0.0,
                                 posinf=np.max(Sm_d[np.isfinite(Sm_d)]) if np.isfinite(Sm_d).any() else 1.0,
                                 neginf=0.0)

            fig, ax = plt.subplots(figsize=(6.2, 6.2))
            cs = ax.contour(Xg_d, Yg_d, Sm_d, levels=16, linewidths=1.2, cmap="magma", alpha=0.95)
            ax.streamplot(Xg_d, Yg_d, Sx_d, Sy_d, color=Sm_d, cmap="magma",
                          density=stream_density, linewidth=1.0, arrowsize=1.2)
            step = max(1, grid // 18)
            ax.quiver(Xg_d[::step, ::step], Yg_d[::step, ::step],
                      Sx_d[::step, ::step], Sy_d[::step, ::step],
                      angles='xy', scale_units='xy', scale=1.0,
                      width=0.003, color='k', alpha=0.9)

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(*xlim_d)
            ax.set_ylim(*ylim_d)
            ax.set_title(f"Score field (denorm) @ t={t_used}")
            fig.colorbar(cs, ax=ax, shrink=0.82, label="||score||")
            fig.tight_layout()
            fig.savefig(out_den / f"score_field_t{t_used:04d}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

# ===================== PLOTTING (TRAJ, FRAMES, GIF) ===================== #
def plot_trajectories_overlay(seq_xy: np.ndarray, out_path: Path,
                              max_lines: int = 512, line_alpha: float = 0.85,
                              dot_size: int = 14, pad_ratio: float = 0.06,
                              title: str = None):
    K, B, _ = seq_xy.shape
    cols = colors_from_noise_ids(np.arange(B), alpha=line_alpha)
    XY_all = seq_xy.reshape(-1, 2)
    xlim, ylim = _square_limits_from(XY_all, pad_ratio=pad_ratio)

    pick = np.linspace(0, B - 1, num=min(B, max_lines), dtype=int)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for b in pick:
        xy = seq_xy[:, b, :]
        ax.plot(xy[:, 0], xy[:, 1], lw=1.2, alpha=line_alpha, c=cols[b])
        ax.scatter(xy[0, 0], xy[0, 1], s=dot_size, c=[cols[b]], marker='o',
                   edgecolors='k', linewidths=0.4, zorder=3)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=dot_size, c=[cols[b]], marker='X',
                   edgecolors='k', linewidths=0.4, zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def save_per_timestep_frames(seq_xy: np.ndarray, ts: np.ndarray, out_dir: Path, dot_size: int = 10):
    K, B, _ = seq_xy.shape
    cols = colors_from_noise_ids(np.arange(B), alpha=0.9)
    XY_all = seq_xy.reshape(-1, 2)
    xlim, ylim = _square_limits_from(XY_all, pad_ratio=0.06)
    frames_dir = ensure_dir(out_dir)
    for k in range(K):
        xy = seq_xy[k]
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.scatter(xy[:, 0], xy[:, 1], s=dot_size, c=cols, edgecolors="none")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"Student $pred x_0$ (t={int(ts[k])})")
        fig.tight_layout()
        fig.savefig(frames_dir / f"t{int(ts[k]):04d}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    return frames_dir

def maybe_make_gif(frames_dir: Path, out_path: Path, fps: int = 10):
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[GIF] imageio not installed; skip GIF. `pip install imageio` to enable.")
        return
    files = list(frames_dir.glob("t*.png"))
    if not files:
        print("[GIF] no frames found.")
        return

    def _t_of(p: Path):
        st = p.stem
        return int(st[1:]) if st.startswith("t") and st[1:].isdigit() else -1

    files_sorted = sorted(files, key=_t_of, reverse=True)  # 큰 t -> 0
    imgs = []
    for p in files_sorted:
        try:
            imgs.append(imageio.imread(str(p)))
        except Exception as e:
            print(f"[GIF] skip frame {p} due to error: {e}")
    if not imgs:
        print("[GIF] no valid frames after loading; skip.")
        return
    imageio.mimsave(str(out_path), imgs, duration=1.0 / max(fps, 1))
    print(f"[GIF] saved (large t -> 0) -> {out_path}")

# ===================== MAIN LOGIC ===================== #
@torch.no_grad()
def visualize_student_ddim_trajectories(cfg: dict,
                                        student_ckpt_path: str,
                                        n_noises: int = 128,
                                        ddim_steps: int = 100,
                                        eta: float = 0.0,
                                        save_frames: bool = True,
                                        make_gif_flag: bool = True,
                                        out_dir_override: str = None,
                                        student_stats_path: str = None,
                                        score_ts: str = "",
                                        score_grid: int = 60,
                                        score_density: float = 1.2,
                                        H_ckpt_path: str = None,
                                        H_eps: float = 1e-6):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    out_root = Path(out_dir_override) if out_dir_override else Path(cfg["out_dir"])
    out_traj = ensure_dir(out_root / "traj")

    # schedulers + student
    _, sample_sched = build_schedulers(cfg["T"])
    student = MLPDenoiser(in_dim=cfg["dim"], time_dim=cfg["student_time_dim"],
                          hidden=cfg["student_hidden"], depth=cfg["student_depth"],
                          out_dim=cfg["dim"]).to(device)
    student.load_state_dict(torch.load(student_ckpt_path, map_location=device), strict=True)
    student.eval()
    print(f"[LOAD] Student checkpoint -> {student_ckpt_path}")

    # noise & collect (normalized)
    z = torch.randn(n_noises, cfg["dim"], device=device)
    seq_norm, ts = collect_student_xt_seq_ddim(student, sample_sched, z, device,
                                               sample_steps=int(ddim_steps), eta=float(eta))
    K, B, _ = seq_norm.shape

    # ---------- (1) norm ----------
    np.save(out_traj / "student_traj_xy_norm.npy", seq_norm)
    np.save(out_traj / "student_traj_ts.npy", ts)
    plot_trajectories_overlay(
        seq_norm,
        out_traj / f"student_traj_norm_steps{K}_N{B}.png",
        max_lines=512,
        title=f"Student DDIM (norm) steps={K} N={B} eta={eta}",
    )
    if save_frames:
        f_norm = ensure_dir(out_traj / "frames_norm")
        save_per_timestep_frames(seq_norm, ts, f_norm)
        if make_gif_flag:
            maybe_make_gif(f_norm, out_traj / f"student_traj_norm_steps{K}_N{B}.gif", fps=10)

    # ---------- (2) H-module (norm + H) ----------
    seq_norm_H = None
    H_module = None
    if H_ckpt_path is not None and len(str(H_ckpt_path)) > 0:
        H_ckpt_path = str(H_ckpt_path)
        if not Path(H_ckpt_path).exists():
            print(f"[H] H_ckpt not found: {H_ckpt_path} -> skip H application.")
        else:
            print(f"[H] Loading LearnableHomography from: {H_ckpt_path}")
            H_module = LearnableHomography(
                init_9=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                eps=H_eps,
                fix_last_row=False,
            ).to(device)
            state = torch.load(H_ckpt_path, map_location=device)
            H_module.load_state_dict(state, strict=True)
            H_module.eval()

            seq_norm_H = apply_H_module_to_seq_per_t(seq_norm, ts, H_module, device)
            np.save(out_traj / "student_traj_xy_norm_H.npy", seq_norm_H)
            plot_trajectories_overlay(
                seq_norm_H,
                out_traj / f"student_traj_norm_H_steps{K}_N{B}.png",
                max_lines=512,
                title=f"Student DDIM (norm + H) steps={K} N={B} eta={eta}",
            )
            if save_frames:
                f_norm_H = ensure_dir(out_traj / "frames_norm_H")
                save_per_timestep_frames(seq_norm_H, ts, f_norm_H)
                if make_gif_flag:
                    maybe_make_gif(
                        f_norm_H,
                        out_traj / f"student_traj_norm_H_steps{K}_N{B}.gif",
                        fps=10,
                    )

    # ---------- (3) denorm ----------
    mu, sg = (None, None)
    seq_den = None
    seq_den_H = None
    if student_stats_path and Path(student_stats_path).exists():
        mu, sg = load_norm_stats(student_stats_path)

        # 기본 denorm (student 도메인)
        seq_den = seq_norm * sg[None, None, :] + mu[None, None, :]
        np.save(out_traj / "student_traj_xy_denorm.npy", seq_den)
        plot_trajectories_overlay(
            seq_den,
            out_traj / f"student_traj_denorm_steps{K}_N{B}.png",
            max_lines=512,
            title=f"Student DDIM (denorm) steps={K} N={B} eta={eta}",
        )
        if save_frames:
            f_den = ensure_dir(out_traj / "frames_denorm")
            save_per_timestep_frames(seq_den, ts, f_den)
            if make_gif_flag:
                maybe_make_gif(
                    f_den,
                    out_traj / f"student_traj_denorm_steps{K}_N{B}.gif",
                    fps=10,
                )

        # H 적용 후 denorm
        if seq_norm_H is not None:
            seq_den_H = seq_norm_H * sg[None, None, :] + mu[None, None, :]
            np.save(out_traj / "student_traj_xy_denorm_H.npy", seq_den_H)
            plot_trajectories_overlay(
                seq_den_H,
                out_traj / f"student_traj_denorm_H_steps{K}_N{B}.png",
                max_lines=512,
                title=f"Student DDIM (denorm + H) steps={K} N={B} eta={eta}",
            )
            if save_frames:
                f_den_H = ensure_dir(out_traj / "frames_denorm_H")
                save_per_timestep_frames(seq_den_H, ts, f_den_H)
                if make_gif_flag:
                    maybe_make_gif(
                        f_den_H,
                        out_traj / f"student_traj_denorm_H_steps{K}_N{B}.gif",
                        fps=10,
                    )
    else:
        print("[DENORM] --student-stats not provided or file missing -> skip denorm trajectory outputs.")

    # ---------- (4) Pure score field (norm & denorm) ----------
    if score_ts:
        t_list = [int(s) for s in score_ts.split(",") if s.strip().isdigit()]
    else:
        t_list = None  # auto-pick (5개)
    render_score_field_both(seq_norm=seq_norm, ts=ts,
                            student=student, sample_sched=sample_sched,
                            out_dir=out_traj, grid=int(score_grid),
                            stream_density=float(score_density),
                            mu=mu, sg=sg, t_list=t_list, device=device)

    print(f"[DONE] out dir: {out_traj.resolve()}")


# ===================== ARGS / ENTRY ===================== #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Student DDIM trajectories + pure score field (norm / denorm / H)."
    )
    parser.add_argument("--ckpt", type=str,
        default=f"runs/1209_lr1e4_n32_H_b1024_T100_ddim_30_50_steps_no_init_rkdW0.08_invW0.1_invinvW1.0_fidW0.1_sameW0.1_x0_pred_rkd_with_teacher_x0_inv_only_x0_no_norm/ckpt_student_step075000.pt",
        help="Path to student checkpoint .pt."
    )
    parser.add_argument("--H-ckpt", type=str, 
        default="runs/1209_lr1e4_n32_H_b1024_T100_ddim_30_50_steps_no_init_rkdW0.08_invW0.1_invinvW1.0_fidW0.1_sameW0.1_x0_pred_rkd_with_teacher_x0_inv_only_x0_no_norm/ckpt_H_step075000.pt",
        help="Path to ckpt_H_stepXXXXX.pt (LearnableHomography state_dict)."
    )
    parser.add_argument("--out", type=str,
                        default="vis_traj_H0_1210_rkd0.08_x0_inv0.1_invinv1.0_fd0.1_same0.1",
                        help="Output directory root.")

    parser.add_argument("--n", type=int, default=32, help="Number of pure noise samples.")
    parser.add_argument("--steps", type=int, default=40, help="DDIM sampling steps.")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta.")
    parser.add_argument("--frames", type=bool, default=True,
                        help="Save per-timestep frames for norm / denorm / H.")
    parser.add_argument("--gif", type=bool, default=True,
                        help="Make GIFs for norm / denorm / H.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--student-stats", type=str,
                        default="smile_data_n8192_scale10_rot0_trans_0_0_H_32/normalization_stats.json",
                        # default="smile_data_n32_scale2_rot60_trans_50_-20/normalization_stats.json",
                        help="JSON with {'mean':[...],'std':[...]} for student denorm.")

    # Pure score field 옵션
    parser.add_argument("--score-ts", type=str, default="",
                        help="Comma-separated raw t's for score field (e.g., '999,750,500,250,0'). Empty -> auto-pick 5.")
    parser.add_argument("--score-grid", type=int, default=80,
                        help="Grid resolution for score field.")
    parser.add_argument("--score-density", type=float, default=1.2,
                        help="Streamplot density.")

    # H-module ckpt
    parser.add_argument("--H-eps", type=float, default=1e-8,
                        help="Small epsilon used inside H-module for W division stability.")

    return parser.parse_args()

def main():
    cfg = CONFIG
    args = parse_args()
    set_seed(args.seed if args.seed is not None else cfg["seed"])

    ckpt_path = args.ckpt
    if ckpt_path in (None, "", "auto"):
        auto = find_latest_student_ckpt(Path(cfg["out_dir"]), cfg["auto_find_ckpt_pattern"])
        if auto is None:
            raise FileNotFoundError("No --ckpt provided and no checkpoint found in out_dir.")
        ckpt_path = auto
        print(f"[AUTO] Using latest checkpoint: {ckpt_path}")

    visualize_student_ddim_trajectories(
        cfg=cfg,
        student_ckpt_path=ckpt_path,
        n_noises=int(args.n),
        ddim_steps=int(args.steps),
        eta=float(args.eta),
        save_frames=bool(args.frames),
        make_gif_flag=bool(args.gif),
        out_dir_override=args.out,
        student_stats_path=args.student_stats,
        score_ts=args.score_ts,
        score_grid=int(args.score_grid),
        score_density=float(args.score_density),
        H_ckpt_path=args.H_ckpt,
        H_eps=float(args.H_eps),
    )

if __name__ == "__main__":
    main()
