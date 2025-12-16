#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Student DDIM Trajectory Visualizer
- STUDENT(MLP epsilon-predictor) 로드
- DDIM으로 x_T -> x_0 궤적 수집 및 저장 (NPY / overlay / frames / GIF)
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
CUDA_NUM = 5
TT = 100
DDIM_STEPS = 40

# runs/1205_lr1e4_n32_b1024_T100_ddim_30_50_steps_no_init_rkdW0.08_invW0.1_invinvW1.0_fidW0.01_sameW0.001_x0_pred_rkd_with_teacher_x0_inv_only_x0_S_no_norm/ckpt_student_step090000.pt
# runs/1205_lr1e4_n32_b1024_T100_ddim_30_50_steps_no_init_rkdW0.08_invW0.1_invinvW1.0_fidW0.01_sameW0.0001_x0_pred_rkd_with_teacher_x0_inv_only_x0_S_no_norm/ckpt_student_step235000.pt

# runs/1212_lr1e4_n32_b1024_T100_ddim_30_50_steps_no_init_rkdW0.1_invW0.1_invinvW1.0_fidW0.1_sameW0.01_x0_pred_rkd_with_teacher_x0_inv_only_x0_no_norm_jit/ckpt_student_step130000.pt

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Student DDIM trajectories.")
    parser.add_argument("--ckpt", type=str, default=f"runs/1212_lr1e4_n32_b1024_T100_ddim_30_50_steps_no_init_rkdW0.1_invW0.1_invinvW1.0_fidW0.1_sameW0.01_x0_pred_rkd_with_teacher_x0_inv_only_x0_no_norm_jit/ckpt_student_step130000.pt", help="Path to student checkpoint .pt.")
    parser.add_argument("--n", type=int, default=32, help="Number of pure noise samples.")
    parser.add_argument("--steps", type=int, default=DDIM_STEPS, help="DDIM sampling steps.")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta.")
    parser.add_argument("--frames", type=bool, default=True, help="Save per-timestep frames.")
    parser.add_argument("--gif", type=bool, default=True, help="Make GIFs.")
    parser.add_argument("--out", type=str, default=f"vis_traj_1215_rkdW0.1_invW0.1_invinvW1.0_fidW0.1_sameW0.01_no_norm", help="Output directory root.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

CONFIG = {
    "device": f"cuda:{CUDA_NUM}",
    "out_dir": f"runs/",
    "T": TT,
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
    data = np.asarray(data)
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

# ===================== MODEL ===================== #
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        t = t.float().unsqueeze(1)
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half, device=t.device) * -1.0
        )
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
            layers += [
                nn.Linear(in_dim + time_dim if i == 0 else hidden, hidden),
                nn.SiLU(),
            ]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x, t):
        te = self.t_embed(t)
        h = torch.cat([x, te], dim=-1)
        return self.out(self.mlp(h))  # ε(x,t)

# ===================== SCHEDULERS ===================== #
def build_schedulers(num_train_timesteps: int):
    train_sched = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        clip_sample=False,
        beta_schedule="linear",
    )
    train_sched.config.prediction_type = "epsilon"
    sample_sched = DDIMScheduler.from_config(train_sched.config)
    sample_sched.config.clip_sample = False
    sample_sched.config.prediction_type = "epsilon"
    return train_sched, sample_sched

# ===================== SAMPLING (TRAJ) ===================== #
@torch.no_grad()
def collect_student_xt_seq_ddim(
    student: nn.Module,
    sample_sched: DDIMScheduler,
    z: torch.Tensor,
    device: torch.device,
    sample_steps: int = 100,
    eta: float = 0.0,
):
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
            xs.append(x.detach().cpu().numpy())
            ts.append(t_int)
            tb = torch.full((B,), t_int, device=device, dtype=torch.long)
            xin = sched.scale_model_input(x, t)
            eps = student(xin, tb)
            x = sched.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample
    finally:
        if was_train:
            student.train()
    return np.stack(xs, 0), np.asarray(ts, dtype=int)

# ===================== PLOTTING (TRAJ, FRAMES, GIF) ===================== #
def plot_trajectories_overlay(
    seq_xy: np.ndarray,
    out_path: Path,
    max_lines: int = 512,
    line_alpha: float = 0.85,
    dot_size: int = 14,
    pad_ratio: float = 0.06,
    title: str = None,
):
    K, B, _ = seq_xy.shape
    cols = colors_from_noise_ids(np.arange(B), alpha=line_alpha)
    XY_all = seq_xy.reshape(-1, 2)
    xlim, ylim = _square_limits_from(XY_all, pad_ratio=pad_ratio)
    pick = np.linspace(0, B - 1, num=min(B, max_lines), dtype=int)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for b in pick:
        xy = seq_xy[:, b, :]
        ax.plot(xy[:, 0], xy[:, 1], lw=1.2, alpha=line_alpha, c=cols[b])
        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            s=dot_size,
            c=[cols[b]],
            marker="o",
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
        )
        ax.scatter(
            xy[-1, 0],
            xy[-1, 1],
            s=dot_size,
            c=[cols[b]],
            marker="X",
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
        )
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

def save_per_timestep_frames(
    seq_xy: np.ndarray, ts: np.ndarray, out_dir: Path, dot_size: int = 10
):
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
        ax.set_title(f"Student $x_t$ (t={int(ts[k])})")
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
    imgs = [imageio.imread(str(p)) for p in files_sorted]
    imageio.mimsave(str(out_path), imgs, duration=1.0 / max(fps, 1))
    print(f"[GIF] saved (large t -> 0) -> {out_path}")

# ===================== MAIN LOGIC ===================== #
@torch.no_grad()
def visualize_student_ddim_trajectories(
    cfg: dict,
    student_ckpt_path: str,
    n_noises: int = 128,
    ddim_steps: int = 100,
    eta: float = 0.0,
    save_frames: bool = True,
    make_gif_flag: bool = True,
    out_dir_override: str = None,
):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    out_root = Path(out_dir_override) if out_dir_override else Path(cfg["out_dir"])
    out_traj = ensure_dir(out_root / "traj")

    # schedulers + student
    _, sample_sched = build_schedulers(cfg["T"])
    student = MLPDenoiser(
        in_dim=cfg["dim"],
        time_dim=cfg["student_time_dim"],
        hidden=cfg["student_hidden"],
        depth=cfg["student_depth"],
        out_dim=cfg["dim"],
    ).to(device)
    student.load_state_dict(
        torch.load(student_ckpt_path, map_location=device), strict=True
    )
    student.eval()
    print(f"[LOAD] Student checkpoint -> {student_ckpt_path}")

    # noise & collect (no normalization/denormalization)
    z = torch.randn(n_noises, cfg["dim"], device=device)
    seq_xy, ts = collect_student_xt_seq_ddim(
        student, sample_sched, z, device, sample_steps=int(ddim_steps), eta=float(eta)
    )
    K, B, _ = seq_xy.shape

    # Save & overlay
    np.save(out_traj / "student_traj_xy.npy", seq_xy)
    np.save(out_traj / "student_traj_ts.npy", ts)
    plot_trajectories_overlay(
        seq_xy,
        out_traj / f"student_traj_steps{K}_N{B}.png",
        max_lines=512,
        title=f"Student DDIM steps={K} N={B} eta={eta}",
    )
    if save_frames:
        frames_dir = ensure_dir(out_traj / "frames")
        save_per_timestep_frames(seq_xy, ts, frames_dir)
        if make_gif_flag:
            maybe_make_gif(
                frames_dir, out_traj / f"student_traj_steps{K}_N{B}.gif", fps=10
            )

    print(f"[DONE] out dir: {out_traj.resolve()}")

# ===================== ARGS / ENTRY ===================== #
            


def main():
    cfg = CONFIG
    args = parse_args()
    set_seed(args.seed if args.seed is not None else cfg["seed"])

    ckpt_path = args.ckpt
    if ckpt_path in (None, "", "auto"):
        auto = find_latest_student_ckpt(
            Path(cfg["out_dir"]), cfg["auto_find_ckpt_pattern"]
        )
        if auto is None:
            raise FileNotFoundError(
                "No --ckpt provided and no checkpoint found in out_dir."
            )
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
    )

if __name__ == "__main__":
    main()
