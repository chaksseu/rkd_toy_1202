#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a trained 2D toy diffusion model (MLPDenoiser), run DDIM sampling with a
user-specified number of steps (or a sweep), and save PNG scatter plots of x0.

- No Weights & Biases dependency
- epsilon prediction
- No normalization (samples are directly plotted in model space)

Example:
python ddim_sample_and_plot.py \
  --ckpt runs/1002_only_diff_loss_teacher8192/ckpt_student_step500000.pt \
  --T 50 --ddim-steps 25 --num-samples 8192 --seed 42 --out runs/1002_only_diff_loss_teacher8192/figs \
  --eta 0.0 --device cuda:1
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, DDIMScheduler


# -------------------- Model -------------------- #
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        t = t.float().unsqueeze(1)  # [B, 1]
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half, device=t.device) * -1.0
        )  # [half]
        angles = t * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [B, 2*half]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MLPDenoiser(nn.Module):
    """ε-predictor for 2D toy; same as training."""

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


# -------------------- Scheduler -------------------- #
def build_ddim_scheduler(num_train_timesteps: int) -> DDIMScheduler:
    """Build DDIM scheduler consistent with linear-beta DDPM training."""
    train_sched = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="linear",
        clip_sample=False,
    )
    train_sched.config.prediction_type = "epsilon"

    sample_sched = DDIMScheduler.from_config(train_sched.config)
    sample_sched.config.clip_sample = False
    sample_sched.config.prediction_type = "epsilon"
    return sample_sched


# -------------------- Utils -------------------- #
def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# -------------------- Sampling -------------------- #
@torch.no_grad()
def sample_x0_ddim(
    model: nn.Module,
    scheduler: DDIMScheduler,
    num_samples: int,
    device: torch.device,
    sample_steps: int,
    dim: int = 2,
    eta: float = 0.0,
):
    """DDIM sampling producing x0 from N(0, I)."""
    scheduler.set_timesteps(sample_steps, device=device)
    x = torch.randn(num_samples, dim, device=device)
    for t in scheduler.timesteps:  # [T-1, ..., 0]
        t_b = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
        x_in = scheduler.scale_model_input(x, t)
        eps = model(x_in, t_b)
        x = scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample
    return x


# -------------------- Main -------------------- #
def main():
    TT = 100
    ddim_step_list = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 400]


    # 공통 인자: 한 번만 파싱
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default=(
            # f"runs/1203_teacher_only_diff_loss_B1024_N65536_T{TT}_no_norm/ckpt_student_step1000000.pt"
            "runs/1204_lr1e4_n32_b1024_T100_ddim_30_50_steps_no_init_rkdW1.0_invW1.0_invinvW1.0_fidW0.0001_sameW0.0_x0_pred_rkd_with_teacher_x0_inv_only_x0_S_no_norm/ckpt_student_step025000.pt"
        ),
        type=str,
        help="Path to model checkpoint (.pt)",
    )
    p.add_argument(
        "--out",
        default=f"vis_ddim/1203_teacher_T{TT}_no_norm",
        type=str,
        help="Output directory for PNGs",
    )
    p.add_argument(
        "--T",
        default=TT,
        type=int,
        help="Training total diffusion steps (0..T-1)",
    )
    p.add_argument(
        "--eta",
        default=0.0,
        type=float,
        help="DDIM eta (0 = deterministic)",
    )
    p.add_argument(
        "--num-samples",
        default=8192,
        type=int,
        help="Number of samples",
    )
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda:5", type=str)
    p.add_argument("--dim", default=2, type=int)
    p.add_argument("--time-dim", default=64, type=int)
    p.add_argument("--hidden", default=256, type=int)
    p.add_argument("--depth", default=8, type=int)
    p.add_argument("--dpi", default=150, type=int)
    p.add_argument(
        "--title",
        default="DDIM Samples (x0)",
        type=str,
    )

    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # scheduler (공통)
    sample_sched = build_ddim_scheduler(args.T)

    # model
    model = MLPDenoiser(
        in_dim=args.dim,
        time_dim=args.time_dim,
        hidden=args.hidden,
        depth=args.depth,
        out_dim=args.dim,
    )

    ckpt_path = Path(args.ckpt)
    sd = torch.load(ckpt_path, map_location="cpu")
    # 다양한 저장 형식 대응
    if isinstance(sd, dict):
        if "model" in sd:
            sd = sd["model"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    # 여러 DDIM step에 대해 반복 샘플링/저장
    for ddim_steps in ddim_step_list:
        x0 = sample_x0_ddim(
            model=model,
            scheduler=sample_sched,
            num_samples=args.num_samples,
            device=device,
            sample_steps=ddim_steps,
            dim=args.dim,
            eta=args.eta,
        )

        x0_np = x0.detach().cpu().numpy()
        x0_plot = x0_np  # normalization 안 쓰므로 그대로 사용

        fig = plt.figure(figsize=(4, 4))
        ax = plt.gca()
        ax.scatter(x0_plot[:, 0], x0_plot[:, 1], s=6, edgecolors="none")
        ax.set_aspect("equal", adjustable="box")
        xlim, ylim = _square_limits_from(x0_plot, pad_ratio=0.05)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(
            f"{args.title}\nsteps={ddim_steps}, eta={args.eta}, N={args.num_samples}"
        )
        fig.tight_layout()
        filename = f"ddim_samples_steps{ddim_steps}.png"
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"[OK] Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
