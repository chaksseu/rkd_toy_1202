#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, random
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DDPMScheduler, DDIMScheduler
from torch.utils.data import Dataset, DataLoader

# ===================== CONFIG ===================== #

TT = 100
WANDB_NAME = f"1223_teacher_only_diff_jit_loss_B1024_N65536_T{TT}_no_norm"

CONFIG = {
    # I/O
    "device": "cuda:6",
    "out_dir": f"runs/{WANDB_NAME}",

    # student ckpt
    "student_init_ckpt": "",
    "resume_student_ckpt": "",

    # diffusion loss 가중치
    "W_DIFF": 1.0,  # ε-pred MSE 가중치

    # 데이터 / 학습
    "batch_size": 1024,
    "epochs_total": 1_000_000,  # 총 스텝 수
    "T": TT,                    # total diffusion steps (timesteps = 0..T-1)
    "seed": 42,

    # 데이터 차원
    "dim": 2,  # 2D toy

    # model sizes (student)
    "student_hidden": 256,
    "student_depth": 8,
    "student_time_dim": 64,

    # optim
    "lr": 1e-4,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,

    # sampling viz
    "vis_interval_epochs": 50_000,
    "ddim_eta": 0.0,

    # wandb
    "use_wandb": True,
    "wandb_project": "RKD-DKDM-AICA-1223",
    "wandb_run_name": WANDB_NAME,
}

CONFIG.update({
    # student 데이터 경로/형식
    "student_data_path": "smile_data_n65536_scale10_rot0_trans_0_0/train.npy",
    "student_data_format": "npy",   # "npy" | "csv"
    "student_dataset_batch_size": 1024,  # 없으면 batch_size 사용
})

# ===================== UTILS ===================== #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_ddim_like(train_sched, device, T: int):
    """
    train_sched.config(베타/타임스텝 등)를 복사해 DDIM 스케줄러 생성.
    timesteps = [T-1, ..., 0] 로 설정.
    """
    ddim = DDIMScheduler.from_config(train_sched.config)
    ddim.config.clip_sample = False
    ddim.config.prediction_type = "sample"
    ddim.set_timesteps(T, device=device)
    return ddim


def _square_limits_from(data: np.ndarray, pad_ratio: float = 0.05):
    """
    데이터의 x/y 범위를 보고, 긴 변 기준(span)으로 여백을 준 뒤
    중앙 정렬된 정사각형 xlim/ylim을 반환.
    """
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
    """ε-predictor for 2D toy; Student 모델로 사용."""
    def __init__(self, in_dim=2, time_dim=64, hidden=128, depth=3, out_dim=2):
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
        return self.out(self.mlp(h))

# ===================== SCHEDULERS ===================== #

def build_schedulers(num_train_timesteps: int):
    train_sched = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="linear",
        clip_sample=False,
    )
    train_sched.config.prediction_type = "sample"

    sample_sched = DDIMScheduler.from_config(train_sched.config)
    sample_sched.config.clip_sample = False
    sample_sched.config.prediction_type = "sample"

    return train_sched, sample_sched

# ===================== Dataset ===================== #

class StudentX0Dataset(Dataset):
    """
    x0 데이터 (2D)만 로드. normalization 전혀 안 함.
    """
    def __init__(self, path: str, fmt: str):
        self.X = self._load(path, fmt)  # (N,2) or (N,D)
        assert self.X.ndim == 2 and self.X.shape[1] >= 2, "Expect (N,2) or (N,D)"
        self.X = self.X[:, :2].astype(np.float32)

    def _load(self, path, fmt):
        p = Path(path)
        if fmt == "npy":
            return np.load(p)
        elif fmt == "csv":
            return np.loadtxt(p, delimiter=",")
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i]  # x0


def build_student_dataloader(cfg):
    bs = int(cfg.get("student_dataset_batch_size", cfg["batch_size"]))
    ds = StudentX0Dataset(cfg["student_data_path"], cfg["student_data_format"])
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True,
                      num_workers=4, pin_memory=True)

# ===================== Training ===================== #

def train_student_uniform_xt(cfg: Dict):
    """
    Student 모델을 diffusion ε-MSE loss로만 학습 (no normalization).
    """
    out_dir = Path(cfg["out_dir"])
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # 스케줄러/모델
    train_sched, _ = build_schedulers(cfg["T"])
    ddim = make_ddim_like(train_sched, device, cfg["T"])

    student = MLPDenoiser(
        in_dim=2,
        time_dim=cfg["student_time_dim"],
        hidden=cfg["student_hidden"],
        depth=cfg["student_depth"],
        out_dim=2,
    ).to(device)

    # init / resume
    if cfg.get("resume_student_ckpt"):
        p = Path(cfg["resume_student_ckpt"])
        if p.exists():
            student.load_state_dict(torch.load(p, map_location=device), strict=True)
            print("[RESUME] Loaded student:", p)
    elif cfg.get("student_init_ckpt"):
        p = Path(cfg["student_init_ckpt"])
        if p.exists():
            student.load_state_dict(torch.load(p, map_location=device), strict=True)
            print("[INIT] Loaded student init:", p)
        else:
            print("[INIT] Student from scratch")

    student.train()
    opt = torch.optim.AdamW(student.parameters(),
                            lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])

    # W&B
    if cfg["use_wandb"]:
        wandb.login()
        wandb.init(project=cfg["wandb_project"],
                   name=cfg["wandb_run_name"],
                   config=cfg)
        wandb.define_metric("step")
        wandb.define_metric("loss/*", step_metric="step")

    # Student 도메인 dataloader
    student_loader = build_student_dataloader(cfg)
    student_iter = iter(student_loader)

    def next_student_batch():
        nonlocal student_iter
        try:
            x0 = next(student_iter)
        except StopIteration:
            student_iter = iter(student_loader)
            x0 = next(student_iter)
        if not torch.is_tensor(x0):
            x0 = torch.as_tensor(x0, dtype=torch.float32)
        return x0.to(device, non_blocking=True)

    T = int(cfg["T"])
    total_steps = int(cfg.get("epochs_total", 50_000))

    for step_i in range(1, total_steps + 1):
        # --- diffusion ε-MSE loss ---
        x0_batch = next_student_batch()       # (B_s, 2)
        B_s = x0_batch.shape[0]

        t_b_s = torch.randint(
            low=0,
            high=T,
            size=(B_s,),           # 실제 배치 크기
            device=device,
            dtype=torch.long,
        )

        eps = torch.randn_like(x0_batch)
        x_t_for_diff = train_sched.add_noise(x0_batch, eps, t_b_s)

        x0_jit_pred = student(x_t_for_diff, t_b_s)
        diff_loss = cfg["W_DIFF"] * F.mse_loss(x0_jit_pred, x0_batch, reduction="mean")

        loss = diff_loss

        opt.zero_grad()
        loss.backward()
        if cfg.get("max_grad_norm", 0) > 0:
            nn.utils.clip_grad_norm_(student.parameters(), cfg["max_grad_norm"])
        opt.step()

        if (step_i % max(1, total_steps // 20) == 0) or (step_i == 1):
            print(f"[step {step_i:06d}] diff={diff_loss.item():.6f}  total={loss.item():.6f}")

        if cfg["use_wandb"]:
            wandb.log({
                "step": step_i,
                "loss/total": float(loss),
                "loss/diff": float(diff_loss),
                "lr": opt.param_groups[0]["lr"],
            }, step=step_i)

        # --- 시각화 & ckpt 저장 ---
        if (step_i % cfg["vis_interval_epochs"] == 0) or (step_i == total_steps):

            @torch.no_grad()
            def sample_x0_ddim(model, sample_scheduler, num_samples, device, sample_steps, dim=2, eta=0.0):
                sample_scheduler.set_timesteps(sample_steps, device=device)
                x = torch.randn(num_samples, dim, device=device)
                for t in sample_scheduler.timesteps:  # [T-1, ..., 0]
                    t_b = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
                    x_in = sample_scheduler.scale_model_input(x, t)
                    eps = model(x_in, t_b)
                    x = sample_scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample
                return x

            student.eval()
            B_plot = 8192
            x0_s = sample_x0_ddim(
                model=student,
                sample_scheduler=ddim,
                num_samples=B_plot,
                device=device,
                sample_steps=40,#int(cfg["T"]),
                dim=int(cfg.get("dim", 2)),
                eta=float(cfg.get("ddim_eta", 0.0)),
            )

            x0_s_plot = x0_s.detach().cpu().numpy()

            figs_dir = Path(cfg["out_dir"]) / "figs"
            figs_dir.mkdir(parents=True, exist_ok=True)

            png_path = figs_dir / f"samples_step{step_i:06d}.png"

            plt.figure(figsize=(4, 4))
            plt.scatter(x0_s_plot[:, 0], x0_s_plot[:, 1], s=6, edgecolors="none")
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
            xlim, ylim = _square_limits_from(x0_s_plot, pad_ratio=0.05)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            plt.title(f"Student samples (x0) @ step {step_i}")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close()

            if cfg["use_wandb"]:
                wandb.log({"img/student_samples": wandb.Image(str(png_path))}, step=step_i)

            if (step_i % cfg["vis_interval_epochs"] == 0) or (step_i == total_steps):
                ckpt_path = out_dir / f"ckpt_student_step{step_i:06d}.pt"
                torch.save(student.state_dict(), ckpt_path)
                print("[CKPT]", ckpt_path)

            student.train()

    print("\n[DONE] Out dir:", out_dir.resolve())
    if cfg["use_wandb"]:
        wandb.finish()

# ===================== MAIN ===================== #

def main(cfg: Dict):
    set_seed(cfg["seed"])
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    train_student_uniform_xt(cfg)


if __name__ == "__main__":
    main(CONFIG)
