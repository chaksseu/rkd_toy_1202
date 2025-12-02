#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA fine-tuning of 2D diffusion teacher on student domain.

- Base: pretrained teacher MLPDenoiser (epsilon predictor)
- Base weights: frozen
- LoRA: low-rank adapters on Linear layers
- Training: student x0 데이터만 이용해서 diffusion ε-MSE loss 로 학습
- Logging: loss / 샘플 scatter plot을 wandb에 로깅
"""

import json, math, random
from pathlib import Path
from typing import Dict
import os

import numpy as np
import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DDPMScheduler, DDIMScheduler
from torch.utils.data import Dataset, DataLoader

# ===================== CONFIG ===================== #

CUDA_NUM = 0
N = 32
BATCH_SIZE = N

WANDB_NAME_LORA = f"1124_lora_H_student_n{N}_lr1e4_b{BATCH_SIZE}_T1000_rank2_alpha2"

CONFIG_LORA = {
    # device / I/O
    "device": f"cuda:{CUDA_NUM}",
    "out_dir": f"runs/{WANDB_NAME_LORA}",

    # teacher & student data
    "teacher_ckpt": "ckpt_teacher_T1000_step370000_1021.pt",   # 기존 teacher ckpt
    "student_data_stats": f"smile_data_n8192_scale10_rot0_trans_0_0_H_32_-13_100_55_8_200_0.05_0.005_1.2_n{N}/normalization_stats.json",
    "student_data_path": f"smile_data_n8192_scale10_rot0_trans_0_0_H_32_-13_100_55_8_200_0.05_0.005_1.2_n{N}/train.npy",
    "student_data_format": "npy",  # "npy" | "csv"

    # diffusion / model
    "dim": 2,
    "T": 1000,
    "teacher_hidden": 256,
    "teacher_depth": 8,
    "teacher_time_dim": 64,

    # train
    "batch_size": BATCH_SIZE,
    "epochs_total": 500000,        # 총 스텝 수
    "lr": 1e-4,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,

    # LoRA 설정
    "lora_rank": 2,
    "lora_alpha": 2.0,
    "lora_dropout": 0.0,

    # sampling & viz
    "ddim_eta": 0.0,
    "ddim_sample_steps": 100,
    "n_vis": 8192,
    "vis_interval_steps": 10000,

    # 기타
    "seed": 42,
    "use_wandb": True,
    "wandb_project": "RKD-DKDM-1125-LORA",
    "wandb_run_name": WANDB_NAME_LORA,
}

# ===================== UTILS ===================== #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_norm_stats(json_path: str):
    """JSON 파일에서 mean / std를 불러와 numpy array로 반환"""
    json_path = Path(json_path)
    with json_path.open("r") as f:
        d = json.load(f)  # {"mean": [...], "std": [...]}
    mu = np.array(d["mean"], dtype=np.float32)
    sigma = np.array(d["std"], dtype=np.float32)
    return mu, sigma


def normalize_np(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (arr - mu) / sigma


def denormalize_np(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return arr * sigma + mu


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

    xmin -= pad; xmax += pad
    ymin -= pad; ymax += pad

    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    half = span / 2.0

    return (xmid - half, xmid + half), (ymid - half, ymid + half)


# ===================== Dataset ===================== #

class StudentX0Dataset(Dataset):
    """
    2D student x0 데이터셋 (정규화 포함)
    """
    def __init__(self, path: str, fmt: str, mu: np.ndarray, sigma: np.ndarray, dim: int = 2):
        self.dim = dim
        self.X = self._load(path, fmt)  # (N, D)
        assert self.X.ndim == 2 and self.X.shape[1] >= dim, "Expect (N,D>=dim)"
        self.X = self.X[:, :dim].astype(np.float32)
        self.X = normalize_np(self.X, mu, sigma).astype(np.float32)

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
        return self.X[i]


# ===================== MODEL (Teacher MLP + LoRA) ===================== #

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        t = t.float().unsqueeze(1)  # [B,1]
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half, device=t.device) * -1.0
        )
        angles = t * freqs.unsqueeze(0)  # [B,half]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [B,2*half]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MLPDenoiser(nn.Module):
    """ε-predictor for 2D toy diffusion."""
    def __init__(self, in_dim=2, time_dim=64, hidden=128, depth=3, out_dim=2):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(time_dim)
        layers = []
        for i in range(depth):
            if i == 0:
                inp_dim = in_dim + time_dim
            else:
                inp_dim = hidden
            layers += [nn.Linear(inp_dim, hidden), nn.SiLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_embed(t)                 # [B, time_dim]
        h = torch.cat([x, te], dim=-1)       # [B, in_dim + time_dim]
        h = self.mlp(h)                      # [B, hidden]
        return self.out(h)                   # [B, out_dim] (epsilon)


class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear:
        y = W x + (alpha/r) * B(Ax)
    base Linear의 weight/bias는 freeze, A/B만 학습.
    """
    def __init__(self, base_layer: nn.Linear, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # freeze base weights
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        if r > 0:
            self.lora_A = nn.Linear(self.base.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.base.out_features, bias=False)
            # init (원래 LoRA 스타일: B는 0으로 시작)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            self.scaling = alpha / r
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0:
            out = out + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))
        return out


def inject_lora(model: nn.Module, r: int, alpha: float, dropout: float) -> nn.Module:
    """
    model 안의 모든 nn.Linear를 LoRALinear로 교체.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        else:
            inject_lora(module, r, alpha, dropout)
    return model


# ===================== SCHEDULERS ===================== #

def build_schedulers(num_train_timesteps: int):
    train_sched = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        clip_sample=False,
    )
    train_sched.config.prediction_type = "epsilon"

    sample_sched = DDIMScheduler.from_config(train_sched.config)
    sample_sched.config.clip_sample = False
    sample_sched.config.prediction_type = "epsilon"

    return train_sched, sample_sched


@torch.no_grad()
def sample_x0_ddim(
    model: nn.Module,
    sample_scheduler: DDIMScheduler,
    num_samples: int,
    dim: int,
    device: torch.device,
    sample_steps: int,
    eta: float = 0.0,
) -> torch.Tensor:
    """
    DDIM 샘플링으로 x_T ~ N(0, I)에서 x_0 샘플 생성
    """
    # 독립된 scheduler 인스턴스 사용
    scheduler = DDIMScheduler.from_config(sample_scheduler.config)
    scheduler.set_timesteps(sample_steps, device=device)

    x = torch.randn(num_samples, dim, device=device)
    model.eval()
    for t in scheduler.timesteps:  # [T'-1, ..., 0]
        t_b = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
        x_in = scheduler.scale_model_input(x, t)
        eps = model(x_in, t_b)
        x = scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample
    model.train()
    return x


# ===================== TRAINING (LoRA) ===================== #

def train_lora_student(cfg: Dict):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])

    # schedulers
    train_sched, sample_sched = build_schedulers(cfg["T"])

    # student 데이터 통계 및 dataloader
    mu_student, sigma_student = load_norm_stats(cfg["student_data_stats"])
    ds_student = StudentX0Dataset(
        cfg["student_data_path"],
        cfg["student_data_format"],
        mu_student,
        sigma_student,
        dim=cfg["dim"],
    )
    loader = DataLoader(
        ds_student,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    data_iter = iter(loader)

    def next_student_batch():
        nonlocal data_iter
        try:
            x0 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0 = next(data_iter)
        if not torch.is_tensor(x0):
            x0 = torch.as_tensor(x0, dtype=torch.float32)
        return x0.to(device, non_blocking=True)

    # ----- base teacher 불러오기 + LoRA 주입 -----
    base_teacher = MLPDenoiser(
        in_dim=cfg["dim"],
        time_dim=cfg["teacher_time_dim"],
        hidden=cfg["teacher_hidden"],
        depth=cfg["teacher_depth"],
        out_dim=cfg["dim"],
    ).to(device)
    base_teacher.load_state_dict(
        torch.load(cfg["teacher_ckpt"], map_location=device),
        strict=True,
    )

    # base teacher를 LoRA 버전으로 변환
    lora_model = inject_lora(
        base_teacher,
        r=int(cfg["lora_rank"]),
        alpha=float(cfg["lora_alpha"]),
        dropout=float(cfg["lora_dropout"]),
    ).to(device)

    # base weight는 완전히 freeze (LoRA weight만 학습)
    for name, p in lora_model.named_parameters():
        if "lora_" not in name:
            p.requires_grad = False

    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    print(f"[LoRA] Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    opt = torch.optim.AdamW(
        trainable_params,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # ----- wandb -----
    if cfg["use_wandb"]:
        wandb.login()
        wandb.init(
            project=cfg["wandb_project"],
            name=cfg["wandb_run_name"],
            config=cfg,
        )
        wandb.define_metric("step")
        wandb.define_metric("loss/*", step_metric="step")

    total_steps = int(cfg["epochs_total"])
    T = int(cfg["T"])

    for step in range(1, total_steps + 1):
        lora_model.train()

        x0 = next_student_batch()            # [B, 2], normalized
        B = x0.size(0)

        # t ~ Uniform{0..T-1}, per sample
        t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)

        # x_t = sqrt(alpha_t) x_0 + sqrt(1-alpha_t) noise
        x_t = train_sched.add_noise(x0, noise=noise, timesteps=t)

        # ε(x_t, t) 예측
        eps_pred = lora_model(x_t, t)

        # diffusion loss: MSE(ε_pred, ε)
        loss = F.mse_loss(eps_pred, noise)

        opt.zero_grad()
        loss.backward()
        if cfg.get("max_grad_norm", 0.0) > 0:
            nn.utils.clip_grad_norm_(trainable_params, cfg["max_grad_norm"])
        opt.step()

        # 로그 & 출력
        if step == 1 or step % 50 == 0:
            print(f"[LoRA step {step:06d}] loss/diffusion = {loss.item():.6f}")

        if cfg["use_wandb"]:
            wandb.log(
                {
                    "step": step,
                    "loss/diffusion": float(loss.item()),
                    "lr": opt.param_groups[0]["lr"],
                },
                step=step,
            )

        # ----- 샘플링 + 시각화 + ckpt -----
        if (step % cfg["vis_interval_steps"] == 0) or (step == total_steps):
            with torch.no_grad():
                num_vis = min(int(cfg["n_vis"]), 8192)
                x0_samples = sample_x0_ddim(
                    model=lora_model,
                    sample_scheduler=sample_sched,
                    num_samples=num_vis,
                    dim=cfg["dim"],
                    device=device,
                    sample_steps=int(cfg["ddim_sample_steps"]),
                    eta=float(cfg["ddim_eta"]),
                )
                x0_samples_np = x0_samples.detach().cpu().numpy()
                x0_samples_np = denormalize_np(x0_samples_np, mu_student, sigma_student)

                plt.figure(figsize=(4, 4))
                plt.scatter(
                    x0_samples_np[:, 0],
                    x0_samples_np[:, 1],
                    s=6,
                    edgecolors="none",
                )
                ax = plt.gca()
                ax.set_aspect("equal", adjustable="box")
                xlim, ylim = _square_limits_from(x0_samples_np, pad_ratio=0.05)
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                plt.title(f"LoRA student samples (x0) @ step {step}")
                plt.tight_layout()

                png_path = figs_dir / f"lora_samples_step{step:06d}.png"
                plt.savefig(png_path, dpi=150, bbox_inches="tight")
                plt.close()

                if cfg["use_wandb"]:
                    wandb.log(
                        {"img/lora_student_samples": wandb.Image(str(png_path))},
                        step=step,
                    )

            # ckpt 저장
            ckpt_path = out_dir / f"ckpt_lora_step{step:06d}.pt"
            torch.save(lora_model.state_dict(), ckpt_path)
            print("[LoRA CKPT]", ckpt_path)

    print("\n[LoRA DONE] Out dir:", out_dir.resolve())
    if cfg["use_wandb"]:
        wandb.finish()


# ===================== MAIN ===================== #

def main(cfg: Dict):
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    train_lora_student(cfg)


if __name__ == "__main__":
    main(CONFIG_LORA)
