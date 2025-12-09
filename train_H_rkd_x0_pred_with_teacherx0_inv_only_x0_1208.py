#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, math, random
from pathlib import Path
from typing import Dict, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from diffusers import DDPMScheduler, DDIMScheduler, DDIMInverseScheduler
from torch.utils.data import Dataset, DataLoader
import itertools
# ===================== CONFIG ===================== #

W_RKD = 0.08
W_INV = 0.1
W_INVINV = 1.0
W_FID = 0.00000001
W_SAME = 0.00000001 #000001

# W_RKD = 0.08
# W_INV = 0.0
# W_INVINV = 0.0
# W_FID = 0.0
# W_SAME = 0.0

CUDA_NUM = 0
BATCH_SIZE = 1024

WANDB_NAME=f"1209_lr1e4_n32_H_b{BATCH_SIZE}_T100_ddim_30_50_steps_no_init_rkdW{W_RKD}_invW{W_INV}_invinvW{W_INVINV}_fidW{W_FID}_sameW{W_SAME}_x0_pred_rkd_with_teacher_x0_inv_only_x0"


CONFIG = {
    # I/O
    "device": f"cuda:{CUDA_NUM}",
    "out_dir": f"runs/{WANDB_NAME}",
    # teacher / student
    "teacher_ckpt": f"ckpt_teacher_B1024_N65536_T100_step1000000.pt", 
    "student_init_ckpt": "",                     
    # "student_init_ckpt": "runs/1025_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.0_invW0.0_invinv_W1.0_diffW0.1/ckpt_student_step200000.pt",                     
    "resume_student_ckpt": f"",        
    "teacher_data_stats": "smile_data_n65536_scale10_rot0_trans_0_0/normalization_stats.json",
    "student_data_stats": "smile_data_n8192_scale10_rot0_trans_0_0_H_32/normalization_stats.json",

    # diffusion loss 가중치
    "W_RKD": W_RKD,
    "W_INV": W_INV,                               # ε-pred MSE 가중치
    "W_INVINV": W_INVINV,                               # ε-pred MSE 가중치
    "W_FID": W_FID,                               # ε-pred MSE 가중치
    "W_SAME": W_SAME,
    "same_mode": "mean",   # 또는 "last" 로 바꿔가며 실험


    "rkd_ddim_steps_to_t": 40,   # t_sel까지 최대 몇 번의 DDIM 전이만 사용할지

    "batch_size": BATCH_SIZE,
    "num_noises": 8192, 
    "epochs_total": 5000000,          # 총 스텝 수 (기존 epochs_per_stage 대신 사용)

    "noise_pool_file": None,        # None이면 out_dir/data/noises_pool.npy 로 저장
    "regen_noise_pool": False,      # True면 항상 새로 만듦
    
    # schedule / time
    "T": 100,                 # total diffusion steps (timesteps = 0..T-1)

    "seed": 42,
    # RKD weights
    "W_NORM": 0.0,
    "use_mu_normalization": True,
    # noises / data dims
    "dim": 2,                 # 2D toy

    # model sizes
    "teacher_hidden": 256, "teacher_depth": 8, "teacher_time_dim": 64,
    "student_hidden": 256, "student_depth": 8, "student_time_dim": 64,
    # optim
    "lr": 1e-4, "weight_decay": 0.0, "max_grad_norm": 1.0,
    # sampling viz
    "vis_interval_epochs": 1000,
    "n_vis": 8192,       # 경로를 수집/표시할 noise 개수
    "ddim_eta": 0.0,
    # wandb
    "use_wandb": True,
    "wandb_project": "RKD-DKDM-AICA-1209-H",
    "wandb_run_name": WANDB_NAME,

    "use_learnable_H": True,
    "H_init": [1,0,0,  0,1,0,  0,0,1],     # 초기값(I)
    "H_eps": 1e-8,                        # w 분모 안정화
    "resume_H_ckpt": "",                  # (옵션) H만 재개 로드 경로
}

CONFIG.update({
    # student 데이터 경로/형식
    "student_data_path": "smile_data_n8192_scale10_rot0_trans_0_0_H_32/tgt.npy",
    # "student_data_path": "smile_data_n8192_scale10_rot0_trans_0_0/train.npy",   # 혹은 .csv
    "student_data_format": "npy",                # "npy" | "csv"
    "student_dataset_batch_size": 32,          # 없으면 batch_size 사용
    # === 새로 추가: teacher/student pair 데이터 (src.npy, tgt.npy 가 있는 폴더) ===
    "pair_data_dir": "smile_data_n8192_scale10_rot0_trans_0_0_H_32",   # 예: "smile_pairs_n32_H" 같은 out_dir
    "pair_batch_size": 32,                      # 보통 32 (pair 전체를 한 번에)
})


# ===================== UTILS ===================== #
import re

from pathlib import Path


class TeacherStudentPairDataset(Dataset):
    """
    pair_data_dir 안에 있는 src.npy(teacher x0), tgt.npy(student x0)를 불러와
    각 도메인별 normalization을 적용한 뒤 (x0_T_norm, x0_S_norm) 을 반환.
    """
    def __init__(self, pair_dir: str,
                 mu_teacher: np.ndarray, sigma_teacher: np.ndarray,
                 mu_student: np.ndarray, sigma_student: np.ndarray):
        pair_dir = Path(pair_dir)
        src = np.load(pair_dir / "src.npy").astype(np.float32)   # (N,2 or more)
        tgt = np.load(pair_dir / "tgt.npy").astype(np.float32)
        assert src.shape == tgt.shape and src.ndim == 2 and src.shape[1] >= 2, \
            "src.npy / tgt.npy must both be (N,2) or (N,D) with same shape."

        src = src[:, :2]
        tgt = tgt[:, :2]

        # 도메인별 normalization
        self.x_teacher = normalize_np(src, mu_teacher, sigma_teacher).astype(np.float32)
        self.x_student = normalize_np(tgt, mu_student, sigma_student).astype(np.float32)

    def __len__(self):
        return self.x_teacher.shape[0]

    def __getitem__(self, idx):
        # 둘 다 normalized 된 x0
        return self.x_teacher[idx], self.x_student[idx]


def _mean_and_cov(X: torch.Tensor, eps: float = 1e-6):
    # X: [N, D] -> (mu[D], C[D,D])
    X = X.to(torch.float64)
    N, D = X.shape
    if N == 0:
        mu = torch.zeros(D, dtype=X.dtype, device=X.device)
        C  = torch.eye(D, dtype=X.dtype, device=X.device)
        return mu, C
    mu = X.mean(dim=0, keepdim=True)
    xc = X - mu
    denom = (N - 1) if N > 1 else 1
    C = (xc.t() @ xc) / denom
    I = torch.eye(D, dtype=X.dtype, device=X.device)
    C = 0.5 * (C + C.t()) + eps * I   # 대칭화 + 지터
    return mu.squeeze(0), C

def _sqrtm_psd(A: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    # A: 대칭 PSD 가정
    A = 0.5 * (A + A.t())
    evals, vecs = torch.linalg.eigh(A)
    evals = (evals + eps).clamp_min(0)
    return (vecs * evals.sqrt().unsqueeze(0)) @ vecs.t()

def fid_gaussian_torch(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert X.dim() == 2 and Y.dim() == 2 and X.size(1) == Y.size(1)
    out_dtype = X.dtype

    mx, Cx = _mean_and_cov(X, eps)
    my, Cy = _mean_and_cov(Y, eps)

    mean_term = ((mx - my) ** 2).sum()

    Cy_sqrt = _sqrtm_psd(Cy, eps=eps)
    B = Cy_sqrt @ Cx @ Cy_sqrt
    B_sqrt = _sqrtm_psd(B, eps=eps)

    trace_term = torch.trace(Cx + Cy - 2.0 * B_sqrt)
    return (mean_term + trace_term).clamp_min(0.0).to(out_dtype)





def plot_triplet_scatter_colored(
    left_xy: np.ndarray,      # Teacher (B,2)
    mid_xy: np.ndarray,       # Student raw (B,2)
    right_xy: np.ndarray,     # Student + H(t) (B,2)
    noise_ids: np.ndarray,
    N_total: int,
    out_path: Path,
    titles: Tuple[str, str, str] = (r"Teacher $x_t$", r"Student $x_t$ (raw)", r"Student $x_t$ + H(t)"),
    dot_size: int = 6,
    sync_axes: bool = True,
    pad_ratio: float = 0.05,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    A = np.asarray(left_xy); B = np.asarray(mid_xy); C = np.asarray(right_xy)
    assert A.shape == B.shape == C.shape and A.shape[1] == 2, "Expect three (B,2) arrays."

    colors = colors_from_noise_ids(np.asarray(noise_ids), N_total)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharex=sync_axes, sharey=sync_axes)

    if sync_axes:
        all_pts = np.vstack([A, B, C])
        xlim_all, ylim_all = _square_limits_from(all_pts, pad_ratio=pad_ratio)

    for ax, data, title in zip(axes, (A, B, C), titles):
        ax.scatter(data[:, 0], data[:, 1], s=dot_size, c=colors, edgecolors="none")
        ax.set_aspect("equal", adjustable="box"); ax.set_title(title)
        if sync_axes:
            ax.set_xlim(*xlim_all); ax.set_ylim(*ylim_all)
        else:
            xlim, ylim = _square_limits_from(data, pad_ratio=pad_ratio)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_and_log_xt_triplets_for_all_t(
    teacher_seq: np.ndarray,   # [K,B,2]
    student_seq: np.ndarray,   # [K,B,2]
    studentH_seq: np.ndarray,  # [K,B,2]
    ts: np.ndarray,            # [K]
    noise_ids: np.ndarray,     # [B]
    out_dir: Path,
    step_i: int,
    use_wandb: bool = False,
    subdir_name: str = "xt_triplets",
    dot_size: int = 6,
    sync_axes: bool = True,
):
    base = out_dir / "figs" / f"{subdir_name}_step{step_i:06d}"
    base.mkdir(parents=True, exist_ok=True)

    logged = {}
    for k, t in enumerate(ts):
        path = base / f"t{int(t):03d}.png"
        plot_triplet_scatter_colored(
            left_xy  = teacher_seq[k],
            mid_xy   = student_seq[k],
            right_xy = studentH_seq[k],
            noise_ids=noise_ids,
            N_total=len(noise_ids),
            out_path=path,
            titles=(fr"Teacher $x_t$ (t={int(t)})", fr"Student $x_t$ (raw)", fr"Student $x_t$ + H(t)"),
            dot_size=dot_size,
            sync_axes=sync_axes,
        )
        if use_wandb:
            logged[f"img/{subdir_name}/t{int(t):03d}"] = wandb.Image(str(path))

    if use_wandb and logged:
        wandb.log(logged, step=step_i)

    return base

@torch.no_grad()
def apply_H_to_seq_per_t(seq_S: np.ndarray, ts: np.ndarray, H_module: nn.Module, device: torch.device) -> np.ndarray:
    """
    seq_S: [K,B,2] (numpy, normalized)
    ts   : [K]     (int timesteps)
    return: seq_S_H [K,B,2] with H(t) applied per timestep
    """
    K, B, _ = seq_S.shape
    outs = []
    for k in range(K):
        t_k = int(ts[k])
        xk  = torch.from_numpy(seq_S[k]).to(device=device, dtype=torch.float32)   # (B,2)
        xk_H = xk #, _ =  H_module(xk, t_k)                                              # (B,2)
        if k == K - 1:  # 마지막 스텝만 H 적용
            xk_H, _ = H_module(xk)
        outs.append(xk_H.detach().cpu().numpy())
    return np.stack(outs, axis=0)


def ensure_noise_pool(cfg) -> Path:
    """노이즈 풀(.npy) 보장 생성 후 경로 반환"""
    out_dir = Path(cfg["out_dir"])
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = Path(cfg["noise_pool_file"] or (data_dir / "noises_pool.npy"))
    if cfg.get("regen_noise_pool", False) or (not path.exists()):
        N, D = cfg["num_noises"], cfg["dim"]
        z = np.random.randn(N, D).astype("float32")
        np.save(path, z)
        print(f"[NOISE] generated pool: {path}  shape={(N,D)}")
    else:
        print(f"[NOISE] using existing pool: {path}")
    return path


def make_ddim_like(train_sched, device, T: int):
    """
    train_sched.config(베타/타임스텝 등)를 복사해 DDIM 스케줄러 생성.
    timesteps = [T-1, ..., 0] 로 설정.
    """
    ddim = DDIMScheduler.from_config(train_sched.config)
    ddim.config.clip_sample = False
    ddim.config.prediction_type = "epsilon"
    ddim.set_timesteps(T, device=device)  # [T-1, ..., 0]
    return ddim

def sample_noise_batch(noise_pool: np.ndarray, B: int, device: torch.device) -> torch.Tensor:
    """노이즈 풀에서 인덱스 랜덤 샘플로 배치 구성"""
    N = noise_pool.shape[0]
    idx = np.random.randint(0, N, size=B)
    z = torch.from_numpy(noise_pool[idx]).to(device=device, dtype=torch.float32)
    return z



def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_norm_stats(json_path: str):
    json_path = Path(json_path)
    with json_path.open("r") as f:
        d = json.load(f)           # {"mean": [...], "std": [...]}
    mu = np.array(d["mean"],  dtype=np.float32)
    sigma = np.array(d["std"], dtype=np.float32)
    return mu, sigma

def load_norm_stats_torch(json_path: str, device: str = 'cpu'):
    json_path = Path(json_path)
    with json_path.open("r") as f:
        d = json.load(f)  # {"mean": [...], "std": [...]}
    
    mu_np = np.array(d["mean"], dtype=np.float32)
    sigma_np = np.array(d["std"], dtype=np.float32)

    mu_tensor = torch.from_numpy(mu_np)
    sigma_tensor = torch.from_numpy(sigma_np)

    mu_tensor = mu_tensor.to(device)
    sigma_tensor = sigma_tensor.to(device)
    
    return mu_tensor, sigma_tensor



def _square_limits_from(data: np.ndarray, pad_ratio: float = 0.05):
    """
    데이터의 x/y 범위를 보고, 긴 변 기준(span)으로 여백을 준 뒤
    중앙 정렬된 정사각형 xlim/ylim을 반환.
    """
    data = np.asarray(data)
    xmin, xmax = float(data[:, 0].min()), float(data[:, 0].max())
    ymin, ymax = float(data[:, 1].min()), float(data[:, 1].max())
    dx, dy = xmax - xmin, ymax - ymin

    # 긴 변 + 최소 폭 보호
    base = max(dx, dy, 1e-3)
    pad  = pad_ratio * base

    # 여백 적용
    xmin -= pad; xmax += pad
    ymin -= pad; ymax += pad

    # 중앙 정렬된 정사각형으로 확장
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    half = span / 2.0

    return (xmid - half, xmid + half), (ymid - half, ymid + half)


def colors_from_noise_ids(ids: np.ndarray, N_total: int, alpha: float = 0.85):
    """
    같은 'global noise id'에는 항상 같은 색이 나오도록 결정적 매핑.
    - hue = (id * φ) mod 1.0  (φ ≈ 0.618…)
    - colormap: hsv (연속)
    """
    ids = np.asarray(ids, dtype=np.int64)
    phi = 0.6180339887498949
    hues = (ids * phi) % 1.0
    cmap = plt.get_cmap("hsv")
    cols = cmap(hues)
    cols[:, 3] = alpha  # alpha
    return cols

def plot_pair_scatter_colored(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    noise_ids: np.ndarray,
    N_total: int,
    left_title: str,
    right_title: str,
    out_path: Path,
    dot_size: int = 8,
    sync_axes: bool = False,
    pad_ratio: float = 0.05,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    A = np.asarray(left_xy); B = np.asarray(right_xy)
    assert A.shape == B.shape and A.shape[1] == 2, "Expect (B,2) arrays for both panels."

    colors = colors_from_noise_ids(np.asarray(noise_ids), N_total)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4),
                             sharex=sync_axes, sharey=sync_axes)

    # 공통 정사각형 축 범위 (sync_axes=True)
    if sync_axes:
        both = np.vstack([A, B])
        xlim_all, ylim_all = _square_limits_from(both, pad_ratio=pad_ratio)

    for ax, data, title in ((axes[0], A, left_title), (axes[1], B, right_title)):
        ax.scatter(data[:, 0], data[:, 1], s=dot_size, c=colors, edgecolors="none")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)

        if sync_axes:
            ax.set_xlim(*xlim_all); ax.set_ylim(*ylim_all)
        else:
            xlim, ylim = _square_limits_from(data, pad_ratio=pad_ratio)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)




@torch.no_grad()
def collect_xt_seq_ddim(
    model: nn.Module,
    ddim: DDIMScheduler,
    z: torch.Tensor,
    t_stop: int = 0,            # 이 t에서의 x_t까지 포함해서 수집하고 멈춤
    return_ts: bool = True,
    device: str = "cuda",
    sample_steps: int = 100,    # 바깥에서 바꿀 수 있게 인자로 노출
    eta: float = 0.0,           # DDIM 결정론(기본 0.0)
):
    # 스케줄러 복제 및 세팅
    local = DDIMScheduler.from_config(ddim.config)
    local.set_timesteps(sample_steps, device=device)  # descending: [T'−1, ..., 0]

    # 상태 준비
    x = z.to(device).detach().clone()
    B = x.size(0)

    xs, ts = [], []

    was_training = model.training
    model.eval()
    try:
        for t_tensor in local.timesteps:                 # ex) [999, 979, ..., 0]
            t_int = int(t_tensor)

            # 1) (현재 t의) x_t 저장
            xs.append(x.detach().cpu().numpy())
            ts.append(t_int)

            # 2) 멈춤 조건: t_stop에 도달했으면 그 x_t까지 포함하고 break
            if t_int <= t_stop:
                break

            # 3) 다음 스텝으로 진행 (local 사용)
            t_b  = torch.full((B,), t_int, device=x.device, dtype=torch.long)
            x_in = local.scale_model_input(x, t_tensor)
            eps  = model(x_in, t_b)                      # 모델이 epsilon 예측한다고 가정
            out  = local.step(model_output=eps, timestep=t_tensor, sample=x, eta=eta)
            x    = out.prev_sample                       # x_{t-1}
    finally:
        if was_training:
            model.train()

    seq = np.stack(xs, axis=0)  # [K, B, ...]
    return (seq, np.asarray(ts, dtype=int)) if return_ts else seq



def sample_ddim_student(
    model, sample_scheduler, z, device, sample_steps=None, eta=0.0, t_sel=0,
):
    x = z.to(device)
    B = x.shape[0]

    local = DDIMScheduler.from_config(sample_scheduler.config)
    local.set_timesteps(sample_steps, device=device)

    xs = []

    model.train()
    for i, t in enumerate(local.timesteps):
        t_int = int(t)

        t_b = torch.full((B,), t_int, device=device, dtype=torch.long)
        x_in = local.scale_model_input(x, t)
        eps = model(x_in, t_b)
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample
        pred_x0 = out.pred_original_sample

        xs.append(pred_x0) 

        if t_int <= t_sel:
            break
    return xs


@torch.no_grad()
def sample_ddim_teacher(
    model, sample_scheduler, z, device, sample_steps=None, eta=0.0, t_sel=0,
):
    x = z.to(device)
    B = x.shape[0]

    local = DDIMScheduler.from_config(sample_scheduler.config)
    local.set_timesteps(sample_steps, device=device)

    xs = []

    model.eval()
    for t in local.timesteps:  # [T-1, ..., 0] 순서
        t_int = int(t)
        
        t_b = torch.full((B,), t_int, device=device, dtype=torch.long)
        x_in = local.scale_model_input(x, t)
        eps = model(x_in, t_b)
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample
        pred_x0 = out.pred_original_sample

        xs.append(pred_x0) 

        if t_int <= t_sel:
            break
    return xs


def sample_ddim_teacher_grad(
    model, sample_scheduler, z, device, sample_steps=None, eta=0.0, t_sel=0,
):
    x = z.to(device)
    B = x.shape[0]

    local = DDIMScheduler.from_config(sample_scheduler.config)
    local.set_timesteps(sample_steps, device=device)

    xs = []

    model.eval()
    for t in local.timesteps:  # [T-1, ..., 0] 순서
        t_int = int(t)
        
        t_b = torch.full((B,), t_int, device=device, dtype=torch.long)
        x_in = local.scale_model_input(x, t)
        eps = model(x_in, t_b)
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample
        pred_x0 = out.pred_original_sample

        xs.append(pred_x0) 

        if t_int <= t_sel:
            break
    return xs


def sample_ddim_inv_student(
    model, sample_scheduler, x0, device, sample_steps=None, eta=0.0,
):
    """
    DDIM 인버전(eta=0 가정): x_0 -> ... -> x_{T-1}
    Teacher에 다시 주입할 z를 만들 때 사용.
    """
    x = x0.to(device)
    B = x.shape[0]

    inv = DDIMInverseScheduler.from_config(sample_scheduler.config)

    inv.set_timesteps(sample_steps, device=device)      # [T' - 1, ..., 0]

    xs = [x]
    model.train()

    for t in inv.timesteps:
        t_b = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
        latent_in = inv.scale_model_input(x, t)
        eps = model(latent_in, t_b)
        out = inv.step(eps, t, x)
        x = out.prev_sample
        pred_x0 = out.pred_original_sample
        xs.append(pred_x0) 

    return xs  

# ===================== MODEL ===================== #

class LearnableHomography(nn.Module):
    """
    Row-vector convention: [x, y, 1] @ H^T -> [X, Y, W], (x',y') = (X/W, Y/W)

    - H: 하나의 3x3 행렬만 학습 (shape: (3,3))
    - 모든 timestep / 모든 sample에서 같은 H를 사용
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

        # 이제는 진짜 (3,3) 하나만 학습
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

    def forward(self, xy: torch.Tensor, t=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xy: (B,2)
        t: 더 이상 쓰지 않음 (호환성용)

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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        t = t.float().unsqueeze(1)
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device) * -1.0)
        angles = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1: emb = F.pad(emb, (0, 1))
        return emb

class MLPDenoiser(nn.Module):
    """ ε-predictor for 2D toy; used for both Teacher and Student. """
    def __init__(self, in_dim=2, time_dim=64, hidden=128, depth=3, out_dim=2):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(time_dim)
        layers = []
        for i in range(depth):
            layers += [nn.Linear(in_dim + time_dim if i == 0 else hidden, hidden), nn.SiLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out_dim)
    def forward(self, x, t):
        te = self.t_embed(t); h = torch.cat([x, te], dim=-1)
        return self.out(self.mlp(h))  # ε(x, t)

# ===================== SCHEDULERS ===================== #

def build_schedulers(num_train_timesteps: int):
    train_sched = DDPMScheduler(
        beta_schedule="linear",
        num_train_timesteps=num_train_timesteps,
        clip_sample=False,
    )
    train_sched.config.prediction_type = "epsilon"

    sample_sched = DDIMScheduler.from_config(train_sched.config)
    sample_sched.config.clip_sample = False
    sample_sched.config.prediction_type = "epsilon"  # DDIM도 동일하게

    return train_sched, sample_sched


def denormalize_np(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return arr * sigma + mu

def normalize_np(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (arr - mu) / sigma

def denormalize_torch(arr: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # NumPy 버전과 연산 로직은 동일합니다.
    return arr * sigma + mu

def normalize_torch(arr: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # NumPy 버전과 연산 로직은 동일합니다.
    return (arr - mu) / sigma
# ===================== Dataset ===================== #

class StudentX0Dataset(Dataset):
    def __init__(self, path: str, fmt: str, mu: np.ndarray, sigma: np.ndarray):
        self.X = self._load(path, fmt)  # (N,2) or (N,D)
        assert self.X.ndim == 2 and self.X.shape[1] >= 2, "Expect (N,2) or (N,D)"
        self.X = self.X[:, :2].astype(np.float32)
        self.X = normalize_np(self.X, mu, sigma).astype(np.float32)  # 모델 입력 스케일로
    def _load(self, path, fmt):
        p = Path(path)
        if fmt == "npy":
            return np.load(p)
        elif fmt == "csv":
            return np.loadtxt(p, delimiter=",")
        else:
            raise ValueError(f"Unsupported format: {fmt}")
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i]  # x0 (normalized)

def build_student_dataloader(cfg, mu, sigma):
    bs = int(cfg.get("student_dataset_batch_size", cfg["batch_size"]))
    ds = StudentX0Dataset(cfg["student_data_path"], cfg["student_data_format"], mu, sigma)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)


# ===================== Training ===================== #

def train_student_uniform_xt(cfg: Dict):
    """
    - 노이즈 풀에서 배치 샘플
    - t ~ Uniform{0..T-1}, 배치 내 t 동일
    - 같은 z로 Teacher/Student를 각각 DDIM 역진행
      * Teacher: eval + no_grad → x_t^T
      * Student: train + grad ON (전체 경로) → x_t^S
    - RKD(x_t^S, x_t^T)로 학습
    """
    out_dir = Path(cfg["out_dir"]); (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # 스케줄러/모델
    train_sched, sample_sched = build_schedulers(cfg["T"])
    ddim = make_ddim_like(train_sched, device, cfg["T"])

    teacher = MLPDenoiser(2, cfg["teacher_time_dim"], cfg["teacher_hidden"], cfg["teacher_depth"], 2).to(device)
    teacher.load_state_dict(torch.load(cfg["teacher_ckpt"], map_location=device), strict=True)
    for p in teacher.parameters(): p.requires_grad = False
    teacher.eval()

    student = MLPDenoiser(2, cfg["student_time_dim"], cfg["student_hidden"], cfg["student_depth"], 2).to(device)
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

    # === Learnable H ===
    H_module = LearnableHomography(init_9=cfg["H_init"], eps=cfg["H_eps"]).to(device)

    if cfg.get("resume_H_ckpt"):
        pH = Path(cfg["resume_H_ckpt"])
        if pH.exists():
            H_module.load_state_dict(torch.load(pH, map_location=device), strict=True)
            print("[RESUME] Loaded H:", pH)

    if not cfg.get("use_learnable_H", True):
        for p in H_module.parameters():
            p.requires_grad = False

    # opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    opt = torch.optim.AdamW(
        list(student.parameters()) + list(H_module.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # W&B
    if cfg["use_wandb"]:
        wandb.login()
        wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg)
        wandb.define_metric("step")
        wandb.define_metric("loss/*", step_metric="step")
        wandb.define_metric("loss_by_t/*", step_metric="step")
        wandb.define_metric("loss_by_tp1_t/*", step_metric="step")

    # 노이즈 풀
    noise_path = ensure_noise_pool(cfg)
    noise_pool = np.load(noise_path)  # (N,D), 작으면 통째 로드 OK

    mu_teacher, sigma_teacher = load_norm_stats(cfg["teacher_data_stats"])
    mu_student, sigma_student = load_norm_stats(cfg["student_data_stats"])

    mu_teacher_tensor, sigma_teacher_tensor = load_norm_stats_torch(cfg["teacher_data_stats"], device)
    mu_student_tensor, sigma_student_tensor = load_norm_stats_torch(cfg["student_data_stats"], device)

    # --- Student 도메인 dataloader (diffusion loss용) ---
    student_loader = build_student_dataloader(cfg, mu_student, sigma_student)
    student_iter = iter(student_loader)

    def next_student_batch():
        nonlocal student_iter
        try:
            x0 = next(student_iter)  # (B_s, 2) torch.Tensor로 들어옴
        except StopIteration:
            student_iter = iter(student_loader)
            x0 = next(student_iter)
        if not torch.is_tensor(x0):
            x0 = torch.as_tensor(x0, dtype=torch.float32)
        return x0.to(device, non_blocking=True)


    T = int(cfg["T"])
    total_steps = int(cfg.get("epochs_total", 50000))
    B = int(cfg["batch_size"])

    for step_i in range(1, total_steps + 1):
        # 1) t ~ Uniform{0..T-1} (배치 내 동일)
        # t_sel = int(np.random.randint(0, T))
        t_sel = int(0)
        # t_sel = int(T-1)
        # t_sel = torch.randint(low=0, high=T, size=(B,), device=device, dtype=torch.long)
        
        # z 샘플
        # 동일 노이즈 안에서만 사용하므로 오버피팅 역할
        z = sample_noise_batch(noise_pool, B, device)
        # # random noise
        z = torch.randn_like(z)

        # ddim_steps = int(cfg.get("rkd_ddim_steps_to_t", 5))
        # ddim_steps = 50

        # possible_steps = [200, 195, 191, 190, 187, 174, 161, 157, 148, 135]
        # ddim_steps = random.choice(possible_steps)

        ddim_steps = int(np.random.randint(30,51))
        # ddim_steps = 151

        with torch.no_grad():
            xt_T_seq = sample_ddim_teacher(
                model=teacher, sample_scheduler=ddim, z=z, 
                device=device, sample_steps=ddim_steps,
                eta=float(cfg.get("ddim_eta", 0.0)), t_sel=t_sel,
            )

        student.train()
        xt_S_seq = sample_ddim_student(
            model=student, sample_scheduler=ddim, z=z, 
            device=device, sample_steps=ddim_steps,
            eta=float(cfg.get("ddim_eta", 0.0)), t_sel=t_sel,
        )


        ##################  STUDENT DOMAIN DATA ##################
        x0_batch = next_student_batch()  # shape (B_s, 2)
        student.train()

        S_inv_z_seq = sample_ddim_inv_student(
            model=student, sample_scheduler=ddim, x0=x0_batch, 
            device=device, sample_steps=ddim_steps,
            eta=float(cfg.get("ddim_eta", 0.0)),
        )
        x0_inv_T = sample_ddim_teacher_grad(
            model=teacher, sample_scheduler=ddim, z=S_inv_z_seq[-1], 
            device=device, sample_steps=ddim_steps,
            eta=float(cfg.get("ddim_eta", 0.0)), t_sel=0,
        )

       ##################  RKD LOSSES ##################

        rkd_loss = torch.tensor(0.0, device=device)
        inversion_loss = torch.tensor(0.0, device=device)
        invinv_loss = torch.tensor(0.0, device=device)
        fid_loss = torch.tensor(0.0, device=device)
        x0_S_same_loss = torch.tensor(0.0, device=device)

        rkd_s_d_list, rkd_t_d_list = [], []

        xt_T_x0 = xt_T_seq[-1]

        xt_S_seq_denorm = [denormalize_torch(H_module(x)[0], mu_student_tensor, sigma_student_tensor) for x in xt_S_seq]
        xt_T_x0_denorm = denormalize_torch(xt_T_x0, mu_teacher_tensor, sigma_teacher_tensor)
        x0_batch_denorm = denormalize_torch(H_module(x0_batch)[0], mu_student_tensor, sigma_student_tensor)
        x0_inv_T_denorm = denormalize_torch(x0_inv_T[-1], mu_teacher_tensor, sigma_teacher_tensor)

        x0_batch_denorm_no_H = denormalize_torch(x0_batch, mu_student_tensor, sigma_student_tensor)
        xt_S_seq_denorm_no_H = denormalize_torch(xt_S_seq[-1], mu_student_tensor, sigma_student_tensor) 

        xt_S_seq_denorm_no_H_seq = [denormalize_torch(x, mu_student_tensor, sigma_student_tensor) for x in xt_S_seq]


        # --- RKD (전 타임스텝) ---  (W_RKD != 0일 때만 계산)
        if cfg["W_RKD"] != 0:
            for xt_S in xt_S_seq_denorm:
                rkd_s_d = torch.pdist(xt_S, p=2).clamp_min(1e-12)
                rkd_t_d = torch.pdist(xt_T_x0_denorm, p=2).clamp_min(1e-12)
                rkd_s_d_list.append(rkd_s_d)
                rkd_t_d_list.append(rkd_t_d)

        # --- INV: x_t^S vs x_0^S / x_0^{T,inv} --- (W_INV != 0일 때만)
        inv_s_d = inv_t_d = None
        if cfg["W_INV"] != 0:
            s_full = torch.cdist(xt_S_seq_denorm[-1], x0_batch_denorm, p=2)
            t_full = torch.cdist(xt_T_x0_denorm, x0_inv_T_denorm, p=2)
            inv_s_d = s_full.reshape(-1).clamp_min(1e-12)
            inv_t_d = t_full.reshape(-1).clamp_min(1e-12)
        # --- INVINV: x_0^S vs x_0^{T,inv} --- (W_INVINV != 0일 때만)
        invinv_s_d = invinv_t_d = None
        if cfg["W_INVINV"] != 0:
            invinv_s_d = torch.pdist(x0_batch_denorm, p=2).clamp_min(1e-12)
            invinv_t_d = torch.pdist(x0_inv_T_denorm, p=2).clamp_min(1e-12)

        # --- mean normalization (글로벌: 활성화된 loss들만) ---
        student_parts = []
        teacher_parts = []

        if cfg["W_RKD"] != 0 and len(rkd_s_d_list) > 0:
            rkd_s_d_all = torch.cat(rkd_s_d_list, dim=0)
            rkd_t_d_all = torch.cat(rkd_t_d_list, dim=0)
            student_parts.append(rkd_s_d_all)
            teacher_parts.append(rkd_t_d_all)
        if cfg["W_INV"] != 0 and inv_s_d is not None:
            student_parts.append(inv_s_d)
            teacher_parts.append(inv_t_d)
        if cfg["W_INVINV"] != 0 and invinv_s_d is not None:
            student_parts.append(invinv_s_d)
            teacher_parts.append(invinv_t_d)
            
        if len(student_parts) > 0:
            student_all = torch.cat(student_parts, dim=0)
            teacher_all = torch.cat(teacher_parts, dim=0)
            student_mean = student_all.mean()
            teacher_mean = teacher_all.mean()
        else:
            student_mean = torch.tensor(1.0, device=device)
            teacher_mean = torch.tensor(1.0, device=device)

        # --- 실제 사용하는 애들만 정규화 ---
        if cfg["W_RKD"] != 0:
            rkd_s_d_list = [d / student_mean for d in rkd_s_d_list]
            rkd_t_d_list = [d / teacher_mean for d in rkd_t_d_list]
        if cfg["W_INV"] != 0 and inv_s_d is not None:
            inv_s_d = inv_s_d / student_mean
            inv_t_d = inv_t_d / teacher_mean
        if cfg["W_INVINV"] != 0 and invinv_s_d is not None:
            invinv_s_d = invinv_s_d / student_mean
            invinv_t_d = invinv_t_d / teacher_mean

        # --- FID  ---
        if cfg["W_FID"] != 0:
            fid_student = fid_gaussian_torch(xt_S_seq_denorm_no_H, x0_batch_denorm_no_H)
            fid_teacher = fid_gaussian_torch(xt_T_x0_denorm, x0_inv_T_denorm)
        else:
            fid_student = torch.tensor(0.0, device=device)
            fid_teacher = torch.tensor(0.0, device=device)

        # --- SAME (trajectory 수축 regularizer) ---
        if cfg["W_SAME"] != 0:
            xt_S_stack = torch.stack(xt_S_seq_denorm_no_H_seq, dim=0)   # [K, B, D]
            K = xt_S_stack.size(0)
            if cfg.get("same_mode", "mean") == "mean":
                # (1) mean 기준: 모든 timestep이 서로 가깝게
                xt_S_time_mean = xt_S_stack.mean(dim=0, keepdim=True)
                xt_S_time_mean_expand = xt_S_time_mean.expand_as(xt_S_stack)

                same_raw = F.mse_loss(
                    xt_S_stack,              # [K, B, D]
                    xt_S_time_mean_expand,   # [K, B, D]
                    reduction="mean",
                )
                x0_S_same_loss = cfg["W_SAME"] * same_raw
            elif cfg["same_mode"] == "last":
                # (2) last 기준: 마지막 pred_x0이 anchor
                ref = xt_S_stack[-1].detach()
                xt_S_except_last = xt_S_stack[:-1]
                ref_expand = ref.unsqueeze(0).expand_as(xt_S_except_last)

                same_raw = F.mse_loss(
                    xt_S_except_last,   # [K-1, B, D]
                    ref_expand,         # [K-1, B, D]
                    reduction="mean",
                )
                x0_S_same_loss = cfg["W_SAME"] * same_raw

        # losses
        if cfg["W_RKD"] != 0:
            for (rkd_s_d, rkd_t_d) in zip(rkd_s_d_list, rkd_t_d_list):
                rkd_loss += cfg["W_RKD"] * F.mse_loss(rkd_s_d, rkd_t_d, reduction="mean") / len(xt_S_seq)
        if cfg["W_INV"] != 0 and inv_s_d is not None:
            inversion_loss += cfg["W_INV"] * F.mse_loss(inv_s_d, inv_t_d, reduction="mean")
        if cfg["W_INVINV"] != 0 and invinv_s_d is not None:
            invinv_loss += cfg["W_INVINV"] * F.mse_loss(invinv_s_d, invinv_t_d, reduction="mean")
        if cfg["W_FID"] != 0:
            fid_loss += cfg["W_FID"] * (fid_student + fid_teacher)
        # TOTAL
        loss = rkd_loss + inversion_loss + invinv_loss + fid_loss + x0_S_same_loss


        opt.zero_grad()
        loss.backward()
        if cfg.get("max_grad_norm", 0) > 0:
            nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(H_module.parameters()),
                cfg["max_grad_norm"]
            )
        opt.step()
        # if (step_i % max(1, total_steps // 20) == 0) or (step_i == 1):
        if (step_i % 100 == 0) or (step_i == 1):
            print(f"[step {step_i:06d}] rkd={rkd_loss.item():.6f}  x0_S_same={x0_S_same_loss.item():.6f}  inv={inversion_loss.item():.6f}   invinv={invinv_loss.item():.6f}  fid_T_loss={fid_teacher.item():.6f}  fid_S_loss={fid_student.item():.6f}  fid_loss={fid_loss.item():.6f}  total={loss.item():.6f}")


        if cfg["use_wandb"]:
            wandb.log({
                "step": step_i,
                "t": t_sel,
                "loss/total": float(loss),
                "loss/rkd": float(rkd_loss),
                "loss/same": float(x0_S_same_loss),
                "loss/inv": float(inversion_loss),
                "loss/invinv": float(invinv_loss),
                "loss/fid": float(fid_loss),
                "lr": opt.param_groups[0]["lr"],
            }, step=step_i)


        if cfg["use_wandb"]:
            wandb.log({
                "loss_raw/rkd": float(rkd_loss) / cfg["W_RKD"] if cfg["W_RKD"] != 0 else 0.0,
                "loss_raw/same": float(x0_S_same_loss) / cfg["W_SAME"] if cfg["W_SAME"] != 0 else 0.0,
                "loss_raw/inv": float(inversion_loss) / cfg["W_INV"] if cfg["W_INV"] != 0 else 0.0,
                "loss_raw/invinv": float(invinv_loss) / cfg["W_INVINV"] if cfg["W_INVINV"] != 0 else 0.0,
                "loss_raw/fid": float(fid_loss) / cfg["W_FID"] if cfg["W_FID"] != 0 else 0.0,
            }, step=step_i)


        # 7) (옵션) 시각화: 그대로 유지 (원 코드와 동일)
        if (step_i % cfg["vis_interval_epochs"] == 0) or (step_i == total_steps):
            with torch.no_grad():
                B_plot = min(int(cfg.get("n_vis", 1024)), B)
                z_vis  = sample_noise_batch(noise_pool, B_plot, device)

                seq_T, ts = collect_xt_seq_ddim(teacher, ddim, z_vis, t_stop=0, return_ts=True, device=device, sample_steps=int(cfg["rkd_ddim_steps_to_t"]))
                seq_S, _  = collect_xt_seq_ddim(student.eval(), ddim, z_vis, t_stop=0, return_ts=True, device=device, sample_steps=int(cfg["rkd_ddim_steps_to_t"]))
                seq_S_H   = apply_H_to_seq_per_t(seq_S, ts, H_module, device)

                stride = 1
                idxs = np.arange(0, len(ts), stride)
                seq_T_s = seq_T[idxs]
                seq_S_s = seq_S[idxs]
                seq_S_H_s = seq_S_H[idxs]
                ts_s    = ts[idxs]


                seq_T_s_plot = denormalize_np(seq_T_s, mu_teacher, sigma_teacher)
                seq_S_s_plot = denormalize_np(seq_S_s, mu_student, sigma_student)
                seq_S_H_s_plot = denormalize_np(seq_S_H_s, mu_student, sigma_student)
                
                _ = save_and_log_xt_triplets_for_all_t(
                    teacher_seq = seq_T_s_plot,
                    student_seq = seq_S_s_plot,
                    studentH_seq= seq_S_H_s_plot,
                    ts         = ts_s,
                    noise_ids  = np.arange(B_plot),
                    out_dir    = out_dir,
                    step_i     = step_i,
                    subdir_name = "xt_triplets",        # 디렉토리/로그 키 접두사
                    sync_axes  = bool(cfg.get("vis_xt_sync_axes", False)),
                    dot_size   = 6,
                    use_wandb  = bool(cfg["use_wandb"]),
                )

        # 7) (옵션) 시각화: 랜덤 노이즈에서 학생 모델로 x0 샘플링하여 저장
        if (step_i % cfg["vis_interval_epochs"] == 0) or (step_i == total_steps):
            @torch.no_grad()
            def sample_x0_ddim(model, sample_scheduler, num_samples, device, sample_steps, dim=2, eta=0.0):
                sample_scheduler = DDIMScheduler.from_config(sample_scheduler.config)
                sample_scheduler.set_timesteps(sample_steps, device=device)                
                x = torch.randn(num_samples, dim, device=device)
                model.eval()
                for t in sample_scheduler.timesteps:      # [T-1, ..., 0]
                    t_b  = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
                    x_in = sample_scheduler.scale_model_input(x, t)
                    eps  = model(x_in, t_b)
                    x    = sample_scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample
                model.train()
                return x

            student.eval()

            B_plot = 8192
            x0_s = sample_x0_ddim(
                model=student,
                sample_scheduler=ddim,
                num_samples=B_plot,
                device=device,
                sample_steps=int(cfg.get("rkd_ddim_steps_to_t", 100)),
                dim=int(cfg.get("dim", 2)),
                eta=float(cfg.get("ddim_eta", 0.0)),
            )

            # 역정규화
            x0_s_plot = denormalize_np(x0_s.detach().cpu().numpy(), mu_student, sigma_student)

            # === 여기서 확실히 디렉토리 보장 ===
            figs_dir = Path(cfg["out_dir"]) / "figs"
            figs_dir.mkdir(parents=True, exist_ok=True)

            png_path = figs_dir / f"samples_step{step_i:06d}.png"

            # 실제 저장 (정사각형 축 범위 적용)
            plt.figure(figsize=(4, 4))
            plt.scatter(x0_s_plot[:, 0], x0_s_plot[:, 1], s=6, edgecolors="none")
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
            xlim, ylim = _square_limits_from(x0_s_plot, pad_ratio=0.05)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            plt.title(f"Student samples (x0) @ step {step_i}")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close()

            if cfg["use_wandb"]:
                wandb.log({"img_T/student_samples": wandb.Image(str(png_path))}, step=step_i)


            # (2) H(t=0) 적용본
            with torch.no_grad():
                x0_s_H, _ = H_module(x0_s)
            x0_s_H_plot = denormalize_np(x0_s_H.detach().cpu().numpy(), mu_teacher, sigma_teacher)
            png_path_H   = figs_dir / f"samples_H_t0_step{step_i:06d}.png"

            plt.figure(figsize=(4, 4))
            plt.scatter(x0_s_H_plot[:, 0], x0_s_H_plot[:, 1], s=6, edgecolors="none")
            ax = plt.gca(); ax.set_aspect("equal", adjustable="box")
            xlim, ylim = _square_limits_from(x0_s_H_plot, pad_ratio=0.05)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            plt.title(f"Student samples (x0) with H(t=0) @ step {step_i}")
            plt.tight_layout(); plt.savefig(png_path_H, dpi=150, bbox_inches="tight"); plt.close()

            if cfg["use_wandb"]:
                wandb.log({"img_T/student_samples_H": wandb.Image(str(png_path_H))}, step=step_i)




                # --- H matrices full HTML viz over ALL timesteps (paged) ---
        # --- H matrix HTML viz (단일 3x3 H) ---
        if cfg.get("use_wandb", False) and ((step_i % cfg["vis_interval_epochs"] == 0) or (step_i == total_steps)):
            import html
            with torch.no_grad():
                eps = 1e-12

                # (3,3) -> CPU float
                H_mat: torch.Tensor = H_module.H.detach().float().cpu()    # (3,3)
                H_norm = H_mat / max(float(H_mat[2, 2].abs()), eps)        # proj scale 제거

                def mat_to_pre(M: torch.Tensor) -> str:
                    arr = M.numpy()
                    s = np.array2string(
                        arr,
                        formatter={'float_kind': lambda x: f"{x: .5f}"},
                        max_line_width=200,
                    )
                    return f"<pre style='margin:0'>{html.escape(s)}</pre>"

                # 역행렬(가능하면)
                try:
                    H_inv = torch.linalg.inv(H_mat)
                    H_inv_norm = H_inv / max(float(H_inv[2, 2].abs()), eps)
                    inv_html  = mat_to_pre(H_inv)
                    invn_html = mat_to_pre(H_inv_norm)
                except Exception:
                    inv_html  = "<pre style='margin:0'>(singular)</pre>"
                    invn_html = "<pre style='margin:0'>(singular)</pre>"

                html_block = f"""
                <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 12px;">
                  <div style="margin-bottom:6px;">Learnable H (single 3x3)</div>
                  <table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;">
                    <thead>
                      <tr>
                        <th>H</th>
                        <th>H / H[2,2]</th>
                        <th>H<span style="vertical-align:super;">-1</span></th>
                        <th>H<span style="vertical-align:super;">-1</span> / [2,2]</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>{mat_to_pre(H_mat)}</td>
                        <td>{mat_to_pre(H_norm)}</td>
                        <td>{inv_html}</td>
                        <td>{invn_html}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                """.strip()

                wandb.log({"H_vis/table": wandb.Html(html_block)}, step=step_i)

                # 간단 통계
                det = torch.det(H_mat).item()
                try:
                    cond = torch.linalg.cond(H_mat).item()
                except Exception:
                    cond = float("nan")

                wandb.log({
                    "H_stats/det_mean": float(det),
                    "H_stats/cond_mean": float(cond),
                }, step=step_i)





        # 8) (옵션) 주기적 체크포인트
        if (step_i % (cfg["vis_interval_epochs"]) == 0) or (step_i == total_steps):
            ckpt_path = out_dir / f"ckpt_student_step{step_i:06d}.pt"
            torch.save(student.state_dict(), ckpt_path)
            torch.save(H_module.state_dict(), out_dir / f"ckpt_H_step{step_i:06d}.pt")
            print("[CKPT]", ckpt_path)

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
