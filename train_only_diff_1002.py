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

from diffusers import DDPMScheduler, DDIMScheduler
from torch.utils.data import Dataset, DataLoader
import itertools
# ===================== CONFIG ===================== #


# WANDB_NAME=f"1002_rkd_xt_b512_init_full_diff_loss_w100"0
# WANDB_NAME=f"1002_rkd_xt_b512_init_full"

TT = 400
WANDB_NAME=f"1202_only_diff_loss_B256_teacher65536_T{TT}"
# WANDB_NAME=f"1002_only_diff_loss_student16"

CONFIG = {
    # I/O
    "device": f"cuda:5",
    "out_dir": f"runs/{WANDB_NAME}",
    # teacher / student
    "teacher_ckpt": "ckpt_teacher_T1000_step370000_1021.pt",  # REQUIRED
    "student_init_ckpt": "",                     
    # "student_init_ckpt": "runs/1002_only_diff_loss_student8/ckpt_student_step500000.pt",                     
    "resume_student_ckpt": f"",        
    "teacher_data_stats": "smile_data_n65536_scale10_rot0_trans_0_0/normalization_stats.json",

    # diffusion loss 가중치
    "W_DIFF": 1.0,                               # ε-pred MSE 가중치

    "batch_size": 256,
    "num_noises": 8192, 
    "epochs_total": 1000000,          # 총 스텝 수 (기존 epochs_per_stage 대신 사용)

    "noise_pool_file": None,        # None이면 out_dir/data/noises_pool.npy 로 저장
    "regen_noise_pool": False,      # True면 항상 새로 만듦
    
    # schedule / time
    "T": TT,                 # total diffusion steps (timesteps = 0..T-1)

    "seed": 42,
    # RKD weights
    "W_RKD": 1.0,
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
    "vis_interval_epochs": 20000,
    "n_vis": 8192,       # 경로를 수집/표시할 noise 개수
    "ddim_steps": 25,
    "ddim_eta": 0.0,
    # wandb
    "use_wandb": True,
    "wandb_project": "RKD-DKDM-AICA-1202-Teacher",
    "wandb_run_name": WANDB_NAME,
}

CONFIG.update({
    # student 데이터 경로/형식
    # "student_data_path": "smile_data_n16_scale2_rot60_trans_50_-20/train.npy",   # 혹은 .csv
    "student_data_path": "smile_data_n65536_scale10_rot0_trans_0_0/train.npy",   # 혹은 .csv
    "student_data_format": "npy",                # "npy" | "csv"
    "student_dataset_batch_size": 256,          # 없으면 batch_size 사용
})

# ===================== UTILS ===================== #
import re

from pathlib import Path

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

def sample_noise_batch(noise_pool: np.ndarray, B: int, device: torch.device) -> torch.Tensor:
    """노이즈 풀에서 인덱스 랜덤 샘플로 배치 구성"""
    N = noise_pool.shape[0]
    idx = np.random.randint(0, N, size=B)
    z = torch.from_numpy(noise_pool[idx]).to(device=device, dtype=torch.float32)
    return z

def get_xt_ddim_student(
    model: nn.Module,
    ddim: DDIMScheduler,
    z: torch.Tensor,
    t_sel: int,
    grad_mode: str = "full",
    last_k: int = 1,
) -> torch.Tensor:
    """
    DDIM(eta=0)로 z=x_{T-1}에서 t_sel까지 내려가 x_t 반환.
    - "full": 모든 스텝 grad
    - "last": 마지막 last_k 스텝만 grad (기본 1스텝: t=t_sel+1)
    구현을 2-패스(no_grad -> grad)로 나눠 grad 경로가 확실히 생기게 함.
    """
    assert 0 <= t_sel <= int(ddim.config.num_train_timesteps) - 1
    x = z
    B = x.size(0)

    # timesteps: 내림차순 [T-1, ..., 0]
    t_list = [int(t) for t in ddim.timesteps]  # python list로
    Tm1 = t_list[0]

    if grad_mode == "full":
        # 전체 grad 한 번에
        for t_int in t_list:
            if t_int == t_sel:
                return x
            t_tensor = torch.tensor(t_int, device=x.device, dtype=ddim.timesteps.dtype)
            t_b  = torch.full((B,), t_int, device=x.device, dtype=torch.long)
            x_in = ddim.scale_model_input(x, t_tensor)
            eps  = model(x_in, t_b)                   # ← grad 경로 생김
            x    = ddim.step(model_output=eps, timestep=t_tensor, sample=x, eta=0.0).prev_sample
        raise RuntimeError("DDIM loop ended without reaching t_sel")

    elif grad_mode == "last":
        # 1) no-grad 프리런: T-1 -> (t_start) 까지 내려와 x_{t_start} 상태 만들기
        t_start = min(Tm1, t_sel + max(1, int(last_k)))  # 최소 1스텝은 grad 켤 것
        with torch.no_grad():
            for t_int in t_list:
                if t_int == t_start:
                    break  # step 하지 않고 x_{t_start} 상태로 종료
                t_tensor = torch.tensor(t_int, device=x.device, dtype=ddim.timesteps.dtype)
                t_b  = torch.full((B,), t_int, device=x.device, dtype=torch.long)
                x_in = ddim.scale_model_input(x, t_tensor)
                eps  = model(x_in, t_b)                 # no_grad
                x    = ddim.step(model_output=eps, timestep=t_tensor, sample=x, eta=0.0).prev_sample

        # 2) grad 켠 채로 t_start -> (t_sel+1)까지 스텝 실행해 x_{t_sel} 만들기
        #    (no-grad 구간과의 경계에서 detach로 리프화)
        x = x.detach()
        for t_int in t_list[t_list.index(t_start):]:
            if t_int == t_sel:
                return x
            t_tensor = torch.tensor(t_int, device=x.device, dtype=ddim.timesteps.dtype)
            t_b  = torch.full((B,), t_int, device=x.device, dtype=torch.long)
            x_in = ddim.scale_model_input(x, t_tensor)
            eps  = model(x_in, t_b)                     # ← grad 경로 생김(마지막 k스텝)
            x    = ddim.step(model_output=eps, timestep=t_tensor, sample=x, eta=0.0).prev_sample
        raise RuntimeError("DDIM loop ended without reaching t_sel")

    else:
        raise ValueError(f"Unknown grad_mode: {grad_mode}")




def get_xt_ddim(model: nn.Module, ddim: DDIMScheduler, z: torch.Tensor, t_sel: int) -> torch.Tensor:
    """
    DDIM(eta=0)으로 z=x_{T-1}에서 t_sel까지 '모델을 거치며' 내려가 x_t를 반환.
    - 호출 컨텍스트에서 no_grad 여부를 제어(Teacher는 no_grad, Student는 grad ON).
    - 루프에서 t_tensor는 [T-1..0] 내림차순.
      각 반복의 '시작 시점'에 x는 현재 t에 해당하는 상태이므로,
      t_int == t_sel 에 도달하면 step 없이 x를 반환.
    """
    assert 0 <= t_sel <= int(ddim.config.num_train_timesteps) - 1
    x = z
    B = x.size(0)
    for t_tensor in ddim.timesteps:
        t_int = int(t_tensor)
        if t_int == t_sel:
            return x  # 현재 x가 곧 x_t
        t_b  = torch.full((B,), t_int, device=x.device, dtype=torch.long)
        x_in = ddim.scale_model_input(x, t_tensor)
        eps  = model(x_in, t_b)
        x    = ddim.step(model_output=eps, timestep=t_tensor, sample=x, eta=0.0).prev_sample
    raise RuntimeError("DDIM loop ended without reaching t_sel")



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

def save_and_log_xt_pairs_for_all_t(
    left_seq: np.ndarray,   # [K,B,2] Teacher
    right_seq: np.ndarray,  # [K,B,2] Student
    ts: np.ndarray,         # [K]     정수 timestep들 (T-1→...→0)
    noise_ids: np.ndarray,  # [B]     색상 매핑용
    out_dir: Path,
    step_i: int,
    sync_axes: bool = False,
    dot_size: int = 6,
    use_wandb: bool = False,
):
    base = out_dir / "figs" / f"xt_pairs_step{step_i:06d}"
    base.mkdir(parents=True, exist_ok=True)

    logged = {}
    for k, t in enumerate(ts):
        left_xy  = left_seq[k]   # (B,2)
        right_xy = right_seq[k]  # (B,2)
        path = base / f"t{int(t):03d}.png"

        plot_pair_scatter_colored(
            left_xy=left_xy,
            right_xy=right_xy,
            noise_ids=noise_ids,
            N_total=len(noise_ids),
            left_title = fr"Teacher $x_t$ (t={int(t)})",
            right_title= fr"Student $x_t$ (t={int(t)})",
            out_path=path,
            dot_size=dot_size,
            sync_axes=sync_axes,
        )

        if use_wandb:
            logged[f"img/xt_pairs/t{int(t):03d}"] = wandb.Image(str(path))

    if use_wandb and logged:
        wandb.log(logged, step=step_i)

    return base  # 저장된 폴더 경로


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
    t_stop: int = 0,
    return_ts: bool = True,
):
    """
    같은 z(x_{T-1})에서 시작해 DDIM(eta=0)으로 t_stop까지 한 스텝씩 내려가며
    각 시점의 x_t를 모두 수집.
    return:
      - seq: [K, B, 2] (K = 수집된 timestep 수, T-1→...→t_stop 순서)
      - ts : [K]       (정수 timesteps)
    """
    x = z.clone()
    B = x.size(0)
    xs, ts = [], []
    for t_tensor in ddim.timesteps:      # [T-1, ..., 0]
        t_int = int(t_tensor)
        xs.append(x.detach().cpu().numpy())  # 현재 x가 곧 x_t
        ts.append(t_int)
        if t_int == t_stop:
            break
        t_b  = torch.full((B,), t_int, device=x.device, dtype=torch.long)
        x_in = ddim.scale_model_input(x, t_tensor)
        eps  = model(x_in, t_b)
        x    = ddim.step(model_output=eps, timestep=t_tensor, sample=x, eta=0.0).prev_sample

    seq = np.stack(xs, axis=0)  # [K, B, 2]
    return (seq, np.asarray(ts, dtype=int)) if return_ts else seq




@torch.no_grad()
def sample_ddim_teacher(
    model, sample_scheduler, z, t_idx, device, sample_steps, eta=0.0
):
    sample_scheduler.set_timesteps(sample_steps, device=device)
    timesteps = sample_scheduler.timesteps  # descending

    x = z.to(device)
    B = x.shape[0]
    t_idx = int(t_idx)

    t_min = int(timesteps[-1]); t_max = int(timesteps[0])
    target_low  = max(t_min, min(t_max, t_idx))      # t_idx

    x_t_low  = None

    for t in timesteps:
        t_int = int(t)
        t_b = torch.full((B,), t_int, device=device, dtype=torch.long)

        # 한 스텝 (t -> t-1)
        x_in = sample_scheduler.scale_model_input(x, t)
        eps  = model(x_in, t_b)
        x    = sample_scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample

        curr_time = t_int - 1

        if x_t_low  is None and curr_time <= target_low:
            x_t_low = x
            break

    return x_t_low.detach()





@torch.no_grad()
def _ddim_step_no_grad(model, scheduler, x, t, t_b, eta):
    x_in = scheduler.scale_model_input(x, t)
    eps  = model(x_in, t_b)
    return scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample

def _ddim_step_grad(model, scheduler, x, t, t_b, eta):
    x_in = scheduler.scale_model_input(x, t)
    eps  = model(x_in, t_b)
    return scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample

def sample_ddim_student(
    model, sample_scheduler, z, t_idx, grad_t, device, sample_steps, eta=0.0):

    sample_scheduler.set_timesteps(sample_steps, device=device)
    timesteps = sample_scheduler.timesteps  # descending

    x = z.to(device)
    B = x.shape[0]
    t_idx  = int(t_idx)
    grad_t = int(grad_t)

    # 타겟 t 설정 (클램프)
    t_min = int(timesteps[-1])
    t_max = int(timesteps[0])
    target_low  = max(t_min, min(t_max, t_idx))      # t_idx

    x_t_low  = None  # x_{t_idx}
    entered_grad_window = False

    for t in timesteps:
        t_int = int(t)
        t_b = torch.full((B,), t_int, device=device, dtype=torch.long)

        # grad 창: t < t_idx + grad_t (t가 작아질수록 t_idx에 가까워지므로 여기서부터 grad ON)
        in_grad_window = (t_int <= t_idx + grad_t)
        if in_grad_window and not entered_grad_window:
            x = x.detach().requires_grad_(True)
            entered_grad_window = True

        # t → t-1 한 스텝 진행
        if in_grad_window:
            x = _ddim_step_grad(model, sample_scheduler, x, t, t_b, eta)
        else:
            x = _ddim_step_no_grad(model, sample_scheduler, x, t, t_b, eta)

        # 방금 만든 x는 개념적으로 시점 (t_int - 1)
        curr_time = t_int - 1

        # 다음 x_{t_idx} 스냅샷
        if x_t_low is None and curr_time <= target_low:
            x_t_low = x
            break  # 둘 다 얻었으니 종료


    return x_t_low



# ===================== MODEL ===================== #

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
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
    )
    train_sched.config.prediction_type = "epsilon"

    sample_sched = DDIMScheduler.from_config(train_sched.config)
    sample_sched.config.clip_sample = False
    sample_sched.config.prediction_type = "epsilon"  # DDIM도 동일하게

    return train_sched, sample_sched


# ===================== PREPARE (SAVE TEACHER ε & x) ===================== #

def denormalize_np(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return arr * sigma + mu

def normalize_np(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
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

# ===================== RKD on epsilon ===================== #

def loss_rkd_xt(
    xt_s: torch.Tensor,      # [B, D]
    xt_s_2: torch.Tensor,      # [B, D]
    xt_t: torch.Tensor,      # [B, D]
    xt_t_2: torch.Tensor,      # [B, D]
    use_mu_norm: bool = True,
    w_rkd: float = 1.0,
    w_norm: float = 0.0,
    eps: float = 1e-12,
):
    B = xt_s.size(0)
    if B < 2:
        z = xt_s.new_zeros(())
        return {"total": z, "rkd": z, "norm": z, "t_norm": z, "s_norm": z}

    # pairwise distances on epsilon vectors
    t_full = torch.cdist(xt_t.detach(), xt_t_2.detach(), p=2)
    s_full = torch.cdist(xt_s,           xt_s_2,           p=2)
    iu = torch.triu_indices(B, B, offset=1, device=xt_s.device)
    t_d = t_full[iu].clamp_min(eps); s_d = s_full[iu].clamp_min(eps)

    if use_mu_norm:
        t_d = t_d / t_d.mean().clamp_min(eps)
        s_d = s_d / s_d.mean().clamp_min(eps)
    
    loss_rkd = w_rkd * F.mse_loss(s_d, t_d, reduction="mean")

    total = loss_rkd
    return {"total": total, "rkd": loss_rkd}


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

    opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

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


    # --- Student 도메인 dataloader (diffusion loss용) ---
    student_loader = build_student_dataloader(cfg, mu_teacher, sigma_teacher)
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


        # ===================== NEW: diffusion ε-MSE loss =====================
        # 학생 도메인 x0 배치 가져오기 (정규화 완료 상태)
        x0_batch = next_student_batch()  # shape (B_s, 2)

        # RKD에서 사용한 것과 동일한 t를 공유(원하면 독립적으로 뽑아도 OK)
        # t_b_s = torch.full((x0_batch.shape[0],), t_sel, device=device, dtype=torch.long)
        t_b_s = torch.randint(
            low=0,
            high=T,                 # high는 exclusive → 0 ~ T-1
            size=(B,),              # 배치 크기
            device=device,
            dtype=torch.long
        )
        # 표준 훈련 루틴: ε 샘플 → x_t 생성 → ε̂ 예측 → MSE
        eps = torch.randn_like(x0_batch)
        x_t_for_diff = train_sched.add_noise(x0_batch, eps, t_b_s)  # q(x_t|x0, ε, t)

        eps_pred = student(x_t_for_diff, t_b_s)  # prediction_type='epsilon'
        diff_loss = cfg["W_DIFF"] * F.mse_loss(eps_pred, eps, reduction="mean")
        # ===============================

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

        # 7) (옵션) 시각화: 랜덤 노이즈에서 학생 모델로 x0 샘플링하여 저장
        if (step_i % cfg["vis_interval_epochs"] == 0) or (step_i == total_steps):
            @torch.no_grad()
            def sample_x0_ddim(model, sample_scheduler, num_samples, device, sample_steps, dim=2, eta=0.0):
                sample_scheduler.set_timesteps(sample_steps, device=device)
                x = torch.randn(num_samples, dim, device=device)
                for t in sample_scheduler.timesteps:      # [T-1, ..., 0]
                    t_b  = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
                    x_in = sample_scheduler.scale_model_input(x, t)
                    eps  = model(x_in, t_b)
                    x    = sample_scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample
                return x

            student.eval()

            B_plot = 8192
            x0_s = sample_x0_ddim(
                model=student,
                sample_scheduler=ddim,
                num_samples=B_plot,
                device=device,
                sample_steps=int(cfg["T"]),
                dim=int(cfg.get("dim", 2)),
                eta=float(cfg.get("ddim_eta", 0.0)),
            )

            # 역정규화
            x0_s_plot = denormalize_np(x0_s.detach().cpu().numpy(), mu_teacher, sigma_teacher)

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
                wandb.log({"img/student_samples": wandb.Image(str(png_path))}, step=step_i)


            # 8) (옵션) 주기적 체크포인트
            if (step_i % (cfg["vis_interval_epochs"]) == 0) or (step_i == total_steps):
                ckpt_path = out_dir / f"ckpt_student_step{step_i:06d}.pt"
                torch.save(student.state_dict(), ckpt_path)
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
