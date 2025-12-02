#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math, os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int):
    np.random.seed(seed)


def rot2d(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


def gen_eyes(n_each: int,
             left_center=(-0.8, 0.6),
             right_center=(0.8, 0.6),
             std=0.05) -> np.ndarray:
    """두 개의 눈(가우시안 클러스터) 샘플"""
    L = np.random.randn(n_each, 2) * std + np.array(left_center,  dtype=np.float64)
    R = np.random.randn(n_each, 2) * std + np.array(right_center, dtype=np.float64)
    return np.concatenate([L, R], axis=0)


def gen_mouth(n: int,
              center=(0.0, -0.3),
              radius=1.6,
              deg_low=200,
              deg_high=340,
              thickness=0.04) -> np.ndarray:
    """
    웃는 곡선(원호) 샘플. 두께(thickness)는 반지름 방향 가우시안.
    deg_low~deg_high 범위의 각도에서 균등 샘플.
    """
    thetas = np.deg2rad(np.random.uniform(deg_low, deg_high, size=n))
    # 원호 위 중심 좌표
    base = np.stack([np.cos(thetas), np.sin(thetas)], axis=1) * radius
    # 반경 방향 두께(정규분포)
    dr = np.random.randn(n, 1) * thickness
    pts = (radius + dr) * np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    pts += np.array(center, dtype=np.float64)
    return pts


def apply_transform(xy: np.ndarray,
                    scale: float = 1.0,
                    rotate_deg: float = 0.0,
                    translate=(0.0, 0.0),
                    anisotropic_scale=None) -> np.ndarray:
    """
    변환: (선택)비등방 스케일 → 등방 스케일 → 회전 → 평행이동
    - anisotropic_scale=(sx, sy)를 주면 먼저 적용
    """
    X = xy.astype(np.float64)

    if anisotropic_scale is not None:
        sx, sy = anisotropic_scale
        S = np.array([[sx, 0.0],
                      [0.0, sy]], dtype=np.float64)
        X = X @ S.T

    if scale != 1.0:
        X = X * float(scale)

    if rotate_deg != 0.0:
        R = rot2d(math.radians(rotate_deg))
        X = X @ R.T

    if translate is not None:
        tx, ty = translate
        X = X + np.array([tx, ty], dtype=np.float64)

    return X.astype(np.float64)


def save_scatter(xy: np.ndarray, path: Path, title: str = "", s: int = 6):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.2, 4.2))
    plt.scatter(xy[:, 0], xy[:, 1], s=s, edgecolors="none")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_heatmap(xy: np.ndarray, path: Path, bins: int = 128):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 자동 범위 + padding
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    dx, dy = xmax - xmin, ymax - ymin
    pad_x, pad_y = 0.05 * max(dx, 1e-6), 0.05 * max(dy, 1e-6)

    H, xedges, yedges = np.histogram2d(
        xy[:, 0], xy[:, 1],
        bins=bins,
        range=[[xmin - pad_x, xmax + pad_x], [ymin - pad_y, ymax + pad_y]]
    )
    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(H.T, origin="lower", aspect="equal",
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.title("Heatmap (all)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

NUM=65536
SCALE=2
ROTATION=60
TRANS_X=50
TRANS_Y=-20

def main():
    ap = argparse.ArgumentParser(description="Generate 2D Smile Distribution")
    # 기본
    ap.add_argument("--out_dir", type=str, default=f"smile_data_n{NUM}_scale{SCALE}_rot{ROTATION}_trans_{TRANS_X}_{TRANS_Y}", help="출력 폴더")
    ap.add_argument("--num_points", type=int, default=NUM, help="전체 포인트 개수 (train.npy/csv로 저장)")
    ap.add_argument("--val_ratio", type=float, default=0.0, help="검증 분할 비율 (0~1)")
    ap.add_argument("--seed", type=int, default=42)

    # 눈(가우시안) 관련
    ap.add_argument("--eye_ratio", type=float, default=0.30, help="전체 중 눈(양쪽 합) 비율 (0~1)")
    ap.add_argument("--eye_std", type=float, default=0.05, help="눈 가우시안 표준편차")
    ap.add_argument("--eye_left", type=float, nargs=2, default=[-0.8, 0.6], help="왼쪽 눈 중심 (x y)")
    ap.add_argument("--eye_right", type=float, nargs=2, default=[0.8, 0.6], help="오른쪽 눈 중심 (x y)")

    # 입(원호) 관련
    ap.add_argument("--mouth_center", type=float, nargs=2, default=[0.0, -0.3], help="입 원의 중심 (x y)")
    ap.add_argument("--mouth_radius", type=float, default=1.6, help="입 원의 반지름")
    ap.add_argument("--mouth_deg_range", type=float, nargs=2, default=[200, 340], help="원호 각도 범위 (deg_low deg_high)")
    ap.add_argument("--mouth_thickness", type=float, default=0.04, help="원호 두께(반경 방향 가우시안 표준편차)")

    # 전역 변환
    ap.add_argument("--scale", type=float, default=SCALE, help="등방 스케일")
    ap.add_argument("--rotate_deg", type=float, default=ROTATION, help="시계반대 회전(deg)")
    ap.add_argument("--translate", type=float, nargs=2, default=[TRANS_X, TRANS_Y], help="평행이동 (tx ty)")
    ap.add_argument("--anisotropic_scale", type=float, nargs=2, default=None, help="비등방 스케일 (sx sy), 생략 가능")

    # 저장 옵션
    ap.add_argument("--csv", action="store_true", help="CSV도 함께 저장 (기본: npy만 저장)")
    ap.add_argument("--scatter_dot", type=int, default=6, help="산점도 점 크기")

    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N = int(args.num_points)
    eye_ratio = float(args.eye_ratio)
    val_ratio = max(0.0, min(1.0, float(args.val_ratio)))

    # 눈/입 개수 배분
    n_eye_total = int(round(N * eye_ratio))
    # 양쪽 눈 균등 분배(홀수 보정)
    n_eye_each = n_eye_total // 2
    n_eye_total = n_eye_each * 2
    n_mouth = max(0, N - n_eye_total)

    # --- 원시 스마일 좌표 생성(캐논컬 공간) ---
    eyes = gen_eyes(
        n_each=n_eye_each,
        left_center=tuple(args.eye_left),
        right_center=tuple(args.eye_right),
        std=float(args.eye_std)
    )
    mouth = gen_mouth(
        n=n_mouth,
        center=tuple(args.mouth_center),
        radius=float(args.mouth_radius),
        deg_low=float(args.mouth_deg_range[0]),
        deg_high=float(args.mouth_deg_range[1]),
        thickness=float(args.mouth_thickness)
    )

    data = np.concatenate([eyes, mouth], axis=0) if n_mouth > 0 else eyes

    # --- 전역 변환(비등방 → 등방 → 회전 → 평행이동) ---
    data = apply_transform(
        data,
        scale=float(args.scale),
        rotate_deg=float(args.rotate_deg),
        translate=tuple(args.translate),
        anisotropic_scale=tuple(args.anisotropic_scale) if args.anisotropic_scale is not None else None
    )

    # 셔플
    perm = np.random.permutation(data.shape[0])
    data = data[perm].astype(np.float32)

    # train/val 분할 (요청 폴더 형식에 맞춰 파일명은 train으로 저장하되, 분포 그림은 둘 다 따로 출력)
    n_val = int(round(N * val_ratio))
    n_train = N - n_val
    data_train = data[:n_train]
    data_val = data[n_train:] if n_val > 0 else None

    # --- 저장 ---
    # npy (필수)
    np.save(out_dir / "train.npy", data_train.astype(np.float32))
    # csv (옵션)
    if args.csv:
        with open(out_dir / "train.csv", "w", encoding="utf-8") as f:
            f.write("x,y\n")
            np.savetxt(f, data_train, fmt="%.6f", delimiter=",")

    # 메타데이터 기록
    meta = {
        "num_points": N,
        "num_train": int(n_train),
        "num_val": int(n_val),
        "seed": int(args.seed),
        "eye_ratio": eye_ratio,
        "eye_std": float(args.eye_std),
        "eye_left": list(map(float, args.eye_left)),
        "eye_right": list(map(float, args.eye_right)),
        "mouth_center": list(map(float, args.mouth_center)),
        "mouth_radius": float(args.mouth_radius),
        "mouth_deg_range": [float(args.mouth_deg_range[0]), float(args.mouth_deg_range[1])],
        "mouth_thickness": float(args.mouth_thickness),
        "scale": float(args.scale),
        "anisotropic_scale": (list(map(float, args.anisotropic_scale)) if args.anisotropic_scale is not None else None),
        "rotate_deg": float(args.rotate_deg),
        "translate": list(map(float, args.translate)),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # --- 시각화 ---
    save_scatter(data, out_dir / "fig_scatter_all.png",
                 title=f"Smile (all)  N={N}", s=int(args.scatter_dot))
    save_heatmap(data, out_dir / "fig_heatmap_all.png", bins=128)

    save_scatter(data_train, out_dir / "fig_scatter_train.png",
                 title=f"Smile (train)  N={n_train}", s=int(args.scatter_dot))
    if data_val is not None and len(data_val) > 0:
        save_scatter(data_val, out_dir / "fig_scatter_val.png",
                     title=f"Smile (val)  N={len(data_val)}", s=int(args.scatter_dot))

    print(f"[OK] Saved to: {out_dir.resolve()}")
    print(" - train.npy", ("(and train.csv)" if args.csv else ""))
    print(" - fig_scatter_all.png / fig_scatter_train.png / fig_scatter_val.png")
    print(" - fig_heatmap_all.png")
    print(" - metadata.json")


if __name__ == "__main__":
    main()
