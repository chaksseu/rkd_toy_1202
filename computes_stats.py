#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np

def load_2d(path: Path):
    if path.suffix.lower() == ".npy":
        X = np.load(path)
    elif path.suffix.lower() == ".csv":
        X = np.loadtxt(path, delimiter=",", skiprows=1) if "x,y" in path.read_text(errors="ignore") else np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported file: {path}")
    assert X.ndim == 2 and X.shape[1] >= 2, f"Expect (N,2+) array, got {X.shape}"
    return X[:, :2].astype(np.float32)

def main():
    ap = argparse.ArgumentParser(description="Compute per-dim mean/std for 2D data and save JSON")
    ap.add_argument("--data", default="smile_data_n65536_scale10_rot0_trans_0_0/train.npy", help="Path to train.npy or train.csv")
    ap.add_argument("--out_json", default="normalization_stats.json", help="Output JSON filename")
    args = ap.parse_args()

    data_path = Path(args.data)
    X = load_2d(data_path)

    mean = X.mean(axis=0).tolist()
    std  = X.std(axis=0, ddof=0).tolist()  # population std (모델 정규화에 보통 사용)

    out = {
        "mean": [float(mean[0]), float(mean[1])],
        "std":  [float(std[0]),  float(std[1])],
        "source": str(data_path),
        "num_samples": int(X.shape[0]),
    }

    out_path = (data_path.parent / args.out_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[OK] saved: {out_path.resolve()}")

if __name__ == "__main__":
    main()
