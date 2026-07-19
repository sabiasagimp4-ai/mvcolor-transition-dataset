#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
year_palette_micd.py
==============================================================================
Binned-KMeans原論文(Pocol, Istead & Kaplan, CBMI 2025)のMICD
(Minimum Colour Difference, 論文Eq.4の重み付き版)をそのまま使い、
year_palette.py で作った「年ごとの代表パレット」同士の距離を定量化する。

MICD(A,B) = 1/2 * ( Σ_i w_i * min_j(ΔE_ij) + Σ_j w_j * min_i(ΔE_ij) )
  ΔE_ij: CIELAB空間でのユークリッド距離(CIE76)
  w_i, w_j: 各パレットにおけるその色の重み(Σw=1)

出す数値:
  ・隣接年どうしのMICD(前年→今年の「色の移動距離」)
  ・2005年と各年とのMICD(基準年からどれだけ離れたか)

使い方:
  python year_palette_micd.py --manifest ../videos.csv --extract-dir /d/自自論_video_data/binned_kmeans_out --k 8 --out out/year_palette_micd.csv
"""

import argparse, os, csv, math
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import deltaE_cie76

from year_palette import fnum, read_manifest, collect_year_colors, rgb_to_hue, allocate_k_per_bin, BIN_ORDER


# shot_dominant_color_binned_kmeans.py と同一のsRGB D65 CIELAB変換式
_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]])
_Xn, _Yn, _Zn = 0.95047, 1.00000, 1.08883


def _inv_gamma(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _f_lab(t):
    return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)


def rgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = _inv_gamma(rgb)
    xyz = lin @ _M_RGB2XYZ.T
    x, y, z = xyz[..., 0] / _Xn, xyz[..., 1] / _Yn, xyz[..., 2] / _Zn
    fx, fy, fz = _f_lab(x), _f_lab(y), _f_lab(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def weighted_kmeans_palette_lab(colors, weights, k):
    """rgbで加重KMeansし、(Lab centers, weight_share)を返す(色相角順にソートしない、MICD専用)。
    【非推奨・後方互換用】ビンを無視した単純KMeansのため、画面占有面積の大きい
    肌色・暗色などが全クラスタを占有しうる。年単位のパレット作成には
    weighted_kmeans_palette_lab_binned を使うこと。"""
    X = np.stack(colors)
    w = np.array(weights)
    k_eff = min(k, X.shape[0])
    km = KMeans(n_clusters=k_eff, random_state=0, n_init=10)
    km.fit(X, sample_weight=w)
    labels = km.labels_
    centers_rgb = km.cluster_centers_
    total_w = w.sum()
    shares = np.array([w[labels == i].sum() / total_w for i in range(k_eff)])
    centers_lab = rgb_to_lab(centers_rgb)
    return centers_lab, shares


def weighted_kmeans_palette_lab_binned(by_bin, k):
    """ビンごとに加重KMeansをかけ、(Lab centers, weight_share)を返す(色相角順にソートしない)。
    by_bin: {hue_bin: [(rgb, weight), ...]}。各ビンに最低1クラスタを保証してから
    残りを重み比例で配分する(year_palette.allocate_k_per_bin)ため、
    year_palette.weighted_kmeans_palette_binnedと同じビン保護をLab空間(MICD/CEMD用)でも行う。"""
    bin_weights = {name: sum(w for _, w in entries) for name, entries in by_bin.items() if entries}
    alloc = allocate_k_per_bin(bin_weights, k)
    total_w_all = sum(bin_weights.values())
    centers_rgb_list, shares_list = [], []
    for name in BIN_ORDER:
        k_bin = alloc.get(name, 0)
        entries = by_bin.get(name)
        if k_bin < 1 or not entries:
            continue
        colors = [c for c, _ in entries]
        weights = [w for _, w in entries]
        X = np.stack(colors)
        w = np.array(weights)
        k_eff = min(k_bin, X.shape[0])
        km = KMeans(n_clusters=k_eff, random_state=0, n_init=10)
        km.fit(X, sample_weight=w)
        labels = km.labels_
        centers = km.cluster_centers_
        for i in range(k_eff):
            mask = labels == i
            cw = w[mask].sum()
            if cw <= 0:
                continue
            centers_rgb_list.append(centers[i])
            shares_list.append(cw / total_w_all)
    centers_lab = rgb_to_lab(np.stack(centers_rgb_list))
    shares = np.array(shares_list)
    return centers_lab, shares


def micd(labA, wA, labB, wB):
    """重み付きMICD(Binned-KMeans論文 Eq.4)。"""
    # 全ペアのΔE(CIE76ユークリッド距離、scikit-image実装。自前実装と数値が完全一致することを確認済み)
    dE = deltaE_cie76(labA[:, None, :], labB[None, :, :])  # (N,M)
    term1 = (wA * dE.min(axis=1)).sum()
    term2 = (wB * dE.min(axis=0)).sum()
    return 0.5 * (term1 + term2)


def main():
    ap = argparse.ArgumentParser(description="年ごとの代表パレットのMICD(色差)を計算")
    ap.add_argument("--manifest", default="../videos.csv")
    ap.add_argument("--extract-dir", default="/d/自自論_video_data/binned_kmeans_out")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="out/year_palette_micd.csv")
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)
    by_year = collect_year_colors(manifest, args.extract_dir)
    years = sorted(by_year)

    palettes = {}
    for year in years:
        palettes[year] = weighted_kmeans_palette_lab_binned(by_year[year], args.k)

    base_year = years[0]
    labA0, wA0 = palettes[base_year]

    rows = []
    prev_lab, prev_w = None, None
    for year in years:
        lab, w = palettes[year]
        d_prev = micd(prev_lab, prev_w, lab, w) if prev_lab is not None else ""
        d_base = micd(labA0, wA0, lab, w)
        rows.append({"year": year, "micd_vs_prev_year": d_prev, "micd_vs_base_year": d_base})
        prev_lab, prev_w = lab, w

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year", "micd_vs_prev_year", "micd_vs_base_year"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"基準年: {base_year}", flush=True)
    for r in rows:
        pv = f"{r['micd_vs_prev_year']:.2f}" if r['micd_vs_prev_year'] != "" else "  -  "
        print(f"  {r['year']}  前年比MICD={pv}  {base_year}年比MICD={r['micd_vs_base_year']:.2f}", flush=True)
    print(f"出力: {args.out}", flush=True)


if __name__ == "__main__":
    main()
