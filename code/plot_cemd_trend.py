#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_cemd_trend.py
==============================================================================
CEMD(Color-based Earth Mover's Distance, Skaff et al. 2011, CIEDE2000版)の
「基準年(2005)からの色距離」トレンドを単独でプロットする。

論文の結果節ではMICDではなくCEMD(Skaff et al.に忠実な最終手法)のみを
報告する方針としたため、MICDとの並列比較プロット(plot_cemd_vs_micd.py)から
CEMD単独のプロットに差し替えた。

使い方:
  python plot_cemd_trend.py --cemd out/year_palette_cemd.csv --out out/cemd_trend.png
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic"]
matplotlib.rcParams["axes.unicode_minus"] = False
from scipy import stats as scipy_stats


def read_csv(path, value_col):
    years, vals = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row[value_col]
            years.append(int(row["year"]))
            vals.append(float(v) if v != "" else None)
    return years, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cemd", default="out/year_palette_cemd.csv")
    ap.add_argument("--out", default="out/cemd_trend.png")
    args = ap.parse_args()

    years, vals = read_csv(args.cemd, "cemd_vs_base_year")

    xs = np.array([y for y, v in zip(years, vals) if v is not None and v != 0.0])
    vs = np.array([v for v in vals if v is not None and v != 0.0])
    slope, intercept, r, p, se = scipy_stats.linregress(xs, vs)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(years, vals, s=45, color="#3a6ea5", zorder=3)
    ax.plot(xs, slope * xs + intercept, color="#c0392b", linewidth=1.5, zorder=2)
    sig = "有意(p<0.05)" if p < 0.05 else "非有意"
    ax.set_title(
        f"CEMD(CIEDE2000, Skaff et al.準拠): 2005年基準の色距離\n"
        f"slope={slope:.3f}/年  r={r:.3f}  p={p:.3f} ({sig})\n"
        f"(TransNetV2版, n=1509, 基準年の自明点は除外)",
        fontsize=11,
    )
    ax.set_xlabel("年")
    ax.set_ylabel("CEMD (2005年比)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"出力: {args.out}")
    print(f"n={len(xs)}  slope={slope:.4f}/年  r={r:.3f}  p={p:.4f}  -> {sig}")


if __name__ == "__main__":
    main()
