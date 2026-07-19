#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_trends.py
==============================================================================
year_summary.csv / year_hue_hist.csv / video_summary.csv から
「年 vs 特徴」のグラフを描く。集約層_design_260716.md 3節・5節に従い、
散布(動画1本1点)＋回帰直線・相関係数、年ごとの箱ひげ、色相ヒストの積み上げを出す。

本研究は探索的（どの次元にトレンドが出るかを事前に決めない）ため、
6次元(平均彩度・平均明度・Colorfulness・%chromatic・暖色寒色比・パレット多様性)
それぞれについて同じフォーマットのグラフを機械的に出す。特定の次元を強調しない。

使い方:
  python plot_trends.py --video-summary out/video_summary.csv --year-summary out/year_summary.csv --year-hue out/year_hue_hist.csv --outdir out/plots
"""

import argparse, csv, os
from collections import defaultdict
import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic"]
matplotlib.rcParams["axes.unicode_minus"] = False

DIM_LABELS = {
    "mean_s": "平均彩度 (mean saturation)",
    "mean_v": "平均明度 (mean value/brightness)",
    "colorfulness": "Colorfulness (Hasler-Süsstrunk)",
    "pct_chromatic": "%chromatic (有彩色画素の割合)",
    "warm_cool_ratio": "暖色/寒色比 (warm share among warm+cool)",
    "hue_entropy": "パレット多様性 (hue histogram entropy, bits)",
    "avg_shot_dur": "平均ショット長 (秒、編集ペースの逆指標)",
    "pct_fast_cut": "速いカットの割合 (is_fast_cut, 編集ペース)",
    "mean_L_lab": "CIELAB明度 L* (ドミナントカラー系)",
    "mean_C_lab": "CIELABクロマ C* (知覚的な彩度、ドミナントカラー系)",
    "color_dynamism": "色のダイナミズム (連続ショット間のCIELAB色差の平均)",
    "color_dynamism_std": "色のダイナミズムの標準偏差 (変化の激しさのばらつき)",
    "color_dynamism_max": "色のダイナミズムの最大値 (動画内で最も激しかった色の切り替わり)",
}
LOG_SCALE_DIMS = {"avg_shot_dur"}  # 1ショットのみのMV(単一テイク演出)が極端な外れ値になるため対数軸にする
HUE_BINS = ["red", "orange", "yellow", "green", "teal", "blue", "purple"]
HUE_COLORS = {"red": "#d62728", "orange": "#ff7f0e", "yellow": "#bcbd22",
              "green": "#2ca02c", "teal": "#17becf", "blue": "#1f77b4", "purple": "#9467bd"}


def fnum(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scatter_box_plot(dim, video_rows, year_rows, outpath):
    """散布(全動画)＋回帰直線・相関係数 と 年ごとの箱ひげを1枚に並べる。"""
    pts = [(int(r["year"]), fnum(r.get(dim))) for r in video_rows
           if r.get("year") and fnum(r.get(dim)) is not None]
    if not pts:
        return False, None
    years = np.array([p[0] for p in pts], dtype=float)
    vals = np.array([p[1] for p in pts], dtype=float)

    by_year = defaultdict(list)
    for y, v in pts:
        by_year[y].append(v)
    sorted_years = sorted(by_year)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    log_scale = dim in LOG_SCALE_DIMS

    # 左: 散布＋回帰直線＋相関係数・p値(scipy.stats.linregress、帰無仮説=傾き0)
    ax1.scatter(years, vals, s=14, alpha=0.45, color="#1f77b4")
    reg = None
    if len(set(years.tolist())) >= 2:
        reg = scipy_stats.linregress(years, vals)
        xs = np.linspace(years.min(), years.max(), 100)
        ax1.plot(xs, reg.slope * xs + reg.intercept, color="#d62728", linewidth=2)
        sig = "*" if reg.pvalue < 0.05 else ""
        ax1.set_title(f"散布＋回帰  (slope={reg.slope:.4g}/年, r={reg.rvalue:.3f}, "
                       f"p={reg.pvalue:.4f}{sig}, n={len(vals)})")
    else:
        ax1.set_title(f"散布 (n={len(vals)})")
    ax1.set_xlabel("year"); ax1.set_ylabel(dim)
    if log_scale:
        ax1.set_yscale("log")

    # 右: 年ごとの箱ひげ
    ax2.boxplot([by_year[y] for y in sorted_years], tick_labels=[str(y) for y in sorted_years], showfliers=True)
    ax2.set_title("年ごとの分布(箱ひげ)" + ("・対数軸(1ショットのみの動画が外れ値のため)" if log_scale else ""))
    ax2.set_xlabel("year"); ax2.set_ylabel(dim)
    ax2.tick_params(axis="x", rotation=90)
    if log_scale:
        ax2.set_yscale("log")

    fig.suptitle(DIM_LABELS.get(dim, dim))
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    stats_row = None
    if reg is not None:
        stats_row = {"dim": dim, "n": len(vals), "slope_per_year": reg.slope,
                     "r": reg.rvalue, "p_value": reg.pvalue, "significant_p<0.05": reg.pvalue < 0.05}
    return True, stats_row


def hue_stack_plot(year_hue_rows, outpath):
    years = [int(r["year"]) for r in year_hue_rows if r.get("n_videos") and int(r["n_videos"]) > 0]
    if not years:
        return False
    series = {b: [] for b in HUE_BINS}
    for r in year_hue_rows:
        if not r.get("n_videos") or int(r["n_videos"]) == 0:
            continue
        for b in HUE_BINS:
            series[b].append(fnum(r.get(f"hue_{b}")) or 0.0)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(years, *[series[b] for b in HUE_BINS],
                 labels=HUE_BINS, colors=[HUE_COLORS[b] for b in HUE_BINS])
    ax.set_xlabel("year"); ax.set_ylabel("色相分布の割合(年内平均)")
    ax.set_title("色相分布の推移(7ビン積み上げ)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="年vs色特徴のトレンドグラフを描く")
    ap.add_argument("--video-summary", default="out/video_summary.csv")
    ap.add_argument("--year-summary", default="out/year_summary.csv")
    ap.add_argument("--year-hue", default="out/year_hue_hist.csv")
    ap.add_argument("--outdir", default="out/plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    video_rows = read_csv(args.video_summary)
    year_hue_rows = read_csv(args.year_hue)

    made = []
    stats_rows = []
    for dim in DIM_LABELS:
        outpath = os.path.join(args.outdir, f"trend_{dim}.png")
        ok, stats_row = scatter_box_plot(dim, video_rows, None, outpath)
        if ok:
            made.append(outpath)
        if stats_row:
            stats_rows.append(stats_row)

    hue_out = os.path.join(args.outdir, "trend_hue_histogram.png")
    if hue_stack_plot(year_hue_rows, hue_out):
        made.append(hue_out)

    stats_path = os.path.join(args.outdir, "..", "regression_stats.csv")
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dim", "n", "slope_per_year", "r", "p_value", "significant_p<0.05"])
        w.writeheader()
        for r in stats_rows:
            w.writerow(r)

    print(f"完了: {len(made)} 枚", flush=True)
    for p in made:
        print(f"  - {p}", flush=True)
    print(f"回帰統計(p値含む) -> {stats_path}", flush=True)


if __name__ == "__main__":
    main()
