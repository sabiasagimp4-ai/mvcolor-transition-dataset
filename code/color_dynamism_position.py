#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
color_dynamism_position.py
==============================================================================
「MV内で色がどのように変化するか」の構造(パターン)そのものに時代差があるかを見る。
既存のcolor_dynamism(動画全体の平均ΔE)は1本=1つの数字に潰してしまうため、
「動画のどの相対時間位置で色変化が起きているか」を別軸として追加する。

各動画について、連続ショット間のCIELAB色差ΔE(color_dynamismと同じ定義)を、
そのショットが動画の何%地点で起きたか(相対時間位置 0〜1)と紐付けて記録する。

1. 相対位置ごとのΔEカーブ: 位置を10ビンに分け、年代グループ別に平均ΔEを比較
   (「終盤に色が激しく変わる構造(山型)は増えているか」を見る)
2. 動画ごとの「後半/前半 比」: 後半(相対位置>=0.5)の平均ΔEを前半の平均ΔEで割った値。
   1より大きいほど「終盤にかけて色変化が激しくなる」構造。これを年で回帰し、
   構造そのものの時代トレンドを1つの数値検定に落とす。

新規ダウンロード・新規抽出なし。既存のtimeseries.csv(weight_rank列)とvideo_summary.csv
(total_dur_s)だけから計算する。

使い方:
  python color_dynamism_position.py --manifest ../videos.csv --video-summary out_transnetv2/video_summary.csv \
      --extract-dir /d/自自論_video_data/binned_kmeans_out_transnetv2 --outdir out_transnetv2
"""

import argparse, os, csv, math
from collections import defaultdict
import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic"]
matplotlib.rcParams["axes.unicode_minus"] = False

N_BINS = 10


def fnum(v):
    v = (v or "").strip() if isinstance(v, str) else v
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_video_summary(path):
    """id -> (year, total_dur_s)"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y = fnum(r.get("year"))
            d = fnum(r.get("total_dur_s"))
            if y is not None and d is not None and d > 0:
                out[r["id"]] = (int(y), d)
    return out


def per_video_transitions(ts_path, total_dur):
    """(relpos, deltaE) のリストを返す。weight_rank==1の色をショット順に並べ、
    連続ショット間のΔEと、その遷移が起きた相対時間位置(前のショットのend_s/total_dur)を対にする。"""
    by_shot = {}
    with open(ts_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("weight_rank") or "").strip() != "1":
                continue
            sh = int(r["shot"])
            end_s = fnum(r.get("end_s"))
            L, a, b = fnum(r.get("L_lab")), fnum(r.get("a_lab")), fnum(r.get("b_lab"))
            if None in (end_s, L, a, b):
                continue
            by_shot[sh] = (end_s, L, a, b)

    seq = [by_shot[sh] for sh in sorted(by_shot)]
    out = []
    for (end1, L1, a1, b1), (end2, L2, a2, b2) in zip(seq, seq[1:]):
        dE = math.sqrt((L2 - L1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)
        relpos = min(max(end1 / total_dur, 0.0), 1.0)
        out.append((relpos, dE))
    return out


def main():
    ap = argparse.ArgumentParser(description="MV内の色変化(ΔE)を相対時間位置ごとに集計し、年代差を見る")
    ap.add_argument("--manifest", default="../videos.csv")
    ap.add_argument("--video-summary", default="out_transnetv2/video_summary.csv")
    ap.add_argument("--extract-dir", default="/d/自自論_video_data/binned_kmeans_out_transnetv2")
    ap.add_argument("--outdir", default="out_transnetv2")
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)
    vs = read_video_summary(args.video_summary)

    # 全体: (year, relpos, deltaE) を集める
    all_records = []
    # 動画ごとの前半/後半比
    ratio_rows = []

    n_ok, n_skip = 0, 0
    for m in manifest:
        vid = (m.get("id") or "").strip()
        if not vid or vid not in vs:
            continue
        year, total_dur = vs[vid]
        base = os.path.splitext((m.get("video_file") or f"{vid}.mp4"))[0]
        ts_path = os.path.join(args.extract_dir, base, f"{base}_binned_kmeans_timeseries.csv")
        if not os.path.isfile(ts_path):
            n_skip += 1
            continue
        trans = per_video_transitions(ts_path, total_dur)
        if len(trans) < 4:  # 前半・後半それぞれに最低限の遷移数を要求
            n_skip += 1
            continue
        for relpos, dE in trans:
            all_records.append((year, relpos, dE))

        first = [dE for relpos, dE in trans if relpos < 0.5]
        second = [dE for relpos, dE in trans if relpos >= 0.5]
        if len(first) >= 2 and len(second) >= 2:
            m1, m2 = sum(first) / len(first), sum(second) / len(second)
            if m1 > 0:
                ratio_rows.append({"id": vid, "year": year,
                                    "deltaE_first_half": m1, "deltaE_second_half": m2,
                                    "back_loaded_ratio": m2 / m1})
        n_ok += 1

    print(f"対象 {n_ok}本 / スキップ {n_skip}本(遷移数不足・出力なし)", flush=True)

    os.makedirs(args.outdir, exist_ok=True)

    # ---- 1. 相対位置ごとの平均ΔE(年代グループ別) ----
    ERA_GROUPS = [("2005-2011", 2005, 2011), ("2012-2018", 2012, 2018), ("2019-2025", 2019, 2025)]
    bin_edges = np.linspace(0, 1, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    position_csv_rows = []
    curves = {}
    for label, y0, y1 in ERA_GROUPS:
        sums = np.zeros(N_BINS)
        counts = np.zeros(N_BINS)
        for year, relpos, dE in all_records:
            if not (y0 <= year <= y1):
                continue
            b = min(int(relpos * N_BINS), N_BINS - 1)
            sums[b] += dE
            counts[b] += 1
        means = np.divide(sums, counts, out=np.full(N_BINS, np.nan), where=counts > 0)
        curves[label] = means
        for bc, mean_v, n in zip(bin_centers, means, counts):
            position_csv_rows.append({"era": label, "position_bin_center": round(float(bc), 3),
                                       "mean_deltaE": mean_v, "n": int(n)})

    pos_csv_path = os.path.join(args.outdir, "color_dynamism_position.csv")
    with open(pos_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["era", "position_bin_center", "mean_deltaE", "n"])
        w.writeheader()
        w.writerows(position_csv_rows)

    # ---- 2. 動画ごとの後半/前半比 → 年で回帰 ----
    ratio_csv_path = os.path.join(args.outdir, "color_dynamism_backloaded_ratio.csv")
    with open(ratio_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "year", "deltaE_first_half", "deltaE_second_half", "back_loaded_ratio"])
        w.writeheader()
        w.writerows(ratio_rows)

    years = [r["year"] for r in ratio_rows]
    ratios = [r["back_loaded_ratio"] for r in ratio_rows]
    res = scipy_stats.linregress(years, ratios)
    print(f"\n後半/前半ΔE比 vs 年: n={len(ratios)}  slope={res.slope:.5f}/年  "
          f"r={res.rvalue:.3f}  p={res.pvalue:.4f}  "
          f"{'有意' if res.pvalue < 0.05 else '非有意'}", flush=True)

    stats_path = os.path.join(args.outdir, "color_dynamism_backloaded_stats.csv")
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "slope_per_year", "r", "p_value"])
        w.writerow(["back_loaded_ratio", len(ratios), res.slope, res.rvalue, res.pvalue])

    # ---- プロット ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for label, _, _ in ERA_GROUPS:
        ax.plot(bin_centers, curves[label], marker="o", label=label)
    ax.set_xlabel("動画内の相対時間位置(0=冒頭, 1=終盤)")
    ax.set_ylabel("平均ΔE(CIELAB色差)")
    ax.set_title("相対時間位置ごとの色変化の激しさ(年代グループ別)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(years, ratios, alpha=0.35, s=14)
    xs = np.array(sorted(set(years)))
    ax.plot(xs, res.intercept + res.slope * xs, color="crimson")
    ax.set_xlabel("年")
    ax.set_ylabel("後半ΔE平均 / 前半ΔE平均")
    ax.set_title(f"「終盤ほど色変化が激しい」度合いの推移\nr={res.rvalue:.3f}, p={res.pvalue:.4f}")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "color_dynamism_position.png")
    plt.savefig(plot_path, dpi=140)

    print(f"\n出力:")
    print(f"  {pos_csv_path}")
    print(f"  {ratio_csv_path}")
    print(f"  {stats_path}")
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()
