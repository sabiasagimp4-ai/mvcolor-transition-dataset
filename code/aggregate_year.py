#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_year.py
==============================================================================
video_summary.csv を year でグループ化し、年単位の分布統計(year_summary.csv)と
年の色相ヒストグラム(year_hue_hist.csv)を作る。集約層_design_260716.md 4節末尾に従う:
  ・スカラー各項目 … その年の動画群の分布 → 平均・標準偏差・中央値・四分位（箱ひげ用）
  ・色相分布 … 動画ごとの7ビンヒストグラムを年内で平均 → 年の色相ヒストグラム

使い方:
  python aggregate_year.py --in out/video_summary.csv --out-summary out/year_summary.csv --out-hue out/year_hue_hist.csv
"""

import argparse, csv, statistics as st
from collections import defaultdict

SCALAR_DIMS = ["mean_s", "mean_v", "colorfulness", "pct_chromatic", "warm_cool_ratio", "hue_entropy",
               "avg_shot_dur", "pct_fast_cut", "mean_L_lab", "mean_C_lab", "color_dynamism",
               "color_dynamism_std", "color_dynamism_max"]
HUE_BINS = ["red", "orange", "yellow", "green", "teal", "blue", "purple"]


def fnum(v):
    v = (v or "").strip() if isinstance(v, str) else v
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def quartiles(vals):
    """中央値・第1四分位・第3四分位(inclusive method)。vals は非空を前提。"""
    q = st.quantiles(vals, n=4, method="inclusive") if len(vals) >= 2 else [vals[0], vals[0], vals[0]]
    return q[0], st.median(vals), q[2]


def main():
    ap = argparse.ArgumentParser(description="video_summary.csvを年単位に集約")
    ap.add_argument("--in", dest="inp", default="out/video_summary.csv")
    ap.add_argument("--out-summary", default="out/year_summary.csv")
    ap.add_argument("--out-hue", default="out/year_hue_hist.csv")
    args = ap.parse_args()

    with open(args.inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_year = defaultdict(list)
    for r in rows:
        y = (r.get("year") or "").strip()
        if y:
            by_year[y].append(r)

    # --- year_summary.csv: dim × (n, mean, std, median, q1, q3) を年ごとに縦持ち ---
    summary_rows = []
    for y in sorted(by_year, key=lambda x: int(x)):
        vids = by_year[y]
        for dim in SCALAR_DIMS:
            vals = [fnum(v.get(dim)) for v in vids]
            vals = [v for v in vals if v is not None]
            if not vals:
                summary_rows.append({"year": y, "dim": dim, "n": 0, "mean": "", "std": "",
                                      "median": "", "q1": "", "q3": ""})
                continue
            mean = st.mean(vals)
            std = st.pstdev(vals) if len(vals) > 1 else 0.0
            q1, med, q3 = quartiles(vals)
            summary_rows.append({"year": y, "dim": dim, "n": len(vals), "mean": mean, "std": std,
                                  "median": med, "q1": q1, "q3": q3})

    with open(args.out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year", "dim", "n", "mean", "std", "median", "q1", "q3"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    # --- year_hue_hist.csv: 動画ごとの7ビンヒストグラムを年内で単純平均 ---
    hue_rows = []
    for y in sorted(by_year, key=lambda x: int(x)):
        vids = by_year[y]
        n_valid = 0
        sums = {b: 0.0 for b in HUE_BINS}
        for v in vids:
            vals = {b: fnum(v.get(f"hue_{b}")) for b in HUE_BINS}
            if any(x is None for x in vals.values()):
                continue  # 暖寒比等と同じく、パレット空だった動画はhue平均から除外
            n_valid += 1
            for b in HUE_BINS:
                sums[b] += vals[b]
        row = {"year": y, "n_videos": n_valid}
        for b in HUE_BINS:
            row[f"hue_{b}"] = (sums[b] / n_valid) if n_valid > 0 else ""
        hue_rows.append(row)

    with open(args.out_hue, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year", "n_videos"] + [f"hue_{b}" for b in HUE_BINS])
        w.writeheader()
        for r in hue_rows:
            w.writerow(r)

    print(f"完了: {len(by_year)} 年分", flush=True)
    print(f"  出力: {args.out_summary}", flush=True)
    print(f"  出力: {args.out_hue}", flush=True)


if __name__ == "__main__":
    main()
