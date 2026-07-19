#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_quality_confound.py
==============================================================================
「彩度・色特徴の年トレンドは、古い動画の画質(解像度・ビットレート)が低いこと
の副産物ではないか」という懸念(集約層_design_260716.md 9節記載の限界)を検証する。

1. 全動画をffprobeで実測し、解像度(height)・ビットレートを取得
2. 各色指標について、年 単独の単回帰と、年+ビットレートの重回帰(OLS)を比較し、
   ビットレートを制御しても年の効果が残るかを確認

使い方:
  python check_quality_confound.py --manifest ../videos.csv --videos-dir /d/自自論_video_data/videos \
      --video-summary out/video_summary.csv --out out/video_quality.csv
"""

import argparse, os, csv, json, subprocess
import numpy as np
from scipy import stats


def fnum(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def probe_one(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,bit_rate",
         "-show_entries", "format=size,bit_rate",
         "-of", "json", path],
        capture_output=True, text=True, timeout=15)
    info = json.loads(r.stdout or "{}")
    st = info.get("streams", [{}])[0]
    fmt = info.get("format", {})
    return {
        "width": st.get("width"), "height": st.get("height"),
        "bit_rate": st.get("bit_rate") or fmt.get("bit_rate"),
        "file_size": fmt.get("size"),
    }


def main():
    ap = argparse.ArgumentParser(description="解像度・画質と色指標の交絡をチェック")
    ap.add_argument("--manifest", default="../videos.csv")
    ap.add_argument("--videos-dir", default="D:/自自論_video_data/videos")
    ap.add_argument("--video-summary", default="out/video_summary.csv")
    ap.add_argument("--out", default="out/video_quality.csv")
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8-sig") as f:
        manifest = list(csv.DictReader(f))

    quality = []
    for m in manifest:
        vid = m["id"]
        path = os.path.join(args.videos_dir, f"{vid}.mp4")
        if not os.path.isfile(path):
            continue
        try:
            q = probe_one(path)
        except Exception:
            q = {"width": "", "height": "", "bit_rate": "", "file_size": ""}
        q["id"] = vid
        quality.append(q)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "width", "height", "bit_rate", "file_size"])
        w.writeheader()
        for q in quality:
            w.writerow(q)
    print(f"実測: {len(quality)}本 -> {args.out}", flush=True)

    qmap = {q["id"]: q for q in quality}
    with open(args.video_summary, newline="", encoding="utf-8") as f:
        srows = {r["id"]: r for r in csv.DictReader(f)}
    with open(args.manifest, newline="", encoding="utf-8-sig") as f:
        yrows = {r["id"]: r for r in csv.DictReader(f)}

    ids = [i for i in qmap if i in srows and fnum(qmap[i].get("height")) is not None]
    years = np.array([fnum(yrows[i]["year"]) for i in ids])
    heights = np.array([fnum(qmap[i]["height"]) for i in ids])
    bitrates = np.array([fnum(qmap[i]["bit_rate"]) for i in ids])
    print(f"\n解像度(height) vs year: {stats.linregress(years, heights)}")
    print(f"ビットレート vs year: {stats.linregress(years, bitrates)}")

    dims = ["pct_chromatic", "warm_cool_ratio", "mean_L_lab", "mean_C_lab", "pct_fast_cut", "color_dynamism"]
    print(f"\n{'dim':16s} {'year単回帰r':>12s} {'年+bitrate重回帰: year係数':>26s} {'年のp値':>10s}")
    for dim in dims:
        idd = [i for i in ids if fnum(srows[i].get(dim)) is not None and fnum(qmap[i].get("bit_rate")) is not None]
        y = np.array([fnum(yrows[i]["year"]) for i in idd])
        br = np.array([fnum(qmap[i]["bit_rate"]) for i in idd]) / 1e6
        val = np.array([fnum(srows[i][dim]) for i in idd])
        r_single = stats.linregress(y, val).rvalue
        X = np.column_stack([np.ones(len(idd)), y, br])
        beta, *_ = np.linalg.lstsq(X, val, rcond=None)
        resid = val - X @ beta
        dof = len(idd) - X.shape[1]
        sigma2 = (resid @ resid) / dof
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        p_year = 2 * (1 - stats.t.cdf(abs(beta[1] / se[1]), dof))
        print(f"{dim:16s} {r_single:12.3f} {beta[1]:26.5f} {p_year:10.4f}")


if __name__ == "__main__":
    main()
