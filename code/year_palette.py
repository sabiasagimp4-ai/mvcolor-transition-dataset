#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
year_palette.py
==============================================================================
先行研究(Istead, Pocol, Kaplan et al., CVMP 2022「The Colour of Horror」
https://doi.org/10.1145/3565516.3565523)の手法にならい、動画ごとの統計を
平均するのではなく「その年の全動画から色をプールし、加重KMeansで再クラスタ
リングして年を代表するパレットを作る」方式で年ごとの色を可視化する。

既存の hue_histogram(7ビンの抽象的な割合)を補う位置づけ:
色相の"割合"だけでなく、実際の色味(同じオレンジでも年によって彩度・明度が
違うか等)まで見えるようにする。

各ショットのパレット色(hue_bin, rgb, share)を weight=share×dur_s で年ごとに
プールし、sklearn KMeansのsample_weightで加重クラスタリング(k色)。
出力クラスタは虹順(色相角)でソートして横並びのスウォッチにする。

使い方:
  python year_palette.py --manifest ../videos.csv --extract-dir /d/自自論_video_data/binned_kmeans_out --k 8 --out out/year_palette.png
"""

import argparse, os, csv, math
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont


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


def collect_year_colors(manifest, extract_dir):
    """year -> [(rgb(3,), weight), ...] を集める。"""
    by_year = defaultdict(list)
    for m in manifest:
        vid = m["id"]
        year = m.get("year")
        if not year:
            continue
        base = os.path.splitext((m.get("video_file") or f"{vid}.mp4"))[0]
        ts_path = os.path.join(extract_dir, base, f"{base}_binned_kmeans_timeseries.csv")
        if not os.path.isfile(ts_path):
            continue
        with open(ts_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("hue_bin") or "").strip() == "":
                    continue
                share = fnum(r.get("share"))
                dur_s = fnum(r.get("dur_s"))
                rr, gg, bb = fnum(r.get("r")), fnum(r.get("g")), fnum(r.get("b"))
                if None in (share, dur_s, rr, gg, bb):
                    continue
                w = share * dur_s
                if w <= 0:
                    continue
                by_year[int(year)].append((np.array([rr, gg, bb]), w))
    return by_year


def rgb_to_hue(rgb):
    r, g, b = [c / 255.0 for c in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d == 0:
        return 0.0
    if mx == r:
        h = ((g - b) / d) % 6
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return (h * 60) % 360


def weighted_kmeans_palette(colors, weights, k):
    """加重KMeansでk色に集約し、(rgb, weight_share)を色相角順で返す。"""
    X = np.stack(colors)
    w = np.array(weights)
    k_eff = min(k, X.shape[0])
    if k_eff < 1:
        return []
    km = KMeans(n_clusters=k_eff, random_state=0, n_init=10)
    km.fit(X, sample_weight=w)
    labels = km.labels_
    centers = km.cluster_centers_
    total_w = w.sum()
    out = []
    for i in range(k_eff):
        mask = labels == i
        cw = w[mask].sum()
        out.append((centers[i], cw / total_w))
    out.sort(key=lambda t: rgb_to_hue(t[0]))
    return out


def main():
    ap = argparse.ArgumentParser(description="年ごとに色をプールし加重KMeansで代表パレットを作る(先行研究の手法)")
    ap.add_argument("--manifest", default="../videos.csv")
    ap.add_argument("--extract-dir", default="/d/自自論_video_data/binned_kmeans_out")
    ap.add_argument("--k", type=int, default=8, help="年あたりの代表色数 既定8")
    ap.add_argument("--out", default="out/year_palette.png")
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)
    by_year = collect_year_colors(manifest, args.extract_dir)

    years = sorted(by_year)
    row_h, width, left_margin = 60, 900, 60
    img_h = len(years) * row_h + 20
    img_w = left_margin + width + 10
    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("meiryo.ttc", 20)
    except Exception:
        font = ImageFont.load_default()

    for i, year in enumerate(years):
        colors = [c for c, _ in by_year[year]]
        weights = [w for _, w in by_year[year]]
        palette = weighted_kmeans_palette(colors, weights, args.k)
        y0 = i * row_h
        x = float(left_margin)
        for rgb, share in palette:
            seg_w = width * share
            rgb_t = tuple(int(max(0, min(255, round(c)))) for c in rgb)
            draw.rectangle([int(x), y0, int(x + seg_w) - 1, y0 + row_h - 4], fill=rgb_t)
            x += seg_w
        draw.text((4, y0 + 15), str(year), fill=(0, 0, 0), font=font)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img.save(args.out)
    print(f"完了: {len(years)}年分 / k={args.k}", flush=True)
    print(f"  出力: {args.out}  ({img_w}x{img_h}px)", flush=True)


if __name__ == "__main__":
    main()
