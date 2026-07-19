#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_video.py
==============================================================================
videos.csv（マニフェスト）と、動画ごとの shot_dominant_color_binned_kmeans.py
出力（<id>_binned_kmeans_timeseries.csv）を読み、動画1本=1行の video_summary.csv
を作る。集約層_design_260716.md 4節の計算方法にそのまま従う。

2系統・重み違い（設計書4節）:
  ・画面全体レベル系(frame_mean_s, frame_mean_v, colorfulness, pct_chromatic)
    … ショット単位の1値を dur_s で加重平均。標準偏差も出す。
  ・ドミナントカラー系(暖色/寒色比, 色相分布, パレット多様性)
    … パレット色ごとに share×dur_s で加重。

エラー処理(設計書7節):
  ・manifestにあるが抽出出力(timeseries.csv)が無い行 → 警告してスキップ。
  ・ショットが1つも読めない(=抽出が実質失敗) → 警告してスキップ。
  ・全ショットでパレットが空(有彩色ゼロ)の動画 → 画面全体レベル系の値は残し、
    ドミナントカラー系(暖色/寒色比・色相分布・エントロピー)のみ空欄にして警告ログに残す
    (完全に無彩色寄りのMVも「色のトレンド」データとして意味があるため、レベル系まで
    落とすと探索的研究の趣旨に反する。設計書7節の「落とさない」の実務的解釈)。

使い方:
  python aggregate_video.py --manifest ../videos.csv --extract-dir /d/.../binned_kmeans_out --out out/video_summary.csv
"""

import argparse, os, sys, csv, math
from collections import defaultdict

HUE_BINS = ["red", "orange", "yellow", "green", "teal", "blue", "purple"]
WARM = {"red", "orange", "yellow"}
COOL = {"teal", "blue", "purple"}


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_timeseries(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v):
    v = (v or "").strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def weighted_mean_std(pairs):
    """pairs: [(value, weight), ...]。値がNoneの要素は除外。戻り値 (mean, std) or (None, None)。"""
    pairs = [(v, w) for v, w in pairs if v is not None and w is not None and w > 0]
    tw = sum(w for _, w in pairs)
    if tw <= 0:
        return None, None
    mean = sum(v * w for v, w in pairs) / tw
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / tw
    return mean, math.sqrt(var)


def aggregate_one(rows, vid, log):
    """1動画のtimeseries行 -> summary辞書。失敗時はNoneを返す。"""
    # ショット単位にグルーピング(パレット色ごとに複数行あるため、レベル系は shot につき1回だけ数える)
    shots = {}
    for r in rows:
        sh = r["shot"]
        if sh not in shots:
            shots[sh] = {
                "dur_s": fnum(r["dur_s"]),
                "frame_mean_s": fnum(r.get("frame_mean_s")),
                "frame_mean_v": fnum(r.get("frame_mean_v")),
                "pct_chromatic": fnum(r.get("pct_chromatic")),
                "colorfulness": fnum(r.get("colorfulness")),
            }
    if not shots:
        log.append(f"[SKIP] {vid}: ショットが0件(抽出失敗の可能性)")
        return None

    mean_s, mean_s_std = weighted_mean_std([(s["frame_mean_s"], s["dur_s"]) for s in shots.values()])
    mean_v, mean_v_std = weighted_mean_std([(s["frame_mean_v"], s["dur_s"]) for s in shots.values()])
    colorfulness, colorfulness_std = weighted_mean_std([(s["colorfulness"], s["dur_s"]) for s in shots.values()])
    pct_chromatic, pct_chromatic_std = weighted_mean_std([(s["pct_chromatic"], s["dur_s"]) for s in shots.values()])

    if mean_s is None:
        log.append(f"[SKIP] {vid}: 全フレーム由来の値が1件も取れない(抽出失敗の可能性)")
        return None

    # ドミナントカラー系: パレット色ごとに share×dur_s で加重
    hue_weight = defaultdict(float)
    total_palette_w = 0.0
    for r in rows:
        bin_name = (r.get("hue_bin") or "").strip()
        if bin_name not in HUE_BINS:
            continue  # パレット空プレースホルダ行 or 不正値
        share = fnum(r.get("share"))
        dur_s = fnum(r.get("dur_s"))
        if share is None or dur_s is None:
            continue
        w = share * dur_s
        hue_weight[bin_name] += w
        total_palette_w += w

    warm_cool_ratio = None
    hue_entropy = None
    hue_hist = {b: None for b in HUE_BINS}
    if total_palette_w > 0:
        hue_hist = {b: hue_weight.get(b, 0.0) / total_palette_w for b in HUE_BINS}
        warm_w = sum(hue_weight.get(b, 0.0) for b in WARM)
        cool_w = sum(hue_weight.get(b, 0.0) for b in COOL)
        if (warm_w + cool_w) > 0:
            warm_cool_ratio = warm_w / (warm_w + cool_w)
        # Shannon entropy (log2), 0分は無視
        hue_entropy = -sum(p * math.log2(p) for p in hue_hist.values() if p and p > 0)
    else:
        log.append(f"[WARN] {vid}: 全ショットで有彩色パレットが空。"
                    f"レベル系(平均彩度等)は採用、暖寒比・色相分布・多様性は空欄にする。")

    out = {
        "id": vid, "n_shots": len(shots),
        "total_dur_s": sum(s["dur_s"] for s in shots.values() if s["dur_s"] is not None),
        "mean_s": mean_s, "mean_s_std": mean_s_std,
        "mean_v": mean_v, "mean_v_std": mean_v_std,
        "colorfulness": colorfulness, "colorfulness_std": colorfulness_std,
        "pct_chromatic": pct_chromatic, "pct_chromatic_std": pct_chromatic_std,
        "warm_cool_ratio": warm_cool_ratio, "hue_entropy": hue_entropy,
    }
    for b in HUE_BINS:
        out[f"hue_{b}"] = hue_hist[b]
    return out


def main():
    ap = argparse.ArgumentParser(description="動画ごとのtimeseries.csvをvideo_summary.csvに集約")
    ap.add_argument("--manifest", default="../videos.csv")
    ap.add_argument("--extract-dir", default="/d/自自論_video_data/binned_kmeans_out",
                     help="shot_dominant_color_binned_kmeans.py の --outdir と同じ場所")
    ap.add_argument("--out", default="out/video_summary.csv")
    ap.add_argument("--log", default="out/aggregate_video.log")
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)
    log = []
    results = []

    for m in manifest:
        vid = (m.get("id") or "").strip()
        if not vid:
            continue
        base = os.path.splitext((m.get("video_file") or f"{vid}.mp4"))[0]
        ts_path = os.path.join(args.extract_dir, base, f"{base}_binned_kmeans_timeseries.csv")
        if not os.path.isfile(ts_path):
            log.append(f"[SKIP] {vid}: 抽出出力が見つからない ({ts_path})")
            continue
        rows = read_timeseries(ts_path)
        agg = aggregate_one(rows, vid, log)
        if agg is None:
            continue
        agg["year"] = m.get("year", "")
        agg["rank"] = m.get("rank", "")
        agg["artist"] = m.get("artist", "")
        agg["title"] = m.get("title", "")
        results.append(agg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = (["id", "year", "rank", "artist", "title", "n_shots", "total_dur_s",
               "mean_s", "mean_s_std", "mean_v", "mean_v_std",
               "colorfulness", "colorfulness_std", "pct_chromatic", "pct_chromatic_std",
               "warm_cool_ratio", "hue_entropy"] + [f"hue_{b}" for b in HUE_BINS])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})

    with open(args.log, "w", encoding="utf-8") as f:
        f.write("\n".join(log) + ("\n" if log else ""))

    print(f"完了: {len(results)} 動画 / manifest {len(manifest)} 行", flush=True)
    print(f"  出力: {args.out}", flush=True)
    print(f"  ログ({len(log)}件): {args.log}", flush=True)


if __name__ == "__main__":
    main()
