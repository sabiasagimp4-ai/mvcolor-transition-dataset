#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_video_extra.py
==============================================================================
aggregate_video.py で作った video_summary.csv に、追加の5列をマージする。
いずれも新規ダウンロード・新規抽出なしで、既存の抽出出力(shots.csv / timeseries.csv)
だけから計算できるもの。

追加列:
  ・avg_shot_dur   … 平均ショット長(秒) = total_dur_s / n_shots (video_summary.csv内で計算可能)
  ・pct_fast_cut   … 速いカット(shots.csvのis_fast_cut=1)の割合。編集ペースの指標
  ・mean_L_lab     … ドミナントカラー系(パレット由来、share×dur_s加重)のCIELAB明度L*
  ・mean_C_lab     … 同、CIELABクロマC*(知覚的な彩度に相当)
  ・color_dynamism … ショットの並び順で、各ショットの最有力色(weight_rank=1)のCIELAB色差
                     (Delta-E76 = sqrt(dL^2+da^2+db^2)) を連続ショット間で計算し平均したもの。
                     「動画内で色がどれだけ激しく切り替わるか」を表す(値が大きいほど動的)。
  ・color_dynamism_std … 上記ΔEの動画内での標準偏差。「色変化の激しさが動画内で一定か、
                     波があるか」を表す(値が大きいほど、激しい変化と穏やかな変化が混在)。
  ・color_dynamism_max … 上記ΔEの動画内での最大値。「動画内で一番激しかった色の切り替わり」
                     (平均は同じでも、瞬間的に激変する箇所があるMVを区別するための指標)。

使い方:
  python aggregate_video_extra.py --video-summary out/video_summary.csv \
      --shots-dir /d/自自論_video_data/shots_out --extract-dir /d/自自論_video_data/binned_kmeans_out
"""

import argparse, os, csv, math


def fnum(v):
    v = (v or "").strip() if isinstance(v, str) else v
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_shots_fastcut(path):
    """shots.csv を読み、is_fast_cutの割合を返す。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    fast = sum(1 for r in rows if r.get("is_fast_cut", "0").strip() == "1")
    return fast / len(rows)


def read_timeseries_extra(path):
    """timeseries.csv から (mean_L_lab, mean_C_lab, color_dynamism) を計算。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # --- ドミナントカラー系: share×dur_s加重のL*/C* ---
    total_w = 0.0
    sum_L, sum_C = 0.0, 0.0
    for r in rows:
        if (r.get("hue_bin") or "").strip() == "":
            continue  # パレット空のプレースホルダ行はスキップ
        share = fnum(r.get("share"))
        dur_s = fnum(r.get("dur_s"))
        L_lab = fnum(r.get("L_lab"))
        C_lab = fnum(r.get("C_lab"))
        if None in (share, dur_s, L_lab, C_lab):
            continue
        w = share * dur_s
        sum_L += L_lab * w
        sum_C += C_lab * w
        total_w += w
    mean_L_lab = (sum_L / total_w) if total_w > 0 else None
    mean_C_lab = (sum_C / total_w) if total_w > 0 else None

    # --- ショット間の色変動: weight_rank==1(最有力色)をショット順に並べてDelta-E76 ---
    by_shot = {}
    for r in rows:
        sh = r["shot"]
        if (r.get("weight_rank") or "").strip() == "1":
            by_shot[int(sh)] = (fnum(r.get("L_lab")), fnum(r.get("a_lab")), fnum(r.get("b_lab")))
    seq = [by_shot[sh] for sh in sorted(by_shot) if None not in by_shot[sh]]
    if len(seq) >= 2:
        deltas = []
        for (L1, a1, b1), (L2, a2, b2) in zip(seq, seq[1:]):
            deltas.append(math.sqrt((L2 - L1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2))
        color_dynamism = sum(deltas) / len(deltas)
        color_dynamism_max = max(deltas)
        if len(deltas) >= 2:
            var = sum((d - color_dynamism) ** 2 for d in deltas) / len(deltas)
            color_dynamism_std = math.sqrt(var)
        else:
            color_dynamism_std = None
    else:
        color_dynamism = color_dynamism_std = color_dynamism_max = None

    return mean_L_lab, mean_C_lab, color_dynamism, color_dynamism_std, color_dynamism_max


def main():
    ap = argparse.ArgumentParser(description="video_summary.csvに編集ペース/CIELAB/色変動の5列を追加")
    ap.add_argument("--video-summary", default="out/video_summary.csv")
    ap.add_argument("--shots-dir", default="/d/自自論_video_data/shots_out")
    ap.add_argument("--extract-dir", default="/d/自自論_video_data/binned_kmeans_out")
    args = ap.parse_args()

    with open(args.video_summary, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    new_cols = ["avg_shot_dur", "pct_fast_cut", "mean_L_lab", "mean_C_lab", "color_dynamism",
                "color_dynamism_std", "color_dynamism_max"]
    for c in new_cols:
        if c not in fields:
            fields.append(c)

    n_ok, n_skip = 0, 0
    for r in rows:
        vid = r["id"]
        n_shots = fnum(r.get("n_shots"))
        total_dur = fnum(r.get("total_dur_s"))
        r["avg_shot_dur"] = (total_dur / n_shots) if (n_shots and total_dur is not None and n_shots > 0) else ""

        shots_csv = os.path.join(args.shots_dir, vid, f"{vid}_shots.csv")
        ts_csv = os.path.join(args.extract_dir, vid, f"{vid}_binned_kmeans_timeseries.csv")
        if not (os.path.isfile(shots_csv) and os.path.isfile(ts_csv)):
            r["pct_fast_cut"] = r["mean_L_lab"] = r["mean_C_lab"] = r["color_dynamism"] = ""
            r["color_dynamism_std"] = r["color_dynamism_max"] = ""
            n_skip += 1
            continue

        pct_fast = read_shots_fastcut(shots_csv)
        mean_L, mean_C, dyn, dyn_std, dyn_max = read_timeseries_extra(ts_csv)
        r["pct_fast_cut"] = "" if pct_fast is None else pct_fast
        r["mean_L_lab"] = "" if mean_L is None else mean_L
        r["mean_C_lab"] = "" if mean_C is None else mean_C
        r["color_dynamism"] = "" if dyn is None else dyn
        r["color_dynamism_std"] = "" if dyn_std is None else dyn_std
        r["color_dynamism_max"] = "" if dyn_max is None else dyn_max
        n_ok += 1

    with open(args.video_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"完了: {n_ok}件に追加列を記入 / {n_skip}件スキップ(shots.csv等が無い)", flush=True)
    print(f"  更新: {args.video_summary}", flush=True)


if __name__ == "__main__":
    main()
