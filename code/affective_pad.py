#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
affective_pad.py
==============================================================================
Schindler & Rauber (2015)"An Audio-Visual Approach to Music Genre
Classification through Affective Color Features" のPleasure-Arousal-Dominance
(PAD)感情モデル(Valdez & Mehrabian, 1994 の実験式に基づく、原論文Eq.1-3)を、
本研究のデータに適用する追加実験。

  Pleasure  = 0.69*B + 0.22*S
  Arousal   = -0.31*B + 0.60*S
  Dominance = 0.76*B + 0.32*S

B(輝度)・S(彩度)は原論文ではIHLS(Improved HLS)色空間から計算されているが、
本研究の既存データ(mean_v, mean_s)はHSV色空間の値(全画素平均、dur_s加重済み)。
IHLSとHSVは彩度・明度の定義が微妙に異なる(IHLSは無彩色画素の彩度を頑健に0へ
寄せる設計)ため、本実験は「原論文の式をそのまま流用した近似」であり、厳密な
再現ではないことに注意(方法の限界として明記する)。

PADは線形式なので、動画1本分に集約済みのmean_s/mean_v(すでにdur_s加重平均済み)
にそのまま適用しても結果は数学的に同一(加重平均の線形性)。そのため新規のショット
単位の再集計は不要で、既存のvideo_summary.csvから直接計算できる。

使い方:
  python affective_pad.py --video-summary out_transnetv2/video_summary.csv --out out_transnetv2/affective_pad.csv
"""

import argparse, csv, os
from scipy import stats as scipy_stats


def fnum(v):
    v = (v or "").strip() if isinstance(v, str) else v
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description="Pleasure/Arousal/Dominanceをvideo_summary.csvから計算し年トレンドを検定")
    ap.add_argument("--video-summary", default="out_transnetv2/video_summary.csv")
    ap.add_argument("--out", default="out_transnetv2/affective_pad.csv")
    ap.add_argument("--stats-out", default="out_transnetv2/affective_pad_stats.csv")
    args = ap.parse_args()

    with open(args.video_summary, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for r in rows:
        S, B = fnum(r.get("mean_s")), fnum(r.get("mean_v"))
        year = fnum(r.get("year"))
        if S is None or B is None or year is None:
            continue
        pleasure = 0.69 * B + 0.22 * S
        arousal = -0.31 * B + 0.60 * S
        dominance = 0.76 * B + 0.32 * S
        out_rows.append({"id": r["id"], "year": int(year), "mean_s": S, "mean_v": B,
                          "pleasure": pleasure, "arousal": arousal, "dominance": dominance})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "year", "mean_s", "mean_v", "pleasure", "arousal", "dominance"])
        w.writeheader()
        w.writerows(out_rows)

    stats_rows = []
    print(f"n={len(out_rows)}\n")
    for dim in ("pleasure", "arousal", "dominance"):
        years = [r["year"] for r in out_rows]
        vals = [r[dim] for r in out_rows]
        res = scipy_stats.linregress(years, vals)
        sig = "有意" if res.pvalue < 0.05 else "非有意"
        print(f"{dim:10s}  slope={res.slope:+.5f}/年  r={res.rvalue:+.3f}  p={res.pvalue:.4g}  ({sig})")
        stats_rows.append({"dim": dim, "n": len(out_rows), "slope_per_year": res.slope,
                            "r": res.rvalue, "p_value": res.pvalue,
                            "significant_p<0.05": res.pvalue < 0.05})

    with open(args.stats_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dim", "n", "slope_per_year", "r", "p_value", "significant_p<0.05"])
        w.writeheader()
        w.writerows(stats_rows)

    print(f"\n出力: {args.out}")
    print(f"統計: {args.stats_out}")


if __name__ == "__main__":
    main()
