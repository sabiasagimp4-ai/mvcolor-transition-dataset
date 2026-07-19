#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
year_palette_cemd.py
==============================================================================
CEMD (Color-based Earth Mover's Distance, Skaff, Marchesotti, Csurka & Saunders,
"A Study on Perceptually Coherent Distance Measures for Color Schemes",
19th Color and Imaging Conference, 2011, pp.247-252, Eq.6) で
year_palette.py が作る「年ごとの代表パレット」同士の距離を定量化する。

背景:
  Binned-KMeans原論文(Pocol, Istead & Kaplan, CBMI 2025)ではパレット類似度に
  MICD(Minimum Colour Difference)を使っていたが、著者本人によると、MICDは
  「各色ごとの最短距離」しか見ておらず「色分布全体を動かすコスト」を
  考慮しないという弱点がある。著者らはその後 CEMD (Skaff et al. の
  Earth Mover's Distanceをパレット長の違いに対応させた重み付き版) に
  切り替えた(Graphics Interface 2026 投稿論文、2026-07-17時点で未公開)。

Skaff et al.の原論文を2026-07-19に直接確認した結果、CEMDの定義(Eq.6)は
コスト行列にCIEDE2000(CIE ΔE2000)を使うことがその核心であると判明した
(論文の主張は「Lab空間でのEuclidean距離=CIE76を使うplain EMDより、
CIEDE2000を組み込んだ方が知覚的に優れる」)。旧版はMICDとの直接比較のため
CIE76で実装していたが(Skaff et al.の分類でいう"plain EMD"相当)、
2026-07-19にCIEDE2000版に切り替えて原論文に忠実にした。
CIEDE2000本体はscikit-imageの`skimage.color.deltaE_ciede2000`を使用
(自前実装は一度書いてSharma, Wu & Dalal 2005の公開テストケース・
colormathライブラリと突き合わせて正しさを確認済みだったが、保守された
標準実装がある以上それに一本化した)。
CIE76版とCIEDE2000版を比較検証したところ結論(非有意)自体は変わらない
ことを確認済み(r=0.371→0.328, p=0.107→0.157)。CIE76版はロバスト性
チェックとして引き続き計算・出力する。

【2026-07-19修正】年ごとの代表パレット作成(weighted_kmeans_palette_lab_binned)
がBinned-KMeansのビン保護を保つよう修正された(year_palette.py参照)。
以前は色相ビンを無視した単純加重KMeansで年パレットを作っていたため、
画面占有面積の大きい肌色・暗色が代表パレットのクラスタを占有し、
CEMD計算の元になるパレットそのものが偏っていた可能性がある。
修正後の数値で再計算し、結論が変わるか要確認。

使い方:
  python year_palette_cemd.py --manifest ../videos.csv --extract-dir /d/自自論_video_data/binned_kmeans_out --k 8 --out out/year_palette_cemd.csv
"""

import argparse, os, csv
import numpy as np
import ot
from scipy import stats as scipy_stats
from skimage.color import deltaE_ciede2000, deltaE_cie76

from year_palette import fnum, read_manifest, collect_year_colors
from year_palette_micd import weighted_kmeans_palette_lab_binned


def ciede2000_matrix(labA, labB):
    """CIEDE2000色差(scikit-image実装、NxMブロードキャストで一括計算)。labA:(N,3), labB:(M,3) -> (N,M)コスト行列。"""
    return deltaE_ciede2000(labA[:, None, :], labB[None, :, :])


def cie76_matrix(labA, labB):
    """CIE76色差(scikit-image実装。自前実装と数値が完全一致することを確認済み)。"""
    return deltaE_cie76(labA[:, None, :], labB[None, :, :])


def cemd(labA, wA, labB, wB, metric="ciede2000"):
    """重み付きCEMD(Skaff et al. Eq.6)。metric: "ciede2000"(既定、原論文どおり) or "cie76"(旧版・ロバスト性チェック用)。"""
    M = ciede2000_matrix(labA, labB) if metric == "ciede2000" else cie76_matrix(labA, labB)
    wA = np.asarray(wA, dtype=np.float64)
    wB = np.asarray(wB, dtype=np.float64)
    wA = wA / wA.sum()
    wB = wB / wB.sum()
    return ot.emd2(wA, wB, M)


def regression(years, values):
    """MICD分析と同じ流儀: 値が空でない点だけで線形回帰し(slope, r, p)を返す。"""
    ys = [y for y, v in zip(years, values) if v != ""]
    vs = [v for v in values if v != ""]
    if len(ys) < 3:
        return None
    slope, intercept, r, p, se = scipy_stats.linregress(ys, vs)
    return {"n": len(ys), "slope_per_year": slope, "r": r, "p_value": p}


def main():
    ap = argparse.ArgumentParser(description="年ごとの代表パレットのCEMD(色分布の移動距離、Skaff et al. CIEDE2000版)を計算")
    ap.add_argument("--manifest", default="../videos.csv")
    ap.add_argument("--extract-dir", default="/d/自自論_video_data/binned_kmeans_out")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="out/year_palette_cemd.csv")
    ap.add_argument("--stats-out", default="out/year_palette_cemd_stats.csv")
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
        d_prev = cemd(prev_lab, prev_w, lab, w, metric="ciede2000") if prev_lab is not None else ""
        d_base = cemd(labA0, wA0, lab, w, metric="ciede2000")
        d_base_cie76 = cemd(labA0, wA0, lab, w, metric="cie76")
        rows.append({
            "year": year,
            "cemd_vs_prev_year": d_prev,
            "cemd_vs_base_year": d_base,
            "cemd_vs_base_year_cie76": d_base_cie76,
        })
        prev_lab, prev_w = lab, w

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = ["year", "cemd_vs_prev_year", "cemd_vs_base_year", "cemd_vs_base_year_cie76"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"基準年: {base_year}", flush=True)
    print("(CEMDはCIEDE2000版が既定。cemd_vs_base_year_cie76は旧CIE76版・ロバスト性チェック用に併記)", flush=True)
    for r in rows:
        pv = f"{r['cemd_vs_prev_year']:.3f}" if r['cemd_vs_prev_year'] != "" else "  -  "
        print(f"  {r['year']}  前年比CEMD={pv}  {base_year}年比CEMD={r['cemd_vs_base_year']:.3f}"
              f"  (CIE76版={r['cemd_vs_base_year_cie76']:.3f})", flush=True)
    print(f"出力: {args.out}", flush=True)

    # 回帰統計: vs_base_yearは基準年自身の0点(自明な点)を除外して計算する
    years_arr = [r["year"] for r in rows]
    vs_prev_vals = [r["cemd_vs_prev_year"] for r in rows]
    vs_base_vals_excl_self = [r["cemd_vs_base_year"] if r["year"] != base_year else "" for r in rows]
    vs_base_cie76_excl_self = [r["cemd_vs_base_year_cie76"] if r["year"] != base_year else "" for r in rows]

    stat_prev = regression(years_arr, vs_prev_vals)
    stat_base = regression(years_arr, vs_base_vals_excl_self)
    stat_base_cie76 = regression(years_arr, vs_base_cie76_excl_self)

    with open(args.stats_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "n", "slope_per_year", "r", "p_value"])
        w.writeheader()
        if stat_prev:
            w.writerow({"metric": "cemd_vs_prev_year", **stat_prev})
        if stat_base:
            w.writerow({"metric": "cemd_vs_base_year", **stat_base})
        if stat_base_cie76:
            w.writerow({"metric": "cemd_vs_base_year_cie76", **stat_base_cie76})

    print("\n--- 回帰統計 ---", flush=True)
    for name, s in [("前年比CEMD(CIEDE2000)", stat_prev),
                    (f"{base_year}年比CEMD(CIEDE2000、基準年除く)", stat_base),
                    (f"{base_year}年比CEMD(CIE76版・ロバスト性チェック、基準年除く)", stat_base_cie76)]:
        if s:
            sig = "有意(p<0.05)" if s["p_value"] < 0.05 else "非有意"
            print(f"  {name}: n={s['n']}  slope={s['slope_per_year']:.4f}/年  r={s['r']:.3f}  p={s['p_value']:.4f}  → {sig}", flush=True)
    print(f"統計出力: {args.stats_out}", flush=True)


if __name__ == "__main__":
    main()
