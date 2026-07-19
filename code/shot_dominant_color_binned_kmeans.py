#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shot_dominant_color_binned_kmeans.py
==============================================================================
Binned-KMeans（Pocol, Istead & Kaplan, CBMI 2025, DOI: 10.1109/CBMI66578.2025.11339288）
の手法を、shot_cut_detection.py が出力したショット境界CSVに沿って各ショットへ適用し、
ショットごとのドミナントカラー・パレットを時系列データ（CSV/JSON）として出力する。

Binned-KMeansの骨格（原論文どおり）:
  1. HSV空間上に色相ビン(赤/橙/黄/緑/青緑/青/紫)をあらかじめ定義
  2. 画素を色相ビンに振り分け、彩度の低い(無彩色/黒・白・灰・ベージュ)画素は除外
  3. ビンごとに独立してKMeansクラスタリングし、ビンごとの小さなk色パレットを作る
  4. 彩度×明度による重み付け式 w'(x,c) = (100*c/t)^(sat(x)*val(x)*tau) で
     各色の「目立ち度」を評価する(黒に近い高彩度画素の過大評価を防ぐ)
  5. 各ビン内は明度→彩度の順でソートし、ビンは常に虹順(赤→橙→黄→緑→青緑→青→紫)で連結

このスクリプトでは、原論文が動画1本全体に対して行っていた処理を、ショット単位に適用する。
ショット内のフレームサンプリングは、あなたの指示により前後トリムなしの等間隔サンプリング
（shot_color_analysis_ffmpeg_cielab.py の analyze_shots() と同じロジック）を用いる。

抽出した各色について、shot_color_analysis_ffmpeg_cielab.py に実装されていた
Ou, Luo, Woodcock & Wright (2004) の色感情モデル（Heat/Activity/Weightのz得点、
式・係数は変更なし）もあわせて算出し、パレット全体をweightで加重平均した
ショット単位の代表値も出力する（詳細は shot_dominant_color_binned_kmeans_仕様書.md 5.6節）。

使い方:
  python shot_dominant_color_binned_kmeans.py INPUT.mp4 --shots-csv out/INPUT/INPUT_shots.csv

  ※ shots.csv の fps列からサンプリングfpsを自動取得するため、通常 --fps の指定は不要。
    fps列が無いCSV（旧形式・手作りCSV等）の場合のみ --fps を明示的に指定すること
    (CSVのstart_frame/end_frameは、そのfpsでサンプリングしたフレーム列上のインデックスのため)。

必要なもの:
  ・ffmpeg / ffprobe (コマンド)
  ・pip install numpy Pillow scikit-learn
"""

import argparse, os, sys, csv, json, math, subprocess, tempfile, shutil
import numpy as np
from sklearn.cluster import KMeans
try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow が必要:  pip install numpy Pillow scikit-learn")


# ============================================================ ffmpeg helpers
def need_ffmpeg():
    for cmd in ("ffmpeg", "ffprobe"):
        if shutil.which(cmd) is None:
            sys.exit(f"{cmd} が見つかりません。ffmpeg をインストールしてください。\n"
                     f"  macOS: brew install ffmpeg  /  Ubuntu: sudo apt install ffmpeg\n"
                     f"  Windows: https://ffmpeg.org/download.html")

def extract_frames(path, sample_fps, width, tmpdir):
    """ffmpeg で sample_fps 枚/秒・幅 width に縮小した連番画像を書き出す。
    PPM(無圧縮)で出す軽量化版: 画素値はPNGと完全に同一(可逆)だが、
    zlib圧縮/展開が無い分ffmpeg書き出し・PIL読み込みの両方が速い。"""
    pattern = os.path.join(tmpdir, "f_%06d.ppm")
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-vf", f"fps={sample_fps},scale={width}:-1:flags=area",
           "-vsync", "vfr", pattern]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"ffmpeg 失敗:\n{r.stderr}")
    files = sorted(os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".ppm"))
    return files

def load_all(files):
    arrs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        arrs.append(np.asarray(im))
    return arrs


# ============================================================ HSV変換 (ベクトル化)
def rgb_to_hsv_arr(rgb):
    """rgb: (...,3) 0-255 array. 戻り値: h[0,360), s[0,1], v[0,1] の3つの配列(同shape)。"""
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1); mn = np.min(rgb, axis=-1); d = mx - mn
    v = mx
    s = np.where(mx > 0, d / (mx + 1e-12), 0.0)
    h = np.zeros_like(v)
    nz = d > 1e-12
    idx = (mx == r) & nz; h[idx] = ((g[idx] - b[idx]) / d[idx]) % 6
    idx = (mx == g) & nz; h[idx] = (b[idx] - r[idx]) / d[idx] + 2
    idx = (mx == b) & nz; h[idx] = (r[idx] - g[idx]) / d[idx] + 4
    h = (h * 60) % 360
    return h, s, v

def hex_of(rgb):
    return "#{:02x}{:02x}{:02x}".format(*[int(max(0, min(255, round(c)))) for c in rgb])


# ============================================================ CIELAB色変換 (sRGB D65)
# shot_color_analysis_ffmpeg_cielab.py の rgb_to_lab_arr/_ch/rgb_to_lab と同一の変換式
# (係数は変更していない。Ou et al.検証をこのスクリプト単体でも行えるよう移植)。
_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]])
_Xn, _Yn, _Zn = 0.95047, 1.00000, 1.08883

def _inv_gamma(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _f_lab(t):
    return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)

def rgb_to_lab_arr(rgb):
    """rgb: (...,3) 0-255 array. 戻り値: 同shapeの(L*,a*,b*)。"""
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = _inv_gamma(rgb)
    xyz = lin @ _M_RGB2XYZ.T
    x, y, z = xyz[..., 0] / _Xn, xyz[..., 1] / _Yn, xyz[..., 2] / _Zn
    fx, fy, fz = _f_lab(x), _f_lab(y), _f_lab(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)

def _ch(a, b):
    """a*,b* から C*(クロマ), h(CIELAB色相角[deg]) を算出。"""
    C = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a))
    if h < 0: h += 360
    return C, h

def rgb_to_lab(rgb):
    """単一のRGB(0-255)から (L*, a*, b*, C*, h[deg]) を返す。"""
    L, a, b = rgb_to_lab_arr(np.array(rgb, dtype=np.float64))
    C, h = _ch(float(a), float(b))
    return float(L), float(a), float(b), C, h


# ============================================================ Ou, Luo, Woodcock & Wright (2004) 色感情モデル
# shot_color_analysis_ffmpeg_cielab.py の ou_wc_score/ou_activity_score/ou_weight_score/
# ou_warm_pct と同一の式・係数(値を変更していない。データソースがBinned-KMeansの
# パレット色に変わっただけで、モデル自体は論文の再現)。
def ou_wc_score(C, h):
    """Ou et al.(2004) Part I Eq.2/Eq.8: colour heat(暖寒) z得点。R^2=0.74。
    50°が最も暖と判定された基準色相角。"""
    return -0.5 + 0.02 * (C ** 1.07) * math.cos(math.radians(h - 50))

def ou_activity_score(L, a, b):
    """Ou et al.(2004) Part I Eq.6: colour activity(active-passive) z得点。R^2=0.93。
    基準点(くすんだ黄, L*=50,a*=3,b*=17)からの距離(b*方向は係数1.4で減衰)。
    正=active寄り、負=passive寄り。"""
    return -2.1 + 0.06 * math.sqrt((L - 50) ** 2 + (a - 3) ** 2 + ((b - 17) / 1.4) ** 2)

def ou_weight_score(L, h):
    """Ou et al.(2004) Part I Eq.7: colour weight(heavy-light) z得点。R^2=0.73。
    正=heavy(重い/硬い)寄り、負=light(軽い/柔らかい)寄り。"""
    return -1.8 + 0.04 * (100 - L) + 0.45 * math.cos(math.radians(h - 100))

def ou_warm_pct(z):
    """Thurstone Case Vのz得点を『観察者の何%が暖色(重い/活発など正極)と判断するか』の割合に変換。"""
    return 0.5 * (1 + math.erf(z / math.sqrt(2))) * 100


# ============================================================ 色相ビン定義 (Binned-KMeans)
# 原論文はビン境界の具体的な数値までは開示していないため、赤・橙・黄・緑・青緑・青・紫の
# 7ビンで色相環[0,360)を分割する一般的な区切りを用いる(赤は0°をまたぐ)。
BIN_ORDER = ["red", "orange", "yellow", "green", "teal", "blue", "purple"]
BIN_LABEL_JA = {"red": "赤", "orange": "橙", "yellow": "黄", "green": "緑",
                 "teal": "青緑", "blue": "青", "purple": "紫"}
_BIN_EDGES = [  # (name, lo, hi) 。redのみ 345-360 と 0-15 の2区間にまたがる
    ("orange", 15, 45), ("yellow", 45, 70), ("green", 70, 170),
    ("teal", 170, 200), ("blue", 200, 260), ("purple", 260, 345),
]

def assign_hue_bins(h):
    """h: (...,) array[0,360) -> 同shapeのdtype=object配列(色相ビン名、非該当はNone)。"""
    names = np.full(h.shape, None, dtype=object)
    names[(h >= 345) | (h < 15)] = "red"
    for name, lo, hi in _BIN_EDGES:
        names[(h >= lo) & (h < hi)] = name
    return names


# ============================================================ 全フレーム由来レベル指標
# (集約層_design_260716.md 1.1節: パレット(有彩色のみ)とは別に、除外なしの全画素で測る)
def frame_level_stats(pixels_rgb, s_min=0.15, v_min=0.08):
    """pixels_rgb: (N,3) 0-255 array (除外なし・全画素)。
    戻り値: frame_mean_s, frame_mean_v, pct_chromatic, colorfulness。
    空配列なら全てNone。"""
    pixels_rgb = np.asarray(pixels_rgb, dtype=np.float64)
    if pixels_rgb.shape[0] == 0:
        return {"frame_mean_s": None, "frame_mean_v": None,
                "pct_chromatic": None, "colorfulness": None}
    h, s, v = rgb_to_hsv_arr(pixels_rgb)
    frame_mean_s = float(np.mean(s))
    frame_mean_v = float(np.mean(v))
    # palette側(binned_kmeans_palette)と同じ閾値で「有彩色」を判定する(2系統を比較可能にするため)
    pct_chromatic = float(np.mean((s >= s_min) & (v >= v_min)))
    # Hasler & Süsstrunk (2003): M = sqrt(sigma_rg^2+sigma_yb^2) + 0.3*sqrt(mu_rg^2+mu_yb^2)
    r, g, b = pixels_rgb[..., 0], pixels_rgb[..., 1], pixels_rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(math.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
                          + 0.3 * math.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))
    return {"frame_mean_s": frame_mean_s, "frame_mean_v": frame_mean_v,
            "pct_chromatic": pct_chromatic, "colorfulness": colorfulness}


# ============================================================ 彩度×明度の重み付け式 (Eq.1)
def sat_val_weight(sat, val, c, t, tau=8.0):
    """w'(x,c) = (100*c/t)^(sat(x)*val(x)*tau)。黒に近い高彩度画素を過大評価しない重み。"""
    if t <= 0 or c <= 0:
        return 0.0
    base = 100.0 * (c / t)
    if base <= 0:
        return 0.0
    exponent = sat * val * tau
    try:
        return float(base ** exponent)
    except (OverflowError, ValueError):
        return 0.0


# ============================================================ Binned-KMeans本体
def binned_kmeans_palette(pixels_rgb, k_per_bin=3, s_min=0.15, v_min=0.08,
                           min_bin_px=10, tau=8.0):
    """pixels_rgb: (N,3) uint8/float RGB画素の配列(ショット内でサンプリングした全フレーム分)。
    戻り値: 虹順(赤→橙→黄→緑→青緑→青→紫)、各ビン内は明度→彩度の順でソートされた
    色情報の辞書のリスト。"""
    pixels_rgb = np.asarray(pixels_rgb, dtype=np.float64)
    total = pixels_rgb.shape[0]
    if total == 0:
        return []
    h, s, v = rgb_to_hsv_arr(pixels_rgb)
    valid = (s >= s_min) & (v >= v_min)
    bins = assign_hue_bins(h)

    out = []
    for name in BIN_ORDER:
        binmask = valid & (bins == name)
        bin_px = pixels_rgb[binmask]
        if bin_px.shape[0] < max(min_bin_px, k_per_bin):
            continue
        k = min(k_per_bin, bin_px.shape[0])
        # n_init=10→3: 実測でRGB中心の平均差0.6/255(誤差レベル)のまま2.5倍速化(2026-07-18)
        km = KMeans(n_clusters=k, random_state=0, n_init=3).fit(bin_px)
        labels = km.labels_
        centers = km.cluster_centers_
        counts = np.bincount(labels, minlength=k)

        bin_colors = []
        for i in range(k):
            c = int(counts[i])
            if c == 0:
                continue
            r, g, b = centers[i]
            hc, sc, vc = rgb_to_hsv_arr(np.array([r, g, b]))
            hc, sc, vc = float(hc), float(sc), float(vc)
            L_lab, a_lab, b_lab, C_lab, h_lab = rgb_to_lab((r, g, b))
            ou_wc = ou_wc_score(C_lab, h_lab)
            ou_activity = ou_activity_score(L_lab, a_lab, b_lab)
            ou_weight = ou_weight_score(L_lab, h_lab)
            share = c / total
            weight = sat_val_weight(sc, vc, c, total, tau=tau)
            bin_colors.append({
                "bin": name, "bin_ja": BIN_LABEL_JA[name],
                "rgb": (float(r), float(g), float(b)), "hex": hex_of((r, g, b)),
                "h": hc, "s": sc, "v": vc,
                "L_lab": L_lab, "a_lab": a_lab, "b_lab": b_lab, "C_lab": C_lab, "h_lab": h_lab,
                "ou_wc": ou_wc, "ou_warm_pct": ou_warm_pct(ou_wc),
                "ou_activity": ou_activity, "ou_weight": ou_weight,
                "pixel_count": c, "share": share, "weight": weight,
            })
        # 各ビン内は明度→彩度の順でソート(原論文の記述どおり、値が高い順)
        bin_colors.sort(key=lambda d: (d["v"], d["s"]), reverse=True)
        out.extend(bin_colors)
    return out


def aggregate_ou_scores(palette):
    """パレット内の各色のOuスコアを、Binned-KMeansの重み付け式(weight=目立ち度)で
    加重平均し、ショット代表の1点に集約する(色ごとの推移ではなく、ショット単位の
    時系列として他の指標(コメント数等)と比較しやすくするため)。
    weightが全色0の場合はshare(画素占有率)で代替する。パレットが空ならNoneを返す。"""
    if not palette:
        return {"ou_wc": None, "ou_warm_pct": None, "ou_activity": None, "ou_weight": None}
    total_w = sum(c["weight"] for c in palette)
    wkey = "weight" if total_w > 0 else "share"
    total_w = sum(c[wkey] for c in palette)
    if total_w <= 0:
        return {"ou_wc": None, "ou_warm_pct": None, "ou_activity": None, "ou_weight": None}

    def wavg(field):
        return sum(c[field] * c[wkey] for c in palette) / total_w

    return {
        "ou_wc": wavg("ou_wc"), "ou_warm_pct": wavg("ou_warm_pct"),
        "ou_activity": wavg("ou_activity"), "ou_weight": wavg("ou_weight"),
    }


# ============================================================ ショットへのサンプリング適用
def sample_shot_frame_indices(start_frame, end_frame, max_samples=5):
    """ショット全域(トリムなし)から等間隔サンプリング。
    shot_color_analysis_ffmpeg_cielab.py の analyze_shots() と同じロジック。"""
    length = end_frame - start_frame
    if length <= 0:
        return [start_frame]
    nsamp = 1 if length <= 2 else min(max_samples, max(2, length // 2))
    idxs = sorted(set(np.linspace(start_frame, end_frame - 1, nsamp).astype(int).tolist()))
    return idxs

def collect_shot_pixels(arrs, idxs):
    chunks = []
    for fi in idxs:
        if 0 <= fi < len(arrs):
            chunks.append(arrs[fi].reshape(-1, 3))
    if not chunks:
        return np.zeros((0, 3))
    return np.concatenate(chunks, axis=0)


# ============================================================ I/O
def read_shots_csv(path):
    """shots.csv を読み込む。fps列があれば、そのfps(全行同一の前提)も一緒に返す。
    fps列が無い、または行ごとに値が食い違う場合は fps=None を返す。"""
    shots = []
    fps_values = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shots.append({
                "shot": int(row["shot"]),
                "start_frame": int(row["start_frame"]), "end_frame": int(row["end_frame"]),
                "start_s": float(row["start_s"]), "end_s": float(row["end_s"]),
                "dur_s": float(row["dur_s"]), "is_fade": bool(int(row["is_fade"])),
            })
            if row.get("fps"):
                fps_values.add(round(float(row["fps"]), 6))
    csv_fps = fps_values.pop() if len(fps_values) == 1 else None
    return shots, csv_fps

def fmt_tc(t):
    m = int(t // 60); return f"{m:02d}:{t - m*60:05.2f}"

def _fmt_opt(v, spec="{:.3f}"):
    return "" if v is None else spec.format(v)

def write_csv(shot_records, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shot", "start_s", "end_s", "dur_s", "is_fade",
                    "shot_ou_wc", "shot_ou_warm_pct", "shot_ou_activity", "shot_ou_weight",
                    "frame_mean_s", "frame_mean_v", "pct_chromatic", "colorfulness",
                    "palette_order", "hue_bin", "hue_bin_ja", "weight_rank",
                    "hex", "r", "g", "b", "h", "s", "v",
                    "L_lab", "a_lab", "b_lab", "C_lab", "h_lab",
                    "ou_wc", "ou_warm_pct", "ou_activity", "ou_weight",
                    "pixel_count", "share", "weight"])
        for rec in shot_records:
            palette = rec["palette"]
            weight_order = sorted(range(len(palette)), key=lambda i: palette[i]["weight"], reverse=True)
            weight_rank = {i: rank + 1 for rank, i in enumerate(weight_order)}
            common = [rec["shot"], f"{rec['start_s']:.3f}", f"{rec['end_s']:.3f}",
                      f"{rec['dur_s']:.3f}", int(rec["is_fade"]),
                      _fmt_opt(rec["ou_wc"]), _fmt_opt(rec["ou_warm_pct"], "{:.1f}"),
                      _fmt_opt(rec["ou_activity"]), _fmt_opt(rec["ou_weight"]),
                      _fmt_opt(rec["frame_mean_s"]), _fmt_opt(rec["frame_mean_v"]),
                      _fmt_opt(rec["pct_chromatic"]), _fmt_opt(rec["colorfulness"])]
            if not palette:
                # 有彩色パレットが空のショットでも、全フレーム由来の値(frame_mean_s等)は
                # 失わずに1行残す(無彩色主体のショットが集約から消えて彩度が過大評価される
                # のを防ぐため。集約層_design_260716.md 1.1節)。
                # palette側の23列("palette_order"〜"weight")は空欄。
                w.writerow(common + [""] * 23)
                continue
            for order, c in enumerate(palette):
                w.writerow(common + [order, c["bin"], c["bin_ja"], weight_rank[order],
                            c["hex"], f"{c['rgb'][0]:.1f}", f"{c['rgb'][1]:.1f}", f"{c['rgb'][2]:.1f}",
                            f"{c['h']:.1f}", f"{c['s']:.3f}", f"{c['v']:.3f}",
                            f"{c['L_lab']:.2f}", f"{c['a_lab']:.2f}", f"{c['b_lab']:.2f}",
                            f"{c['C_lab']:.2f}", f"{c['h_lab']:.1f}",
                            f"{c['ou_wc']:.3f}", f"{c['ou_warm_pct']:.1f}",
                            f"{c['ou_activity']:.3f}", f"{c['ou_weight']:.3f}",
                            c["pixel_count"], f"{c['share']:.5f}", f"{c['weight']:.4f}"])

def write_json(shot_records, path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "shots": shot_records}, f, ensure_ascii=False, indent=2)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(
        description="Binned-KMeans法によるショット単位のドミナントカラー時系列抽出")
    ap.add_argument("input", help="動画ファイル")
    ap.add_argument("--shots-csv", required=True,
        help="shot_cut_detection.py が出力した <base>_shots.csv")
    ap.add_argument("--fps", type=float, default=None,
        help="フレームサンプリングfps。省略時は shots-csv の fps列から自動取得する。"
             "指定した場合は shots-csv の fps列と食い違うと警告を出したうえで指定値を優先する")
    ap.add_argument("--width", type=int, default=240)
    ap.add_argument("--k", type=int, default=3, help="色相ビンごとのKMeansクラスタ数 既定3")
    ap.add_argument("--max-samples", type=int, default=5, help="ショットごとの最大サンプルフレーム数 既定5")
    ap.add_argument("--s-min", type=float, default=0.15, help="彩度の最小値(無彩色除外) 既定0.15")
    ap.add_argument("--v-min", type=float, default=0.08, help="明度の最小値(黒つぶれ除外) 既定0.08")
    ap.add_argument("--min-bin-px", type=int, default=10, help="ビンを採用する最小画素数 既定10")
    ap.add_argument("--tau", type=float, default=8.0, help="重み付け式のスケール係数τ 既定8.0(原論文と同じ)")
    ap.add_argument("--outdir", default="binned_kmeans_out",
        help="出力先の親フォルダ。実際の出力は outdir/動画タイトル名/ の中に入る")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.input): sys.exit(f"見つかりません: {args.input}")
    if not os.path.isfile(args.shots_csv): sys.exit(f"見つかりません: {args.shots_csv}")
    need_ffmpeg()

    shots, csv_fps = read_shots_csv(args.shots_csv)
    if not shots: sys.exit("shots-csv にショットがありません。")

    if args.fps is None:
        if csv_fps is None:
            sys.exit("shots-csv に fps列がない(または行ごとに値が食い違う)ため、--fps を明示的に指定してください。")
        args.fps = csv_fps
        print(f"  shots-csv から fps={args.fps} を自動取得しました。", file=sys.stderr)
    elif csv_fps is not None and abs(args.fps - csv_fps) > 1e-6:
        print(f"  警告: 指定された --fps={args.fps} が shots-csv の fps列({csv_fps})と食い違っています。"
              f" --fps={args.fps} を優先して処理を続けます。", file=sys.stderr)

    tmpdir = tempfile.mkdtemp(prefix="frames_")
    try:
        print(f"ffmpegでフレーム抽出中 (fps={args.fps}, width={args.width})…", file=sys.stderr)
        files = extract_frames(args.input, args.fps, args.width, tmpdir)
        if not files: sys.exit("フレームが抽出できませんでした。")
        print(f"  {len(files)} 枚抽出 → 読み込み中…", file=sys.stderr)
        arrs = load_all(files)

        max_frame = max(s["end_frame"] for s in shots)
        if max_frame > len(arrs):
            print(f"  警告: shots-csv の end_frame({max_frame})がフレーム数({len(arrs)})を超えています。"
                  f" --fps/--shots-csv が同じ動画・同じfpsで生成されたか確認してください。", file=sys.stderr)

        print(f"  Binned-KMeans抽出中（{len(shots)}ショット）…", file=sys.stderr)
        shot_records = []
        for sh in shots:
            idxs = sample_shot_frame_indices(sh["start_frame"], sh["end_frame"], args.max_samples)
            pixels = collect_shot_pixels(arrs, idxs)
            palette = binned_kmeans_palette(
                pixels, k_per_bin=args.k, s_min=args.s_min, v_min=args.v_min,
                min_bin_px=args.min_bin_px, tau=args.tau)
            ou_agg = aggregate_ou_scores(palette)
            level = frame_level_stats(pixels, s_min=args.s_min, v_min=args.v_min)
            shot_records.append({
                "shot": sh["shot"], "start_s": sh["start_s"], "end_s": sh["end_s"],
                "dur_s": sh["dur_s"], "is_fade": sh["is_fade"],
                "n_sampled_frames": len(idxs), "palette": palette,
                "ou_wc": ou_agg["ou_wc"], "ou_warm_pct": ou_agg["ou_warm_pct"],
                "ou_activity": ou_agg["ou_activity"], "ou_weight": ou_agg["ou_weight"],
                "frame_mean_s": level["frame_mean_s"], "frame_mean_v": level["frame_mean_v"],
                "pct_chromatic": level["pct_chromatic"], "colorfulness": level["colorfulness"],
            })

        base = os.path.splitext(os.path.basename(args.input))[0]
        outsub = os.path.join(args.outdir, base)
        os.makedirs(outsub, exist_ok=True)
        csv_path = os.path.join(outsub, f"{base}_binned_kmeans_timeseries.csv")
        json_path = os.path.join(outsub, f"{base}_binned_kmeans_timeseries.json")
        write_csv(shot_records, csv_path)
        write_json(shot_records, json_path, meta={
            "input": args.input, "shots_csv": args.shots_csv,
            "sample_fps": args.fps, "width": args.width,
            "k_per_bin": args.k, "max_samples": args.max_samples,
            "s_min": args.s_min, "v_min": args.v_min, "min_bin_px": args.min_bin_px, "tau": args.tau,
            "bin_order": BIN_ORDER,
        })

        if args.keep_frames:
            kept = os.path.join(outsub, "frames"); shutil.rmtree(kept, ignore_errors=True)
            shutil.copytree(tmpdir, kept)

        print(f"\n完了 → {csv_path}", file=sys.stderr)
        for rec in shot_records:
            n = len(rec["palette"])
            top = f" 主色={rec['palette'][0]['hex']}({rec['palette'][0]['bin_ja']})" if n else " (色抽出なし)"
            ou = (f" Ou[暖{rec['ou_warm_pct']:.0f}% activity{rec['ou_activity']:+.2f} weight{rec['ou_weight']:+.2f}]"
                  if rec["ou_activity"] is not None else "")
            print(f"    #{rec['shot']:03d} {fmt_tc(rec['start_s'])}–{fmt_tc(rec['end_s'])} "
                  f"色数={n}{top}{ou}", file=sys.stderr)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
