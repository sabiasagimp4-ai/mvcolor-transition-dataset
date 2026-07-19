#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shot_cut_detection.py — カット境界検出（shot_color_analysis_ffmpeg_cielab.py からの切り出し）
==============================================================================
shot_color_analysis_ffmpeg_cielab.py のうち、ffmpegによるフレーム抽出から
カット/フェード境界検出(detect_shots)までを単独スクリプトとして切り出したもの。
色抽出・CIELAB変換・寒暖判定・レポート生成など「色彩分析」パートは含まない。

カット検出手法は2種類選択可能(--method):
  transnetv2 (既定, 2026-07-18追加): TransNetV2(Souček & Lokoč, 2020)による
    深層学習ベースのショット境界検出。CUDA(GPU)推論。動画のネイティブfpsで
    推論した後、結果を従来のパイプラインが期待する仮想fpsグリッド(既定8fps)に
    量子化して出力するため、下流スクリプト(shot_dominant_color_binned_kmeans.py等)
    は無改造で動作する。詳細は shot_cut_detection_transnetv2_仕様書.md 参照。
  hsv (旧来手法、後方互換用): HSVヒストグラム(Bhattacharyya距離)ベース。
    627本→1567本まで使用していた手法をそのまま残してある(意図的にHSVのまま)。

使い方:
  python shot_cut_detection.py INPUT.mp4                              # TransNetV2(既定)
  python shot_cut_detection.py INPUT.mp4 --method hsv --cut-thresh 0.45 --fade-lo 0.20 --outdir out

  HSV法の抽出フレーム幅は既定で元動画の横幅の1/3に自動リサイズされる(--widthで上書き可能)。
  カット検出は解像度に鈍感なヒストグラムベースの手法のため、縮小しても精度への影響は
  ほぼ無く、処理速度・メモリ使用量を削減できる。

出力 (outdir/動画タイトル名/ 以下):
  <base>_shots.csv   ショット一覧 (shot, start_frame, end_frame, start_s, end_s, dur_s, is_fade,
                      is_fast_cut, fps)
                      ※ fps列は全行に同じ値(サンプリングfps)を埋め込んだもの。
                      下流スクリプトが --fps を別途指定しなくても自動復元できるようにするため。
                      ※ is_fast_cut列は dur_s が --fast-cut-thresh(既定0.5秒)未満のショットに
                      自動で1が立つ。AEでの目視修正時に手動で[fast]タグを付けても1になる
                      （ae_markers_workflow_仕様書.md 参照）。
  <base>_shots.json  同内容 + メタ情報をJSONでも出力（他スクリプトから読み込みやすい形）

必要なもの:
  ・ffmpeg / ffprobe (コマンド)
  ・pip install numpy Pillow
  ・(transnetv2法) pip install torch --index-url https://download.pytorch.org/whl/cu126
    と、transnetv2_model/ 以下に transnetv2_pytorch.py + transnetv2-pytorch-weights.pth

詳しい仕様は shot_cut_detection_仕様書.md / shot_cut_detection_transnetv2_仕様書.md を参照。
"""

import argparse, os, sys, csv, json, subprocess, tempfile, shutil
import numpy as np
try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow が必要:  pip install numpy Pillow")

TRANSNET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transnetv2_model")


# ============================================================ ffmpeg helpers
def need_ffmpeg():
    for cmd in ("ffmpeg", "ffprobe"):
        if shutil.which(cmd) is None:
            sys.exit(f"{cmd} が見つかりません。ffmpeg をインストールしてください。\n"
                     f"  macOS: brew install ffmpeg  /  Ubuntu: sudo apt install ffmpeg\n"
                     f"  Windows: https://ffmpeg.org/download.html")

def probe_duration_fps(path):
    """動画の総尺・fpsに加えて、元の横幅・縦幅(px)も取得する
    (--width未指定時に「元解像度の1/3」を自動計算するため)。"""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,duration,width,height",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True)
    info = json.loads(out.stdout or "{}")
    dur, fps = 0.0, 30.0
    src_w, src_h = None, None
    try:
        st = info.get("streams", [{}])[0]
        num, den = st.get("avg_frame_rate", "30/1").split("/")
        if float(den) != 0:
            fps = float(num) / float(den)
        dur = float(st.get("duration") or info.get("format", {}).get("duration") or 0)
        src_w = int(st["width"]); src_h = int(st["height"])
    except Exception:
        pass
    return dur, fps, src_w, src_h

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


# ============================================================ load frames
def load_all(files):
    """全PNGを読み、RGB配列として保持(メモリ節約のため配列は縮小済み)。"""
    arrs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        arrs.append(np.asarray(im))
    return arrs


# ============================================================ cut detection (HSVヒストグラム差分)
def hs_histogram(arr, hbins=32, sbins=32):
    """カット検出用 H-S ヒストグラム(正規化)。arr: (H,W,3) uint8 RGB.
    意図的にHSVのまま(ショット境界検出という工学的な用途であり、色彩の科学的
    測定値のCIELAB化とは別の関心事のため)。"""
    flat = arr.reshape(-1, 3).astype(np.float32) / 255.0
    mx = flat.max(1); mn = flat.min(1); d = mx - mn
    v = mx; s = np.where(mx > 0, d / (mx + 1e-12), 0)
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    h = np.zeros_like(v)
    nz = d > 1e-12
    idx = (mx == r) & nz; h[idx] = ((g[idx] - b[idx]) / d[idx]) % 6
    idx = (mx == g) & nz; h[idx] = (b[idx] - r[idx]) / d[idx] + 2
    idx = (mx == b) & nz; h[idx] = (r[idx] - g[idx]) / d[idx] + 4
    h = (h * 60) % 360
    H = np.clip((h / 360 * hbins).astype(int), 0, hbins - 1)
    S = np.clip((s * sbins).astype(int), 0, sbins - 1)
    hist = np.zeros((hbins, sbins), float)
    np.add.at(hist, (H, S), 1.0)
    n = hist.sum()
    return hist / n if n > 0 else hist

def bhattacharyya(h1, h2):
    return 1.0 - np.sum(np.sqrt(h1 * h2))

def detect_shots(arrs, sample_fps, cut_thresh, fade_lo):
    """隣接フレーム間のH-Sヒストグラム差分(Bhattacharyya距離)から
    ハードカット/フェード境界を検出し、[(start_frame, end_frame, is_fade), ...] を返す。"""
    n = len(arrs)
    hists = [hs_histogram(a) for a in arrs]
    diffs = np.array([bhattacharyya(hists[i], hists[i + 1]) for i in range(n - 1)])

    cut_set = set(); fade_starts = set()
    i = 0
    while i < len(diffs):
        if diffs[i] >= cut_thresh:
            cut_set.add(i + 1); i += 1
        elif diffs[i] >= fade_lo:
            j = i
            while j < len(diffs) and diffs[j] >= fade_lo:
                j += 1
            cut_set.add(i + 1); cut_set.add(j + 1); fade_starts.add(i + 1)
            i = j + 1
        else:
            i += 1

    boundaries = sorted({0, n} | {b for b in cut_set if 0 < b < n})
    raw = [[a, b, a in fade_starts] for a, b in zip(boundaries, boundaries[1:]) if b > a]

    min_frames = max(1, int(round(sample_fps * 0.25)))  # ~0.25秒未満の破片を吸収(低fpsでも1以上)
    cleaned = []
    for a, b, isf in raw:
        if (b - a) < min_frames and not isf and cleaned:
            cleaned[-1][1] = b
        else:
            cleaned.append([a, b, isf])
    return [(a, b, isf) for a, b, isf in cleaned]


# ============================================================ cut detection (TransNetV2, CUDA)
def load_transnetv2_model(device="cuda"):
    """transnetv2_model/ の重み(TF原論文からPyTorchへ変換済み、100%一致検証済み)を読み込む。
    1プロセスにつき1回だけ呼ぶこと(バッチ処理では呼び出し側でキャッシュして使い回す)。"""
    if not os.path.isdir(TRANSNET_DIR):
        sys.exit(f"{TRANSNET_DIR} が見つかりません。transnetv2_pytorch.py と "
                  f"transnetv2-pytorch-weights.pth を配置してください。")
    sys.path.insert(0, TRANSNET_DIR)
    import torch
    from transnetv2_pytorch import TransNetV2
    model = TransNetV2()
    state_dict = torch.load(os.path.join(TRANSNET_DIR, "transnetv2-pytorch-weights.pth"),
                             map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model

def extract_frames_native_raw(path):
    """TransNetV2の入力仕様(27x48 RGB, ネイティブfps)に合わせてffmpegで直接rawvideoパイプ抽出。
    ファイルに書き出さずメモリ上のバッファとして受け取るため、HSV法のPPM書き出しより高速。
    戻り値: (frames [N,27,48,3] uint8 ndarray, ネイティブfps)"""
    _, native_fps, _, _ = probe_duration_fps(path)
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-vf", "scale=48:27", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        sys.exit(f"ffmpeg(TransNetV2用抽出) 失敗:\n{r.stderr.decode('utf-8', 'replace')}")
    frames = np.frombuffer(r.stdout, np.uint8).reshape([-1, 27, 48, 3])
    return frames, native_fps

def transnetv2_predict(frames, model, device):
    """predict_frames()(公式TF/PyTorch実装と同一の100フレーム窓・前後25フレームパディング方式)を移植。
    戻り値: (single_frame_pred [N] float, all_frame_pred [N] float) いずれもsigmoid後の確率。"""
    import torch
    n = len(frames)
    no_pad_start = 25
    no_pad_end = 25 + 50 - (n % 50 if n % 50 != 0 else 50)
    start_frame = frames[:1]
    end_frame = frames[-1:]
    padded = np.concatenate([np.repeat(start_frame, no_pad_start, axis=0), frames,
                              np.repeat(end_frame, no_pad_end, axis=0)], axis=0)

    singles, manys = [], []
    ptr = 0
    with torch.no_grad():
        while ptr + 100 <= len(padded):
            window = padded[ptr:ptr + 100]
            x = torch.from_numpy(window[np.newaxis]).to(device)  # [1,100,27,48,3] uint8
            single_logits, extra = model(x)
            single = torch.sigmoid(single_logits)[0, 25:75, 0].cpu().numpy()
            many = torch.sigmoid(extra["many_hot"])[0, 25:75, 0].cpu().numpy()
            singles.append(single); manys.append(many)
            ptr += 50
    single_frame_pred = np.concatenate(singles)[:n]
    all_frame_pred = np.concatenate(manys)[:n]
    return single_frame_pred, all_frame_pred

def predictions_to_scenes(predictions, threshold=0.5):
    """公式実装と同一のロジック。0/1化した予測値から、境界フレームで区切られた
    ショット区間 [start, end](両端含む、ネイティブフレーム番号)のリストを返す。"""
    preds = (predictions > threshold).astype(np.uint8)
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(preds):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i - 1])
        t_prev = t
    if t == 0:
        scenes.append([start, len(preds) - 1])
    if not scenes:
        return [[0, len(preds) - 1]]
    return scenes

def detect_shots_transnetv2(path, model, device, pipeline_fps=8.0, threshold=0.5,
                             min_fade_gap_native_frames=2):
    """TransNetV2でネイティブfps推論→区間検出したのち、結果を pipeline_fps の仮想フレーム
    グリッドに量子化して、旧HSV版 detect_shots() と同じ (start_frame, end_frame, is_fade) の
    リストを返す(下流の shots_to_records/write_csv/write_json は完全に無改造で使える)。

    ショット間の遷移(トランジション)区間は、ネイティブフレーム番号でその中点を境界に採用し、
    タイムライン全体を隙間なくショットで覆う(旧HSV版と同じ「連続ショット」前提を維持)。
    遷移がmin_fade_gap_native_frames以上に及ぶ場合のみ is_fade=True とする
    (1フレームだけの遷移=瞬時のハードカットとみなす)。"""
    frames, native_fps = extract_frames_native_raw(path)
    single_pred, _ = transnetv2_predict(frames, model, device)
    scenes = predictions_to_scenes(single_pred, threshold)  # [[s0,e0],[s1,e1],...] 両端含む

    n_native = len(frames)
    bounds_native = [0]
    is_fade_flags = []  # shot i (i>=1) が「直前の遷移がfadeか」
    for i in range(len(scenes) - 1):
        prev_end = scenes[i][1]
        next_start = scenes[i + 1][0]
        gap = next_start - prev_end - 1  # 遷移区間の長さ(ネイティブフレーム数)
        boundary = int(round((prev_end + next_start + 1) / 2))
        bounds_native.append(boundary)
        is_fade_flags.append(gap >= min_fade_gap_native_frames)
    bounds_native.append(n_native)

    # 秒に変換してから pipeline_fps の仮想グリッドに量子化(既存パイプラインとの互換性のため)
    bounds_s = [b / native_fps for b in bounds_native]
    bounds_frame = [int(round(s * pipeline_fps)) for s in bounds_s]
    bounds_frame[0] = 0
    bounds_frame[-1] = max(bounds_frame[-1], bounds_frame[0] + 1)
    for i in range(1, len(bounds_frame)):
        if bounds_frame[i] <= bounds_frame[i - 1]:
            bounds_frame[i] = bounds_frame[i - 1] + 1

    shots = []
    for i in range(len(bounds_frame) - 1):
        a, b = bounds_frame[i], bounds_frame[i + 1]
        isf = is_fade_flags[i - 1] if i >= 1 else False
        shots.append((a, b, isf))
    return shots


# ============================================================ output helpers
def fmt_tc(t):
    m = int(t // 60); return f"{m:02d}:{t - m*60:05.2f}"

def shots_to_records(shots, sample_fps, fast_cut_thresh):
    """detect_shotsの(start_frame, end_frame, is_fade)タプル列を、
    フレーム番号と秒数を両方持つ辞書列に変換する。
    dur_s が fast_cut_thresh 秒未満のショットは is_fast_cut=True とする
    （高速カットを後段で分けて集計できるようにするための自動フラグ。
    AEでの目視修正時に手動で[fast]タグを付けた場合も1になる）。"""
    records = []
    for i, (a, b, isf) in enumerate(shots, start=1):
        dur_s = (b - a) / sample_fps
        records.append({
            "shot": i,
            "start_frame": a, "end_frame": b,
            "start_s": a / sample_fps, "end_s": b / sample_fps,
            "dur_s": dur_s,
            "is_fade": bool(isf),
            "is_fast_cut": dur_s < fast_cut_thresh,
        })
    return records

def write_csv(records, path, sample_fps):
    """sample_fps を全行に埋め込む(下流スクリプトが --fps を別途指定しなくても
    このCSVから自動的にサンプリングfpsを復元できるようにするため)。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shot", "start_frame", "end_frame", "start_s", "end_s", "dur_s",
                     "is_fade", "is_fast_cut", "fps"])
        for r in records:
            w.writerow([r["shot"], r["start_frame"], r["end_frame"],
                        f"{r['start_s']:.3f}", f"{r['end_s']:.3f}", f"{r['dur_s']:.3f}",
                        int(r["is_fade"]), int(r["is_fast_cut"]), sample_fps])

def write_json(records, path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "shots": records}, f, ensure_ascii=False, indent=2)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="カット/フェード境界検出(TransNetV2既定、HSV旧法も選択可)")
    ap.add_argument("input")
    ap.add_argument("--method", choices=["transnetv2", "hsv"], default="transnetv2",
        help="検出手法。transnetv2(既定): CUDA上のTransNetV2。hsv: 旧来のHSVヒストグラム法")
    ap.add_argument("--device", default="cuda", help="transnetv2法のデバイス 既定cuda")
    ap.add_argument("--fps", type=float, default=8.0,
        help="hsv法: 抽出フレームレート(枚/秒)。transnetv2法: 出力する仮想サンプリングfps"
             "(下流スクリプトが期待するfps。ネイティブfps推論の結果をこのグリッドに量子化する) 既定8")
    ap.add_argument("--cut-thresh", type=float, default=0.45, help="[hsv法] ハードカット判定 既定0.45")
    ap.add_argument("--fade-lo", type=float, default=0.20, help="[hsv法] フェード帯下限 既定0.20")
    ap.add_argument("--transnet-threshold", type=float, default=0.5,
        help="[transnetv2法] 遷移フレーム判定の確率閾値 既定0.5(公式実装と同じ)")
    ap.add_argument("--fast-cut-thresh", type=float, default=0.5,
        help="この秒数未満のショットに自動で is_fast_cut=1 を立てる 既定0.5")
    ap.add_argument("--width", type=int, default=None,
        help="[hsv法] 抽出フレームの幅(px)。省略時は元動画の横幅の1/3を自動計算して使う"
             "(取得できない場合のみ既定値240pxにフォールバック)")
    ap.add_argument("--outdir", default="cut_out",
        help="出力先の親フォルダ。実際の出力は outdir/動画タイトル名/ の中に入る 既定cut_out")
    ap.add_argument("--keep-frames", action="store_true", help="[hsv法] 抽出した連番PPMを消さずに残す")
    args = ap.parse_args()

    if not os.path.isfile(args.input): sys.exit(f"見つかりません: {args.input}")
    need_ffmpeg()

    dur, fps, src_w, src_h = probe_duration_fps(args.input)
    base = os.path.splitext(os.path.basename(args.input))[0]
    outsub = os.path.join(args.outdir, base)
    os.makedirs(outsub, exist_ok=True)
    csv_path = os.path.join(outsub, f"{base}_shots.csv")
    json_path = os.path.join(outsub, f"{base}_shots.json")

    if args.method == "transnetv2":
        print(f"TransNetV2({args.device})でカット検出中…", file=sys.stderr)
        model = load_transnetv2_model(args.device)
        shots_raw = detect_shots_transnetv2(args.input, model, args.device,
                                             pipeline_fps=args.fps, threshold=args.transnet_threshold)
        records = shots_to_records(shots_raw, args.fps, args.fast_cut_thresh)
        write_csv(records, csv_path, args.fps)
        write_json(records, json_path, meta={
            "input": args.input, "duration_s": dur, "source_fps": fps,
            "source_width": src_w, "source_height": src_h,
            "method": "transnetv2", "transnet_threshold": args.transnet_threshold,
            "pipeline_fps": args.fps, "fast_cut_thresh": args.fast_cut_thresh,
        })
        n_fade = sum(1 for r in records if r["is_fade"])
        n_fast = sum(1 for r in records if r["is_fast_cut"])
        print(f"\n完了 → {csv_path}", file=sys.stderr)
        print(f"  ショット{len(records)}(フェード境界{n_fade}、高速カット{n_fast})", file=sys.stderr)
        for r in records:
            tag = (" [fade]" if r["is_fade"] else "") + (" [fast]" if r["is_fast_cut"] else "")
            print(f"    #{r['shot']} {fmt_tc(r['start_s'])}–{fmt_tc(r['end_s'])} ({r['dur_s']:.2f}s){tag}",
                  file=sys.stderr)
        return

    # ---- hsv法(旧来) ----
    if args.width is None:
        if src_w:
            args.width = max(2, src_w // 3)
            print(f"  --width未指定のため、元動画の横幅({src_w}px)の1/3 = {args.width}px を自動計算して使用します。",
                  file=sys.stderr)
        else:
            args.width = 240
            print(f"  警告: 元動画の解像度を取得できなかったため、既定値 --width=240 を使用します。", file=sys.stderr)

    tmpdir = tempfile.mkdtemp(prefix="frames_")
    try:
        print(f"ffmpegでフレーム抽出中 (fps={args.fps}, width={args.width})…", file=sys.stderr)
        files = extract_frames(args.input, args.fps, args.width, tmpdir)
        if not files: sys.exit("フレームが抽出できませんでした。")
        print(f"  {len(files)} 枚抽出 → 読み込み中…", file=sys.stderr)
        arrs = load_all(files)
        print(f"  カット検出…", file=sys.stderr)
        shots_raw = detect_shots(arrs, args.fps, args.cut_thresh, args.fade_lo)
        records = shots_to_records(shots_raw, args.fps, args.fast_cut_thresh)

        write_csv(records, csv_path, args.fps)
        write_json(records, json_path, meta={
            "input": args.input, "duration_s": dur, "source_fps": fps,
            "source_width": src_w, "source_height": src_h,
            "method": "hsv",
            "sample_fps": args.fps, "sample_width": args.width,
            "cut_thresh": args.cut_thresh, "fade_lo": args.fade_lo,
            "fast_cut_thresh": args.fast_cut_thresh,
            "n_frames_sampled": len(files),
        })

        if args.keep_frames:
            kept = os.path.join(outsub, "frames"); shutil.rmtree(kept, ignore_errors=True)
            shutil.copytree(tmpdir, kept)

        n_fade = sum(1 for r in records if r["is_fade"])
        n_fast = sum(1 for r in records if r["is_fast_cut"])
        print(f"\n完了 → {csv_path}", file=sys.stderr)
        print(f"  ショット{len(records)}(フェード境界{n_fade}、高速カット{n_fast})", file=sys.stderr)
        for r in records:
            tag = (" [fade]" if r["is_fade"] else "") + (" [fast]" if r["is_fast_cut"] else "")
            print(f"    #{r['shot']} {fmt_tc(r['start_s'])}–{fmt_tc(r['end_s'])} ({r['dur_s']:.2f}s){tag}",
                  file=sys.stderr)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
