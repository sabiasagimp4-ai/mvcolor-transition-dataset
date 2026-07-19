#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_transnetv2_shots.py
==============================================================================
videos.csv の全動画について、TransNetV2(CUDA)でショット境界検出だけをバッチ実行する。
shot_cut_detection.py を動画ごとにサブプロセス起動すると毎回モデル読み込み+CUDA初期化の
オーバーヘッドが乗るため、本スクリプトは同一プロセス内でモデルを1回だけ読み込み、
全動画をループ処理する(GPUは1台なのでシーケンシャル実行、並列化はしない)。

出力先は既存のHSV版shots_out(627→1003→1567本で使用)とは別ディレクトリにし、
旧結果を上書きしない(HSV版とTransNetV2版を並べて比較できるように保持する)。

使い方:
  python run_transnetv2_shots.py --manifest videos.csv \
      --videos-dir /d/自自論_video_data/videos \
      --shots-dir /d/自自論_video_data/shots_out_transnetv2
"""
import argparse, os, sys, csv, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import shot_cut_detection as scd


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="TransNetV2でvideos.csv全動画のショット検出をバッチ実行(モデル1回読込)")
    ap.add_argument("--manifest", default="videos.csv")
    ap.add_argument("--videos-dir", default="/d/自自論_video_data/videos")
    ap.add_argument("--shots-dir", default="/d/自自論_video_data/shots_out_transnetv2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=float, default=8.0, help="出力する仮想サンプリングfps 既定8")
    ap.add_argument("--transnet-threshold", type=float, default=0.5)
    ap.add_argument("--fast-cut-thresh", type=float, default=0.5)
    ap.add_argument("--log", default="/d/自自論_video_data/run_transnetv2_shots.log")
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)
    todo, skip_done, skip_novideo = [], [], []
    for m in manifest:
        vid = (m.get("id") or "").strip()
        if not vid:
            continue
        outsub = os.path.join(args.shots_dir, vid)
        csv_path = os.path.join(outsub, f"{vid}_shots.csv")
        if os.path.isfile(csv_path):
            skip_done.append(vid); continue
        video_path = os.path.join(args.videos_dir, f"{vid}.mp4")
        if not os.path.isfile(video_path):
            skip_novideo.append(vid); continue
        todo.append((vid, video_path))

    print(f"対象 {len(todo)}本 / 既完了スキップ {len(skip_done)} / 動画未取得スキップ {len(skip_novideo)}", flush=True)
    print("TransNetV2モデルを読み込み中…", flush=True)
    model = scd.load_transnetv2_model(args.device)
    print("読み込み完了。検出開始。", flush=True)

    ok, fail = [], []
    log = []
    t0 = time.time()
    for i, (vid, video_path) in enumerate(todo, 1):
        try:
            dur, fps, src_w, src_h = scd.probe_duration_fps(video_path)
            shots_raw = scd.detect_shots_transnetv2(video_path, model, args.device,
                                                     pipeline_fps=args.fps,
                                                     threshold=args.transnet_threshold)
            records = scd.shots_to_records(shots_raw, args.fps, args.fast_cut_thresh)
            outsub = os.path.join(args.shots_dir, vid)
            os.makedirs(outsub, exist_ok=True)
            csv_path = os.path.join(outsub, f"{vid}_shots.csv")
            json_path = os.path.join(outsub, f"{vid}_shots.json")
            scd.write_csv(records, csv_path, args.fps)
            scd.write_json(records, json_path, meta={
                "input": video_path, "duration_s": dur, "source_fps": fps,
                "source_width": src_w, "source_height": src_h,
                "method": "transnetv2", "transnet_threshold": args.transnet_threshold,
                "pipeline_fps": args.fps, "fast_cut_thresh": args.fast_cut_thresh,
            })
            ok.append(vid)
            print(f"[{i:>4}/{len(todo)}] {vid} [OK] ショット{len(records)}", flush=True)
        except Exception as e:
            fail.append(vid)
            log.append(f"[FAIL] {vid}: {e}")
            print(f"[{i:>4}/{len(todo)}] {vid} [FAIL] {e}", flush=True)

    with open(args.log, "a", encoding="utf-8") as f:
        f.write(f"\n=== run {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write("\n".join(log) + ("\n" if log else ""))

    dt = time.time() - t0
    print(f"\n完了 ({dt/60:.1f}分): OK {len(ok)} / 既完了スキップ {len(skip_done)} / "
          f"動画未取得スキップ {len(skip_novideo)} / 失敗 {len(fail)}", flush=True)
    if fail:
        print("失敗id(ログ参照): " + ", ".join(fail), flush=True)


if __name__ == "__main__":
    main()
