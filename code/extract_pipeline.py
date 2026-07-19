#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_pipeline.py
==============================================================================
videos.csv の各行について、
  1. shot_cut_detection.py でショット境界を検出
  2. shot_dominant_color_binned_kmeans.py で色抽出
を順に実行するオーケストレーター。

再開可能: <id>_binned_kmeans_timeseries.csv が既にあるidはスキップする。
動画ファイルがまだ無いid(=ダウンロード未完了)もスキップし、次のidへ進む
(download_videos.py と並行実行し、後で再実行すれば取りこぼしを拾える)。
1本失敗しても全体を止めず、ログに残して次へ進む。

使い方:
  python extract_pipeline.py --manifest videos.csv \
      --videos-dir /d/自自論_video_data/videos \
      --shots-dir /d/自自論_video_data/shots_out \
      --extract-dir /d/自自論_video_data/binned_kmeans_out
"""

import argparse, os, sys, csv, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
SHOT_CUT = os.path.join(HERE, "shot_cut_detection.py")
COLOR_EXTRACT = os.path.join(HERE, "shot_dominant_color_binned_kmeans.py")


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_exclude_ids(path):
    """check_video_authenticity.pyが出力した除外リストCSVからidの集合を読む。"""
    if not path:
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f)}


def process_one(vid, video_path, shots_dir, extract_dir):
    """1本処理。戻り値: (vid, 'ok'|'fail', ログ文字列 or None)。
    別プロセスで実行するためモジュールレベル関数にしている(ProcessPoolExecutor制約)。"""
    shots_csv = os.path.join(shots_dir, vid, f"{vid}_shots.csv")
    ts_csv = os.path.join(extract_dir, vid, f"{vid}_binned_kmeans_timeseries.csv")

    if not os.path.isfile(shots_csv):
        r = subprocess.run(
            [sys.executable, SHOT_CUT, video_path, "--outdir", shots_dir],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not os.path.isfile(shots_csv):
            return vid, "fail", f"[FAIL:shot_cut] {vid}\n{r.stderr[-1000:]}"

    r = subprocess.run(
        [sys.executable, COLOR_EXTRACT, video_path, "--shots-csv", shots_csv, "--outdir", extract_dir],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not os.path.isfile(ts_csv):
        return vid, "fail", f"[FAIL:color_extract] {vid}\n{r.stderr[-1000:]}"
    return vid, "ok", None


def main():
    ap = argparse.ArgumentParser(description="videos.csvの全動画にshot_cut+色抽出を順に実行(再開可能)")
    ap.add_argument("--manifest", default="videos.csv")
    ap.add_argument("--videos-dir", default="/d/自自論_video_data/videos")
    ap.add_argument("--shots-dir", default="/d/自自論_video_data/shots_out")
    ap.add_argument("--extract-dir", default="/d/自自論_video_data/binned_kmeans_out")
    ap.add_argument("--log", default="/d/自自論_video_data/extract_pipeline.log")
    ap.add_argument("--workers", type=int, default=8, help="並列実行本数 既定8")
    ap.add_argument("--exclude-csv", default=None,
        help="check_video_authenticity.pyが出力した除外リストCSV(id列)。該当idは処理をスキップする")
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)
    exclude_ids = read_exclude_ids(args.exclude_csv)
    ok, skip_done, skip_novideo, skip_excluded, fail = [], [], [], [], []
    log = []
    t0 = time.time()

    todo = []
    for m in manifest:
        vid = (m.get("id") or "").strip()
        if not vid:
            continue
        if vid in exclude_ids:
            skip_excluded.append(vid)
            continue
        ts_csv = os.path.join(args.extract_dir, vid, f"{vid}_binned_kmeans_timeseries.csv")
        if os.path.isfile(ts_csv):
            skip_done.append(vid)
            continue
        video_path = os.path.join(args.videos_dir, f"{vid}.mp4")
        if not os.path.isfile(video_path):
            skip_novideo.append(vid)
            continue
        todo.append((vid, video_path))

    print(f"対象 {len(todo)}本 / 既完了スキップ {len(skip_done)} / 動画未取得スキップ {len(skip_novideo)} "
          f"/ 除外リストスキップ {len(skip_excluded)} / 並列数{args.workers}", flush=True)

    done_n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, vid, video_path, args.shots_dir, args.extract_dir): vid
                   for vid, video_path in todo}
        for fut in as_completed(futures):
            vid, status, err = fut.result()
            done_n += 1
            if status == "ok":
                ok.append(vid)
                print(f"[{done_n:>4}/{len(todo)}] {vid} [OK]", flush=True)
            else:
                fail.append(vid)
                log.append(err)
                print(f"[{done_n:>4}/{len(todo)}] {vid} [FAIL] (詳細はログ参照)", flush=True)

    with open(args.log, "a", encoding="utf-8") as f:
        f.write(f"\n=== run {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write("\n".join(log) + ("\n" if log else ""))

    dt = time.time() - t0
    print(f"\n完了 ({dt/60:.1f}分): OK {len(ok)} / 既完了スキップ {len(skip_done)} / "
          f"動画未取得スキップ {len(skip_novideo)} / 除外リストスキップ {len(skip_excluded)} / 失敗 {len(fail)}", flush=True)
    if fail:
        print("失敗id(ログ参照): " + ", ".join(fail), flush=True)
    if skip_novideo:
        print(f"動画未取得: {len(skip_novideo)}件 (download_videos.py の完了後に再実行すれば拾えます)", flush=True)


if __name__ == "__main__":
    main()
