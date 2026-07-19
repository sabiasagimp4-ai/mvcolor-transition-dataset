# MVの色は時代とともに変化するか — データセット, 解析コード

Billboard Year-End Hot 100（2005〜2025年、各年上位75曲）のミュージックビデオ n=1509本から
色彩・編集ペースに関する特徴量を抽出し、年代トレンドを分析した研究の再現用データ・コード一式です。
コードの多くはClaude codeにより作成されたことをご留意ください。

高校の課題研究として実施したもので、本文（論文PDF/docx）はこのリポジトリには含みません。
ここには数値データと解析コードのみを置いています。また、実際の論文には載っていないMICD(Minimum Color Difference)を用いた場合のデータも含んでいます。

## 更新履歴

- **2026-07-19**: 年ごとの代表パレット作成（`year_palette.py`の`weighted_kmeans_palette_binned`、
  `year_palette_micd.py`の`weighted_kmeans_palette_lab_binned`）を修正。
  修正前は色相ビンを無視した単純加重KMeansで年パレットを作っていたため、画面占有面積の大きい
  肌色・暗色が代表パレットのクラスタを占有し、CEMD/MICDの計算元になるパレットそのものが
  偏っていた（レアな色相が代表パレットに反映されない）。ショット抽出時のBinned-KMeansが
  ビンごとに保証していた公平性を年単位の再クラスタリングでも維持するよう修正した結果、
  **CEMD（Skaff et al., CIEDE2000版, 対2005年）の結論が非有意（r=0.328, p=0.157）から
  有意（r=0.701, p=0.0006）に変わった**。`data/year_palette_cemd.csv`,
  `data/year_palette_cemd_stats.csv`, `data/year_palette_micd.csv`は修正後の値。

## 含まれていないもの

著作権・容量・セキュリティ上の理由で、以下は一切含みません。

- ダウンロードした動画ファイル本体（mp4）
- 動画から抽出したフレーム画像・shot単位の生ピクセルデータ
- YouTube Data API キー
- TransNetV2の学習済み重み（[原論文・公式実装](https://github.com/soCzech/TransNetV2)を参照してください）


## フォルダ構成

```
data/   最終集計CSV（すべて n=1509、TransNetV2版shot検出を使用した最終確定版）
code/   特徴量抽出・集計・統計・プロット用のPythonコード
```

## data/ データ辞書

### `videos.csv` — 分析対象動画マニフェスト（n=1575行、うち1509本が最終分析対象）
| 列名 | 内容 |
|---|---|
| id | `<年>_<Billboard順位2桁>` 形式のID |
| title, artist | 曲名・アーティスト名 |
| year, rank | Billboard Year-End Hot 100の年・順位 |
| yt_id, yt_url | 対応するYouTube動画のID・URL |
| yt_title, yt_channel | YouTube側のタイトル・チャンネル名 |
| verified | 目視確認フラグ（空欄は未実施） |

### `excluded_non_official_videos.csv` — 分析から除外した動画とその理由
非公式アップロード・セッション動画等、公式MVとして不適切と判定され除外された動画のリストと除外理由（`reason_ja`, `reason_raw`）。

### `video_summary.csv` — 動画単位の特徴量（n=1509、論文の全統計量の一次データ）
| 列名 | 内容 |
|---|---|
| id, year, rank, artist, title | 動画識別情報 |
| n_shots, total_dur_s | 検出されたショット数・動画全体の長さ（秒） |
| mean_s, mean_v | HSV色空間での平均彩度・平均明度（各shotの代表色の重み付き平均） |
| mean_s_std, mean_v_std | 同・年内での標準偏差ではなく動画内shot間の標準偏差 |
| colorfulness | Hasler & Süsstrunk (2003) のcolorfulness指標 |
| pct_chromatic | 無彩色（グレー・白・黒）ではない有彩色shotの割合 |
| warm_cool_ratio | 暖色shot数 / 寒色shot数 |
| hue_entropy | 色相分布のシャノンエントロピー（色相の多様性） |
| hue_red 〜 hue_purple | 7色相ビンそれぞれの出現割合 |
| avg_shot_dur | 平均ショット長（秒） |
| pct_fast_cut | 1秒未満の高速カットの割合 |
| mean_L_lab, mean_C_lab | CIE Lab色空間での平均明度L*・平均彩度C* |
| color_dynamism, color_dynamism_std, color_dynamism_max | 隣接shot間のCIEDE2000色差ΔEの平均・標準偏差・最大値（動画内でどれだけ色が動くか） |

### `year_summary.csv` — 年単位の記述統計
`video_summary.csv`の各`dim`（列）について、年ごとのn・平均・標準偏差・中央値・四分位を集計したもの。

### `regression_stats.csv` — 年次トレンドの回帰分析結果一覧（論文「結果」節の親表）
| 列名 | 内容 |
|---|---|
| dim | 対象の特徴量名（`video_summary.csv`の列名に対応） |
| n | 回帰に使用したサンプル数（欠損値を除く） |
| slope_per_year | 年あたりの変化量（線形回帰の傾き） |
| r | ピアソンの相関係数 |
| p_value | 対応するp値 |
| significant_p<0.05 | p<0.05かどうかの真偽値 |

### `year_palette_cemd.csv` / `year_palette_cemd_stats.csv` — CEMD（色パレット間距離）
Skaff et al. (2011) のCEMD（Color-based Earth Mover's Distance、CIEDE2000版）を用いて、
各年の代表カラーパレット（k=8のk-meansクラスタ）と2005年（基準年）・前年との距離を計算したもの。
`cemd_vs_base_year`が論文の正式な報告値。基準年の自明点（距離0）は回帰から除外している。

### `year_palette_micd.csv` — MICD（参考値、本文では採用していない手法）
Pocol, Istead & Kaplan (2025) のMICD（CIE76ベース）。分布全体を見ない弱点があり、
最終的にCEMDへ乗り換えた経緯の参考値として同梱（論文の結果として報告していない）。

### `year_hue_hist.csv` — 年ごとの色相分布（7ビン）
### `color_dynamism_position.csv` — 動画内の再生位置（0〜1に正規化）ごとの色変化量
### `color_dynamism_backloaded_ratio.csv` — 動画後半/前半の色変化量の比率（`back_loaded_ratio`）
### `affective_pad.csv` / `affective_pad_stats.csv` — Pleasure-Arousal-Dominance感情モデルへの変換値と年次トレンド（探索的分析、論文の主結果には含めていない）
### `video_quality.csv` — 動画の解像度・ビットレート・ファイルサイズ（画質が年代トレンドの交絡要因になっていないかの確認用、`check_quality_confound.py`が使用）

## code/ 処理の流れ

1. **ショット境界検出**: `shot_cut_detection.py`（TransNetV2 (Souček & Lokoč, 2020) 版・HSVヒストグラム版の両方を実装。最終結果はTransNetV2版を採用）
2. **色特徴量抽出**: `shot_dominant_color_binned_kmeans.py`（各ショットの代表色をbinned k-meansで抽出）
3. **オーケストレーション**: `extract_pipeline.py`（1・2を全動画に対して実行、レジューム可能）
4. **動画単位の集計**: `aggregate_video.py`, `aggregate_video_extra.py`
5. **年単位の集計**: `aggregate_year.py`
6. **年代別パレット距離**: `year_palette.py`（共通処理）→ `year_palette_cemd.py`（CEMD, 正式値）/ `year_palette_micd.py`（MICD, 参考値）
7. **感情モデル変換**: `affective_pad.py`
8. **色変化の時間的分布**: `color_dynamism_position.py`
9. **画質交絡チェック**: `check_quality_confound.py`
10. **プロット**: `plot_trends.py`, `plot_cemd_trend.py`

`run_transnetv2_shots.py`は`shot_cut_detection.py`のTransNetV2モードを呼び出すラッパー。
TransNetV2本体（学習済み重み含む）は同梱していないため、実行するには
[soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) を別途セットアップしてください。

動画ファイル自体・抽出フレームは含まれていないため、1〜3のコードはそのままでは実行できません
（`data/video_summary.csv`以降は`video_summary.csv`さえあれば再現可能です）。

## 主な参考文献

参考文献に関しては、元論文をご参照ください。

## ライセンス

- コード（`code/`）: MIT License（`LICENSE`参照）
- データ（`data/`）: CC BY 4.0（動画そのものは含まれていません。派生数値データのみです）

## 引用

このデータを使う場合は、対応する論文（準備中）またはこのリポジトリを引用してください。
