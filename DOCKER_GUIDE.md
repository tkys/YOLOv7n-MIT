# 🐳 YOLOv7n MIT Edition - Docker使用ガイド

CPU最適化されたDocker環境での包括的なMLOpsパイプライン

## 📋 目次

- [概要](#概要)
- [前提条件](#前提条件)
- [クイックスタート](#クイックスタート)
- [推論API使用方法](#推論api使用方法)
- [学習・評価環境](#学習評価環境)
- [量子化・最適化](#量子化最適化)
- [ログとモニタリング](#ログとモニタリング)
- [トラブルシューティング](#トラブルシューティング)

---

## 🎯 概要

このDocker環境では以下の機能を提供します：

### 🚀 推論機能
- **単一画像推論**: CPUで高速な単体画像処理
- **バッチ画像推論**: 複数画像の並列処理
- **REST API**: FastAPIベースの推論エンドポイント

### 🔧 モデル最適化
- **ONNX変換**: PyTorchモデルのONNX形式変換
- **INT8量子化**: モデルサイズ削減とCPU推論高速化
- **性能ベンチマーク**: 包括的な性能評価

### 📊 評価・分析
- **精度評価**: 検出成功率、mAP計算
- **速度評価**: 推論時間、FPS、スループット測定
- **リソース監視**: CPU・メモリ使用量分析
- **HTMLレポート**: 評価結果の可視化

---

## 🛠️ 前提条件

### システム要件
- Docker 20.10+
- Docker Compose 2.0+
- 最低4GB RAM
- CPU: 2コア以上推奨

### 準備するファイル
```
YOLO/
├── models/
│   └── yolov7n.onnx          # 学習済みONNXモデル
├── data/
│   └── test_images/          # テスト画像ディレクトリ
└── uploads/                  # API用アップロードディレクトリ
```

---

## 🚀 クイックスタート

### 1. Docker環境構築

```bash
# リポジトリクローン
git clone https://github.com/tkys/YOLOv7n-MIT.git
cd YOLOv7n-MIT

# Dockerイメージビルド
docker-compose build yolo-api

# 環境テスト
docker-compose run --rm yolo-api uv run python scripts/docker_test.py
```

### 2. 推論APIサーバー起動

```bash
# APIサーバー起動（バックグラウンド）
docker-compose up -d yolo-api

# ヘルスチェック
curl http://localhost:8000/health

# APIドキュメント確認
open http://localhost:8000/docs
```

### 3. 基本的な推論テスト

```bash
# 単一画像推論（CLI）
docker-compose run --rm yolo-api uv run python inference/single_image.py \
  --model models/yolov7n.onnx \
  --image data/test_images/sample.jpg

# バッチ推論（CLI）
docker-compose run --rm yolo-api uv run python inference/batch_images.py \
  --model models/yolov7n.onnx \
  --input data/test_images/ \
  --output results/batch_results
```

---

## 🌐 推論API使用方法

### REST API エンドポイント

#### 1. 単一画像推論
```bash
# cURLでの画像アップロード推論
curl -X POST "http://localhost:8000/predict/single" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg" \
  -F "confidence=0.5"
```

#### 2. バッチ画像推論
```bash
# 複数画像同時推論
curl -X POST "http://localhost:8000/predict/batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "parallel=true"
```

#### 3. Python クライアント例
```python
import requests

# 単一画像推論
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/single",
        files={"file": f},
        data={"confidence": 0.5}
    )
    
result = response.json()
print(f"検出数: {result['num_detections']}")
print(f"推論時間: {result['total_time']:.3f}s")
```

### レスポンス形式

```json
{
  "image_name": "test_image.jpg",
  "detections": [
    {
      "bbox": [100, 50, 300, 250],
      "confidence": 0.87,
      "class_id": 0
    }
  ],
  "num_detections": 1,
  "inference_time": 0.045,
  "total_time": 0.078
}
```

---

## 🎓 学習・評価環境

### 1. 学習環境の起動

```bash
# 学習用コンテナ起動（対話モード）
docker-compose run --rm yolo-train bash

# コンテナ内での学習実行
uv run python -m yolo.lazy train \
  data=mezzopiano \
  model=yolov7n \
  epochs=100 \
  device=cpu
```

### 2. モデル評価

```bash
# 包括的評価レポート生成
docker-compose run --rm yolo-api uv run python scripts/evaluation_report.py \
  --model models/yolov7n.onnx \
  --test-data data/test_images/ \
  --output analysis/evaluation_report

# HTMLレポート確認
open analysis/evaluation_report/evaluation_report.html
```

### 3. ベンチマーク実行

```bash
# 包括的ベンチマーク（すべてのモデル・データセット）
docker-compose up yolo-benchmark

# または手動実行
docker-compose run --rm yolo-api uv run python scripts/benchmark_suite.py \
  --models-dir models \
  --data-dir data \
  --output analysis/benchmark_results.json
```

---

## ⚙️ 量子化・最適化

### 1. モデル量子化

```bash
# 動的量子化（高速変換）
docker-compose run --rm yolo-api uv run python scripts/quantize_onnx.py \
  --model models/yolov7n.onnx \
  --output-dir models/quantized \
  --dynamic-only

# 静的量子化（高精度、キャリブレーション必要）
docker-compose run --rm yolo-api uv run python scripts/quantize_onnx.py \
  --model models/yolov7n.onnx \
  --output-dir models/quantized \
  --calibration-data data/calibration_images \
  --static-only
```

### 2. 量子化効果の確認

```bash
# 元モデルと量子化モデルの比較
docker-compose run --rm yolo-api uv run python scripts/quantize_onnx.py \
  --model models/yolov7n.onnx \
  --test-image data/test_images/sample.jpg \
  --output-dir models/quantized
```

#### 期待される結果例
```
動的量子化:
  圧縮比: 4.2x
  サイズ削減: 76.2%
  推論高速化: 1.8x
  精度保持: 98.5%
```

---

## 📊 ログとモニタリング

### 1. ログ確認

```bash
# アプリケーションログ
docker-compose run --rm yolo-api tail -f logs/api.log

# エラーログ
docker-compose run --rm yolo-api tail -f logs/api_error.log

# 推論ログ（JSON形式）
docker-compose run --rm yolo-api cat logs/inference.log | jq
```

### 2. リソースモニタリング

```bash
# リアルタイムリソース監視
docker-compose exec yolo-api uv run python -c "
import psutil
import time
while True:
    print(f'CPU: {psutil.cpu_percent():.1f}%, Memory: {psutil.virtual_memory().percent:.1f}%')
    time.sleep(1)
"
```

### 3. ログレベル調整

```bash
# 環境変数でログレベル設定
docker-compose run --rm -e LOG_LEVEL=DEBUG yolo-api uv run python scripts/docker_test.py
```

---

## 🔧 カスタマイズ

### 1. 信頼度閾値の調整

```bash
# APIサーバーの設定変更
docker-compose run --rm -e CONFIDENCE_THRESHOLD=0.3 yolo-api

# または実行時指定
curl -X POST "http://localhost:8000/predict/single" \
  -F "file=@image.jpg" \
  -F "confidence=0.3"
```

### 2. 並列処理数の調整

```yaml
# docker-compose.yml 編集
services:
  yolo-api:
    environment:
      - MAX_WORKERS=8  # 並列処理数
      - BATCH_SIZE=16  # バッチサイズ
```

### 3. メモリ制限設定

```yaml
# docker-compose.yml 編集
services:
  yolo-api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

---

## 🚨 トラブルシューティング

### よくある問題と解決策

#### 1. メモリ不足エラー
```bash
# メモリ使用量確認
docker stats

# バッチサイズ削減
docker-compose run --rm -e BATCH_SIZE=4 yolo-api
```

#### 2. モデルファイルが見つからない
```bash
# モデルファイル確認
docker-compose run --rm yolo-api ls -la models/

# ボリュームマウント確認
docker-compose config
```

#### 3. API応答が遅い
```bash
# CPU最適化設定確認
docker-compose run --rm yolo-api uv run python -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
"

# 並列処理数調整
docker-compose run --rm -e MAX_WORKERS=2 yolo-api
```

#### 4. 依存関係エラー
```bash
# 環境テスト実行
docker-compose run --rm yolo-api uv run python scripts/docker_test.py

# 依存関係再インストール
docker-compose build --no-cache yolo-api
```

### デバッグモード

```bash
# 詳細ログ出力
docker-compose run --rm -e LOG_LEVEL=DEBUG yolo-api

# 対話シェル起動
docker-compose run --rm yolo-api bash

# Python対話環境
docker-compose run --rm yolo-api uv run python
```

---

## 📚 さらなる情報

### ドキュメント
- [README_MIT_EDITION.md](README_MIT_EDITION.md) - 基本的な使用方法
- [scripts/](scripts/) - 各種スクリプトの詳細
- [inference/](inference/) - 推論機能の実装

### API仕様
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### パフォーマンステスト
```bash
# 負荷テスト（Apache Bench）
ab -n 100 -c 10 -T 'multipart/form-data; boundary=----WebKitFormBoundary' \
   -p test_image.jpg http://localhost:8000/predict/single
```

---

## 🤝 貢献

Issue報告や機能改善提案は [GitHub Issues](https://github.com/tkys/YOLOv7n-MIT/issues) でお願いします。

## 📜 ライセンス

MIT License - 詳細は [LICENSE](LICENSE) をご確認ください。

---

**🎉 Happy Inference! 効率的なDocker環境でのYOLO物体検出をお楽しみください！**