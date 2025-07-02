# YOLOv7n MIT Edition - Production Docker Image
FROM python:3.10-slim

# システム依存関係のインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# uv のインストール
RUN pip install uv

# 依存関係ファイルのコピー
COPY pyproject.toml uv.lock ./

# CPU最適化版の依存関係インストール
RUN uv sync --no-dev

# アプリケーションコードのコピー
COPY . .

# 必要なディレクトリの作成
RUN mkdir -p /app/logs /app/uploads /app/models /app/results

# 環境変数の設定
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO
ENV MODEL_PATH=/app/models
ENV UPLOAD_PATH=/app/uploads
ENV RESULTS_PATH=/app/results

# ポート公開
EXPOSE 8000

# デフォルトコマンド（FastAPI サーバー起動）
CMD ["uv", "run", "uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000"]