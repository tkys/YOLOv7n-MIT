"""
YOLOv7n MIT Edition - FastAPI推論サーバー
REST APIエンドポイントでの推論サービス提供
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .batch_images import BatchImageInference
from .logger import api_logger
from .single_image import SingleImageInference

# レスポンスモデル定義
class DetectionResult(BaseModel):
    bbox: List[int]
    confidence: float
    class_id: int

class SingleInferenceResponse(BaseModel):
    image_name: str
    detections: List[DetectionResult]
    num_detections: int
    inference_time: float
    total_time: float

class BatchInferenceResponse(BaseModel):
    num_images: int
    results: List[SingleInferenceResponse]
    statistics: Dict
    processing_mode: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime: float

# FastAPIアプリケーション
app = FastAPI(
    title="YOLOv7n MIT Edition API",
    description="高性能物体検出API（CPU最適化ONNX Runtime）",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数
single_inferencer: Optional[SingleImageInference] = None
batch_inferencer: Optional[BatchImageInference] = None
app_start_time = time.time()

# 設定
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov7n.onnx")
UPLOAD_DIR = Path(os.getenv("UPLOAD_PATH", "uploads"))
RESULTS_DIR = Path(os.getenv("RESULTS_PATH", "results"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB

# ディレクトリ作成
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    global single_inferencer, batch_inferencer
    
    try:
        api_logger.info("FastAPI推論サーバー起動開始")
        
        # モデルファイルの存在確認
        if not Path(MODEL_PATH).exists():
            api_logger.error(f"モデルファイルが見つかりません: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        # 推論器初期化
        single_inferencer = SingleImageInference(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        batch_inferencer = BatchImageInference(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            max_workers=2  # API使用時は控えめに設定
        )
        
        api_logger.info(
            "FastAPI推論サーバー起動完了",
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
    except Exception as e:
        api_logger.critical(
            "FastAPI起動エラー",
            error=str(e)
        )
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """ヘルスチェックエンドポイント"""
    uptime = time.time() - app_start_time
    
    return HealthResponse(
        status="healthy",
        model_loaded=single_inferencer is not None,
        uptime=uptime
    )

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "YOLOv7n MIT Edition API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "single_inference": "/predict/single",
            "batch_inference": "/predict/batch",
            "docs": "/docs"
        }
    }

@app.post("/predict/single", response_model=SingleInferenceResponse)
async def predict_single_image(
    file: UploadFile = File(...),
    confidence: Optional[float] = None
):
    """
    単一画像推論エンドポイント
    
    Args:
        file: アップロード画像ファイル
        confidence: 信頼度閾値（オプション）
    
    Returns:
        推論結果
    """
    if single_inferencer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="推論器が初期化されていません"
        )
    
    # ファイルサイズチェック
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"ファイルサイズが大きすぎます (最大: {MAX_FILE_SIZE} bytes)"
        )
    
    # ファイル形式チェック
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="画像ファイルをアップロードしてください"
        )
    
    try:
        # 一時ファイル保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        api_logger.info(
            "単一画像推論開始",
            filename=file.filename,
            file_size=len(content),
            confidence=confidence or CONFIDENCE_THRESHOLD
        )
        
        # 信頼度閾値の動的変更
        if confidence is not None:
            single_inferencer.confidence_threshold = confidence
        
        # 推論実行
        result = single_inferencer.predict(tmp_file_path)
        
        # レスポンス構築
        detections = [
            DetectionResult(**detection) for detection in result["detections"]
        ]
        
        response = SingleInferenceResponse(
            image_name=file.filename,
            detections=detections,
            num_detections=result["num_detections"],
            inference_time=result["timing"]["inference_time"],
            total_time=result["timing"]["total_time"]
        )
        
        api_logger.info(
            "単一画像推論完了",
            filename=file.filename,
            num_detections=result["num_detections"],
            total_time=result["timing"]["total_time"]
        )
        
        return response
        
    except Exception as e:
        api_logger.error(
            "単一画像推論エラー",
            filename=file.filename,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"推論処理エラー: {str(e)}"
        )
    
    finally:
        # 一時ファイル削除
        try:
            os.unlink(tmp_file_path)
        except:
            pass

@app.post("/predict/batch", response_model=BatchInferenceResponse)
async def predict_batch_images(
    files: List[UploadFile] = File(...),
    confidence: Optional[float] = None,
    parallel: bool = True
):
    """
    バッチ画像推論エンドポイント
    
    Args:
        files: アップロード画像ファイルリスト
        confidence: 信頼度閾値（オプション）
        parallel: 並列処理フラグ
    
    Returns:
        バッチ推論結果
    """
    if batch_inferencer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="バッチ推論器が初期化されていません"
        )
    
    # ファイル数制限
    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="一度にアップロードできるファイル数は50個までです"
        )
    
    tmp_file_paths = []
    
    try:
        # 一時ファイル保存
        for file in files:
            # ファイルサイズチェック
            if file.size and file.size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"ファイル '{file.filename}' のサイズが大きすぎます"
                )
            
            # ファイル形式チェック
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"'{file.filename}' は画像ファイルではありません"
                )
            
            # 一時ファイル作成
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_paths.append(tmp_file.name)
        
        api_logger.info(
            "バッチ画像推論開始",
            num_files=len(files),
            parallel=parallel,
            confidence=confidence or CONFIDENCE_THRESHOLD
        )
        
        # 信頼度閾値の動的変更
        if confidence is not None:
            for worker in batch_inferencer.inference_workers:
                worker.confidence_threshold = confidence
        
        # バッチ推論実行
        if parallel:
            results = batch_inferencer.predict_batch_parallel(tmp_file_paths)
        else:
            results = batch_inferencer.predict_batch_sequential(tmp_file_paths)
        
        # 統計計算
        total_time = sum(r.get("timing", {}).get("total_time", 0) for r in results if "error" not in r)
        stats = batch_inferencer.calculate_batch_statistics(results, total_time)
        
        # レスポンス構築
        response_results = []
        for i, result in enumerate(results):
            if "error" in result:
                # エラー結果
                response_results.append(
                    SingleInferenceResponse(
                        image_name=files[i].filename,
                        detections=[],
                        num_detections=0,
                        inference_time=0,
                        total_time=0
                    )
                )
            else:
                # 正常結果
                detections = [
                    DetectionResult(**detection) for detection in result["detections"]
                ]
                response_results.append(
                    SingleInferenceResponse(
                        image_name=files[i].filename,
                        detections=detections,
                        num_detections=result["num_detections"],
                        inference_time=result["timing"]["inference_time"],
                        total_time=result["timing"]["total_time"]
                    )
                )
        
        response = BatchInferenceResponse(
            num_images=len(files),
            results=response_results,
            statistics=stats,
            processing_mode="parallel" if parallel else "sequential"
        )
        
        api_logger.info(
            "バッチ画像推論完了",
            num_files=len(files),
            successful_files=stats["successful_images"],
            total_detections=stats["total_detections"],
            total_time=stats["total_time"]
        )
        
        return response
        
    except Exception as e:
        api_logger.error(
            "バッチ画像推論エラー",
            num_files=len(files),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"バッチ推論処理エラー: {str(e)}"
        )
    
    finally:
        # 一時ファイル削除
        for tmp_path in tmp_file_paths:
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """グローバル例外ハンドラー"""
    api_logger.error(
        "API未処理例外",
        path=str(request.url),
        method=request.method,
        error=str(exc)
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "サーバー内部エラーが発生しました"}
    )

def main():
    """開発サーバー起動"""
    uvicorn.run(
        "inference.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()