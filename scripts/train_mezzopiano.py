#!/usr/bin/env python3
"""
Mezzopiano Transfer Learning Script for YOLOv7n MIT
"""
import os
import sys
import time
import torch
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"mezzopiano_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_dataset_integrity():
    """データセットの整合性確認"""
    dataset_path = Path("/home/tkys/playground/signage-narumiya/pre-20250701/data/yolo_dataset_fixed")
    
    logger.info("=== データセット整合性確認開始 ===")
    
    # 基本パス確認
    train_images = dataset_path / "images" / "train"
    train_labels = dataset_path / "labels" / "train"
    val_images = dataset_path / "images" / "val"
    val_labels = dataset_path / "labels" / "val"
    
    paths = [train_images, train_labels, val_images, val_labels]
    for path in paths:
        if not path.exists():
            logger.error(f"パスが存在しません: {path}")
            return False
    
    # 画像・ラベル数確認（.pngファイルを検索）
    train_img_count = len(list(train_images.glob("*.png")))
    train_lbl_count = len(list(train_labels.glob("*.txt")))
    val_img_count = len(list(val_images.glob("*.png")))
    val_lbl_count = len(list(val_labels.glob("*.txt")))
    
    logger.info(f"学習データ: 画像 {train_img_count}, ラベル {train_lbl_count}")
    logger.info(f"検証データ: 画像 {val_img_count}, ラベル {val_lbl_count}")
    
    if train_img_count != train_lbl_count:
        logger.error(f"学習データの画像・ラベル数不一致: {train_img_count} vs {train_lbl_count}")
        return False
    
    if val_img_count != val_lbl_count:
        logger.error(f"検証データの画像・ラベル数不一致: {val_img_count} vs {val_lbl_count}")
        return False
    
    logger.info("✓ データセット整合性確認完了")
    return True

def check_environment():
    """学習環境確認"""
    logger.info("=== 学習環境確認開始 ===")
    
    # PyTorch確認
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA使用可能: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU{i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CPU学習モードで実行")
    
    # メモリ確認
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"システムメモリ: {memory.total // (1024**3)}GB (使用可能: {memory.available // (1024**3)}GB)")
    except ImportError:
        logger.warning("psutilがインストールされていません")
    
    logger.info("✓ 学習環境確認完了")
    return True

def prepare_training_environment():
    """学習環境準備"""
    logger.info("=== 学習環境準備開始 ===")
    
    # 出力ディレクトリ作成
    output_dirs = ["runs", "runs/train", "runs/val", "checkpoints"]
    for dir_name in output_dirs:
        Path(dir_name).mkdir(exist_ok=True)
        logger.info(f"ディレクトリ作成: {dir_name}")
    
    # 事前訓練済みモデル確認
    pretrained_dir = Path("pretrained_weights")
    if pretrained_dir.exists():
        weights = list(pretrained_dir.glob("*.pt"))
        if weights:
            logger.info(f"事前訓練済みモデル発見: {[w.name for w in weights]}")
        else:
            logger.warning("事前訓練済みモデルが見つかりません")
    
    logger.info("✓ 学習環境準備完了")
    return True

def run_training():
    """メイン学習処理"""
    logger.info("=== YOLOv7n Mezzopiano学習開始 ===")
    
    # 学習コマンド構築（GPU使用）
    train_cmd = [
        "python", "yolo/lazy.py",
        "task=train",
        "dataset=mezzopiano",
        "model=v9-s",  # より軽量なモデル
        "device=0",  # GPU 0使用
        "task.epoch=30",  # エポック数削減
        "task.data.batch_size=8",  # GPU用にバッチサイズ増加
        "task.optimizer.args.lr=0.001",
        "name=mezzopiano_gpu",
        "use_wandb=false"  # Wandb無効化
    ]
    
    # 事前訓練済みモデルは互換性の問題でスキップ
    # pretrained_path = Path("pretrained_weights/yolov9-c.pt")
    # if pretrained_path.exists():
    #     train_cmd.extend([f"weight={pretrained_path}"])
    #     logger.info(f"事前訓練済みモデル使用: {pretrained_path}")
    logger.info("スクラッチ学習（事前訓練モデルなし）で実行")
    
    logger.info(f"実行コマンド: {' '.join(train_cmd)}")
    
    # 学習実行
    import subprocess
    try:
        start_time = time.time()
        
        # 実際の学習実行
        result = subprocess.run(
            train_cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=7200  # 2時間タイムアウト
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"学習時間: {training_time:.2f}秒 ({training_time/60:.1f}分)")
        
        if result.returncode == 0:
            logger.info("✓ 学習完了")
            logger.info("標準出力:")
            logger.info(result.stdout)
        else:
            logger.error("✗ 学習失敗")
            logger.error("エラー出力:")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("学習がタイムアウトしました")
        return False
    except Exception as e:
        logger.error(f"学習実行中にエラー: {e}")
        return False
    
    return True

def validate_training_results():
    """学習結果検証"""
    logger.info("=== 学習結果検証開始 ===")
    
    # 出力ファイル確認
    runs_dir = Path("runs")
    if not runs_dir.exists():
        logger.error("学習結果ディレクトリが見つかりません")
        return False
    
    # 最新の学習結果を探す
    train_dirs = list(runs_dir.glob("train*"))
    if not train_dirs:
        logger.error("学習結果が見つかりません")
        return False
    
    latest_train_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
    logger.info(f"最新の学習結果: {latest_train_dir}")
    
    # 重要ファイル確認
    important_files = ["best.pt", "last.pt", "results.csv"]
    for file_name in important_files:
        file_path = latest_train_dir / file_name
        if file_path.exists():
            logger.info(f"✓ {file_name} 存在確認")
        else:
            logger.warning(f"✗ {file_name} が見つかりません")
    
    logger.info("✓ 学習結果検証完了")
    return True

def main():
    """メイン処理"""
    logger.info("=== Mezzopiano YOLOv7n Transfer Learning 開始 ===")
    
    # 段階的処理実行
    steps = [
        ("データセット整合性確認", check_dataset_integrity),
        ("学習環境確認", check_environment),
        ("学習環境準備", prepare_training_environment),
        ("学習実行", run_training),
        ("学習結果検証", validate_training_results)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n--- {step_name} ---")
        try:
            success = step_func()
            if not success:
                logger.error(f"{step_name}が失敗しました")
                return False
        except Exception as e:
            logger.error(f"{step_name}中にエラー: {e}")
            return False
    
    logger.info("\n=== Mezzopiano YOLOv7n Transfer Learning 完了 ===")
    logger.info(f"ログファイル: {log_file}")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)