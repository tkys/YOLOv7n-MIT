#!/usr/bin/env python3
"""
最適化されたYOLO学習スクリプト
CPU/GPU自動判定、安定設定
"""
import subprocess
import sys
import time
import torch
from pathlib import Path

def check_gpu_availability():
    """GPU利用可能性確認"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU利用可能: {torch.cuda.device_count()}台")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU{i}: {torch.cuda.get_device_name(i)}")
            return True, 0  # GPU 0使用
        else:
            print("⚠️ GPU利用不可、CPU学習実行")
            return False, "cpu"
    except Exception as e:
        print(f"⚠️ GPU確認エラー: {e}, CPU学習実行")
        return False, "cpu"

def run_optimized_training():
    """最適化学習実行"""
    print("=== 最適化YOLO学習開始 ===")
    
    # GPU/CPU自動判定
    gpu_available, device = check_gpu_availability()
    
    # 設定調整
    if gpu_available:
        batch_size = 16  # GPU用大きなバッチ
        epochs = 50      # GPU高速学習
        workers = 8
    else:
        batch_size = 4   # CPU用小さなバッチ
        epochs = 20      # CPU用短期学習
        workers = 2
    
    # 学習コマンド構築
    cmd = [
        "python", "yolo/lazy.py",
        "task=train",
        "dataset=mezzopiano", 
        "model=v9-s",
        f"device={device}",
        f"task.epoch={epochs}",
        f"task.data.batch_size={batch_size}",
        f"task.data.cpu_num={workers}",
        "task.optimizer.args.lr=0.001",
        "use_wandb=false",
        "name=mezzopiano_optimized"
    ]
    
    print(f"実行コマンド: {' '.join(cmd)}")
    print(f"設定: device={device}, batch={batch_size}, epochs={epochs}")
    
    # 学習実行
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        end_time = time.time()
        
        print(f"\n学習時間: {(end_time - start_time)/60:.1f}分")
        
        if result.returncode == 0:
            print("✅ 学習完了!")
            print("=== 標準出力 ===")
            print(result.stdout[-2000:])  # 最後の2000文字
        else:
            print("❌ 学習失敗")
            print("=== エラー出力 ===") 
            print(result.stderr[-1000:])  # 最後の1000文字
            
    except subprocess.TimeoutExpired:
        print("⏰ 学習タイムアウト (1時間)")
    except Exception as e:
        print(f"❌ 実行エラー: {e}")

if __name__ == "__main__":
    run_optimized_training()