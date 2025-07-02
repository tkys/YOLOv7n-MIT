#!/usr/bin/env python3
"""
シンプルなYOLO学習実行スクリプト
ログエラーを回避して確実に学習完了
"""
import subprocess
import sys
import os
import time
from pathlib import Path

def run_simple_training():
    """シンプル学習実行"""
    print("=== シンプルYOLO学習開始 ===")
    
    # 環境変数設定（ログエラー対策）
    env = os.environ.copy()
    env['HYDRA_FULL_ERROR'] = '1'
    env['PYTORCH_DISABLE_TYPED_STORAGE'] = '1'
    
    # 最小構成コマンド
    cmd = [
        "python", "yolo/lazy.py",
        "task=train",
        "dataset=mezzopiano",
        "model=v9-s", 
        "device=cpu",
        "task.epoch=3",  # 最小エポック数
        "task.data.batch_size=2",  # 最小バッチサイズ
        "task.data.cpu_num=1",  # 最小ワーカー数
        "task.optimizer.args.lr=0.01",
        "use_wandb=false",
        "use_tensorboard=false",
        "name=simple_test"
    ]
    
    print(f"実行コマンド: {' '.join(cmd)}")
    
    # 学習実行
    try:
        print("学習開始...")
        start_time = time.time()
        
        # リアルタイム出力で実行
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )
        
        # リアルタイム出力表示
        for line in process.stdout:
            print(line.rstrip())
            
        process.wait()
        end_time = time.time()
        
        print(f"\n学習時間: {(end_time - start_time)/60:.1f}分")
        print(f"終了コード: {process.returncode}")
        
        if process.returncode == 0:
            print("✅ 学習完了!")
        else:
            print("❌ 学習失敗")
            
        return process.returncode == 0
        
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        return False

def check_results():
    """結果確認"""
    print("\n=== 学習結果確認 ===")
    
    # 最新の学習結果ディレクトリを探す
    runs_dir = Path("runs/train")
    if runs_dir.exists():
        train_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if train_dirs:
            latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
            print(f"最新学習結果: {latest_dir}")
            
            # 重要ファイル確認
            important_files = ["best.pt", "last.pt", "results.txt", "output.log"]
            for file_name in important_files:
                file_path = latest_dir / file_name
                if file_path.exists():
                    print(f"✅ {file_name}: {file_path.stat().st_size} bytes")
                else:
                    print(f"❌ {file_name}: 見つからない")
        else:
            print("❌ 学習結果ディレクトリなし")
    else:
        print("❌ runs/trainディレクトリなし")

if __name__ == "__main__":
    success = run_simple_training()
    check_results()
    
    if success:
        print("\n🎉 Phase 2 Transfer Learning 完了!")
    else:
        print("\n⚠️ 学習に問題が発生しました")