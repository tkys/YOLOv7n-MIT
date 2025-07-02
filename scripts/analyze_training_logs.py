#!/usr/bin/env python3
"""
学習過程分析・ロギングスクリプト
Loss推移、精度変化、学習率などを詳細に分析・可視化
"""
import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def parse_lightning_logs(log_dir):
    """PyTorch Lightning ログから詳細メトリクス抽出"""
    print(f"=== ログ解析開始: {log_dir} ===")
    
    metrics = {
        'epoch': [],
        'train_loss': [],
        'box_loss': [],
        'dfl_loss': [],
        'bce_loss': [],
        'learning_rate': [],
        'val_ap_50': [],
        'val_ap_50_95': [],
        'val_recall': [],
        'timestamp': []
    }
    
    # ログファイル探索
    log_files = []
    for ext in ['*.log', '*.txt', 'events.out.tfevents*']:
        log_files.extend(list(Path(log_dir).rglob(ext)))
    
    print(f"発見ログファイル数: {len(log_files)}")
    
    # 各ログファイルを解析
    for log_file in log_files:
        try:
            print(f"解析中: {log_file}")
            parse_single_log(log_file, metrics)
        except Exception as e:
            print(f"⚠️ ログファイル解析エラー {log_file}: {e}")
    
    return metrics

def parse_single_log(log_file, metrics):
    """単一ログファイルの解析"""
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # エポック情報抽出
    epoch_pattern = r'Epoch\s+(\d+)'
    epochs = re.findall(epoch_pattern, content)
    
    # Loss値抽出
    loss_patterns = {
        'BoxLoss': r'BoxLoss[^\d]*([0-9\.]+)',
        'DFLLoss': r'DFLLoss[^\d]*([0-9\.]+)', 
        'BCELoss': r'BCELoss[^\d]*([0-9\.]+)'
    }
    
    for loss_type, pattern in loss_patterns.items():
        values = re.findall(pattern, content)
        print(f"  {loss_type}: {len(values)}個の値を発見")
    
    # AP値抽出（表形式）
    ap_pattern = r'AP @\s+\.5.*?(\d+\.?\d*)'
    ap_values = re.findall(ap_pattern, content)
    print(f"  AP@0.5値: {len(ap_values)}個発見")
    
    return metrics

def extract_tensorboard_logs(log_dir):
    """TensorBoard形式ログ抽出"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        tb_logs = list(Path(log_dir).rglob('events.out.tfevents*'))
        if not tb_logs:
            print("TensorBoardログが見つかりません")
            return {}
            
        print(f"TensorBoardログファイル: {len(tb_logs)}個")
        
        metrics = {}
        for tb_log in tb_logs:
            ea = event_accumulator.EventAccumulator(str(tb_log))
            ea.Reload()
            
            # 利用可能なメトリクス取得
            scalars = ea.Tags()['scalars']
            print(f"利用可能メトリクス: {scalars}")
            
            for scalar in scalars:
                scalar_events = ea.Scalars(scalar)
                metrics[scalar] = {
                    'steps': [e.step for e in scalar_events],
                    'values': [e.value for e in scalar_events],
                    'timestamps': [e.wall_time for e in scalar_events]
                }
        
        return metrics
        
    except ImportError:
        print("TensorBoard未インストール、スキップ")
        return {}

def parse_checkpoint_info(checkpoint_dir):
    """チェックポイントファイルから学習状態分析"""
    print(f"\n=== チェックポイント分析: {checkpoint_dir} ===")
    
    checkpoint_files = list(Path(checkpoint_dir).glob('*.ckpt'))
    if not checkpoint_files:
        print("チェックポイントファイルが見つかりません")
        return {}
    
    print(f"チェックポイントファイル: {len(checkpoint_files)}個")
    
    try:
        import torch
        
        info = {}
        for ckpt_file in checkpoint_files:
            print(f"解析中: {ckpt_file.name}")
            
            try:
                checkpoint = torch.load(ckpt_file, map_location='cpu')
                
                info[ckpt_file.name] = {
                    'epoch': checkpoint.get('epoch', 'N/A'),
                    'global_step': checkpoint.get('global_step', 'N/A'),
                    'lr_schedulers': len(checkpoint.get('lr_schedulers', [])),
                    'optimizer_states': len(checkpoint.get('optimizer_states', [])),
                    'model_size_mb': ckpt_file.stat().st_size / (1024*1024),
                    'keys': list(checkpoint.keys())[:10]  # 最初の10キー
                }
                
            except Exception as e:
                print(f"  ⚠️ チェックポイント読み込みエラー: {e}")
        
        return info
        
    except ImportError:
        print("PyTorch未インストール、スキップ")
        return {}

def create_training_visualization(metrics, output_dir):
    """学習過程可視化"""
    print(f"\n=== 可視化作成: {output_dir} ===")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # データが空の場合のハンドリング
    if not any(metrics.values()):
        print("⚠️ 可視化用データが不足しています")
        return
    
    # 1. Loss推移グラフ
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    loss_types = ['box_loss', 'dfl_loss', 'bce_loss']
    for loss_type in loss_types:
        if metrics.get(loss_type):
            plt.plot(metrics[loss_type], label=loss_type.replace('_', ' ').title())
    plt.title('Loss推移')
    plt.xlabel('Step/Epoch')
    plt.ylabel('Loss値')
    plt.legend()
    plt.grid(True)
    
    # 2. 学習率推移
    plt.subplot(2, 3, 2)
    if metrics.get('learning_rate'):
        plt.plot(metrics['learning_rate'], label='Learning Rate', color='red')
        plt.title('学習率推移')
        plt.xlabel('Step/Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
    
    # 3. 精度推移
    plt.subplot(2, 3, 3)
    ap_metrics = ['val_ap_50', 'val_ap_50_95']
    for ap_metric in ap_metrics:
        if metrics.get(ap_metric):
            plt.plot(metrics[ap_metric], label=ap_metric.replace('_', ' ').upper())
    plt.title('精度推移')
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.legend()
    plt.grid(True)
    
    # 4. 総合Loss
    plt.subplot(2, 3, 4)
    if metrics.get('train_loss'):
        plt.plot(metrics['train_loss'], label='Train Loss', color='blue')
        plt.title('総合Loss推移')
        plt.xlabel('Step/Epoch')
        plt.ylabel('Total Loss')
        plt.grid(True)
    
    # 5. Recall推移
    plt.subplot(2, 3, 5)
    if metrics.get('val_recall'):
        plt.plot(metrics['val_recall'], label='Validation Recall', color='green')
        plt.title('Recall推移')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.grid(True)
    
    # 6. サマリー統計
    plt.subplot(2, 3, 6)
    summary_data = []
    summary_labels = []
    
    for metric_name, values in metrics.items():
        if values and isinstance(values[0], (int, float)):
            summary_data.append([np.min(values), np.mean(values), np.max(values)])
            summary_labels.append(metric_name.replace('_', ' ').title())
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data, 
                                columns=['Min', 'Mean', 'Max'],
                                index=summary_labels)
        
        # ヒートマップ表示
        sns.heatmap(summary_df.T, annot=True, fmt='.3f', cmap='viridis')
        plt.title('メトリクス統計サマリー')
    
    plt.tight_layout()
    
    # 保存
    viz_path = output_dir / 'training_analysis.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 可視化保存: {viz_path}")

def generate_training_report(metrics, checkpoint_info, output_dir):
    """詳細学習レポート生成"""
    print(f"\n=== 学習レポート生成: {output_dir} ===")
    
    output_dir = Path(output_dir)
    report_path = output_dir / 'training_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# YOLOv7n Mezzopiano学習分析レポート\n\n")
        f.write(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # メトリクス概要
        f.write("## 📊 学習メトリクス概要\n\n")
        for metric_name, values in metrics.items():
            if values and isinstance(values[0], (int, float)):
                f.write(f"### {metric_name.replace('_', ' ').title()}\n")
                f.write(f"- **データ数**: {len(values)}\n")
                f.write(f"- **最小値**: {np.min(values):.4f}\n")
                f.write(f"- **平均値**: {np.mean(values):.4f}\n")
                f.write(f"- **最大値**: {np.max(values):.4f}\n")
                f.write(f"- **最終値**: {values[-1]:.4f}\n\n")
        
        # チェックポイント情報
        f.write("## 💾 チェックポイント情報\n\n")
        for ckpt_name, info in checkpoint_info.items():
            f.write(f"### {ckpt_name}\n")
            f.write(f"- **エポック**: {info.get('epoch', 'N/A')}\n")
            f.write(f"- **グローバルステップ**: {info.get('global_step', 'N/A')}\n")
            f.write(f"- **ファイルサイズ**: {info.get('model_size_mb', 0):.1f} MB\n")
            f.write(f"- **オプティマイザー状態**: {info.get('optimizer_states', 0)}個\n\n")
        
        # 学習成果サマリー
        f.write("## 🎯 学習成果サマリー\n\n")
        if metrics.get('val_ap_50'):
            final_ap50 = metrics['val_ap_50'][-1]
            f.write(f"- **最終AP@0.5**: {final_ap50:.3f} ({final_ap50*100:.1f}%)\n")
        
        if metrics.get('val_ap_50_95'):
            final_ap50_95 = metrics['val_ap_50_95'][-1]
            f.write(f"- **最終AP@0.5:0.95**: {final_ap50_95:.3f} ({final_ap50_95*100:.1f}%)\n")
        
        if metrics.get('train_loss'):
            initial_loss = metrics['train_loss'][0]
            final_loss = metrics['train_loss'][-1]
            loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
            f.write(f"- **Loss削減**: {initial_loss:.3f} → {final_loss:.3f} ({loss_reduction:.1f}%削減)\n")
    
    print(f"✅ レポート保存: {report_path}")

def main():
    """メイン実行"""
    print("🔍 YOLOv7n学習過程詳細分析開始")
    
    # 最新の学習結果を特定
    runs_dir = Path("runs/train")
    if not runs_dir.exists():
        print("❌ runs/trainディレクトリが見つかりません")
        return
    
    train_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not train_dirs:
        print("❌ 学習結果ディレクトリが見つかりません")
        return
    
    # 最新ディレクトリ（GPU Fine-tuning優先）
    gpu_dirs = [d for d in train_dirs if 'gpu' in d.name.lower()]
    if gpu_dirs:
        latest_dir = max(gpu_dirs, key=lambda x: x.stat().st_mtime)
    else:
        latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
    
    print(f"📂 分析対象: {latest_dir}")
    
    # 出力ディレクトリ作成
    analysis_dir = Path("analysis") / latest_dir.name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ログ解析
    metrics = parse_lightning_logs(latest_dir)
    
    # 2. TensorBoardログ（あれば）
    tb_metrics = extract_tensorboard_logs(latest_dir)
    
    # 3. チェックポイント分析
    checkpoint_dir = latest_dir / "checkpoints"
    checkpoint_info = parse_checkpoint_info(checkpoint_dir)
    
    # 4. 可視化作成
    create_training_visualization(metrics, analysis_dir)
    
    # 5. レポート生成
    generate_training_report(metrics, checkpoint_info, analysis_dir)
    
    print(f"\n🎉 学習分析完了！")
    print(f"📁 出力ディレクトリ: {analysis_dir}")
    print(f"📊 可視化: {analysis_dir}/training_analysis.png")
    print(f"📄 レポート: {analysis_dir}/training_report.md")

if __name__ == "__main__":
    main()