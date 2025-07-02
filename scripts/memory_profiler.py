#!/usr/bin/env python3
"""
ONNX メモリ使用量詳細プロファイリングツール
推論中のメモリ使用パターンを詳細分析し最適化案提示
"""
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import psutil
import gc
import tracemalloc
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import threading

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class MemoryProfiler:
    """メモリ使用量プロファイラー"""
    
    def __init__(self):
        self.memory_samples = []
        self.time_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 0.001):
        """メモリ監視開始"""
        self.memory_samples = []
        self.time_samples = []
        self.monitoring = True
        self.start_time = time.time()
        
        def monitor():
            while self.monitoring:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                current_time = time.time() - self.start_time
                
                self.memory_samples.append(current_memory)
                self.time_samples.append(current_time)
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """メモリ監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_stats(self):
        """メモリ統計取得"""
        if not self.memory_samples:
            return None
        
        memory_array = np.array(self.memory_samples)
        
        return {
            'max_memory_mb': np.max(memory_array),
            'min_memory_mb': np.min(memory_array),
            'avg_memory_mb': np.mean(memory_array),
            'peak_increase_mb': np.max(memory_array) - np.min(memory_array),
            'samples_count': len(memory_array),
            'duration_seconds': self.time_samples[-1] if self.time_samples else 0
        }

def profile_model_loading():
    """モデルロード時のメモリプロファイリング"""
    print("=== モデルロード メモリプロファイリング ===")
    
    onnx_path = project_root / "exports/onnx/yolov7n_mezzopiano.onnx"
    
    # メモリ監視開始
    profiler = MemoryProfiler()
    profiler.start_monitoring()
    
    # 初期メモリ
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"🔍 初期メモリ: {initial_memory:.1f}MB")
    
    # モデルロード
    print("📁 モデルロード開始...")
    load_start = time.time()
    
    session = ort.InferenceSession(str(onnx_path))
    
    load_end = time.time()
    
    # ロード後メモリ
    post_load_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"📁 ロード完了: {load_end - load_start:.2f}秒")
    print(f"📊 ロード後メモリ: {post_load_memory:.1f}MB")
    print(f"📈 メモリ増加: {post_load_memory - initial_memory:.1f}MB")
    
    profiler.stop_monitoring()
    
    return {
        'session': session,
        'load_time': load_end - load_start,
        'initial_memory': initial_memory,
        'post_load_memory': post_load_memory,
        'memory_increase': post_load_memory - initial_memory,
        'profiler_stats': profiler.get_stats()
    }

def profile_inference_memory(session, num_inferences: int = 100):
    """推論時メモリプロファイリング"""
    print(f"\n=== 推論時メモリプロファイリング ({num_inferences}回) ===")
    
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    # メモリ監視開始
    profiler = MemoryProfiler()
    profiler.start_monitoring()
    
    # ウォームアップ
    print("🔥 ウォームアップ実行...")
    for _ in range(5):
        _ = session.run(None, {input_name: dummy_input})
    
    # 推論実行
    print(f"⚡ 推論実行中 ({num_inferences}回)...")
    inference_start = time.time()
    
    for i in range(num_inferences):
        outputs = session.run(None, {input_name: dummy_input})
        
        if (i + 1) % 20 == 0:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"   {i+1:3d}回目: {current_memory:.1f}MB")
    
    inference_end = time.time()
    
    profiler.stop_monitoring()
    
    stats = profiler.get_stats()
    total_time = inference_end - inference_start
    avg_inference_time = total_time / num_inferences * 1000  # ms
    
    print(f"✅ 推論完了: {total_time:.2f}秒")
    print(f"📊 平均推論時間: {avg_inference_time:.1f}ms")
    print(f"📈 最大メモリ: {stats['max_memory_mb']:.1f}MB")
    print(f"📉 最小メモリ: {stats['min_memory_mb']:.1f}MB")
    print(f"📊 メモリ変動: {stats['peak_increase_mb']:.1f}MB")
    
    return {
        'inference_stats': stats,
        'total_time': total_time,
        'avg_inference_time': avg_inference_time,
        'memory_samples': profiler.memory_samples,
        'time_samples': profiler.time_samples
    }

def profile_batch_inference(session, batch_sizes: List[int] = [1, 2, 4, 8]):
    """バッチサイズ別メモリプロファイリング"""
    print(f"\n=== バッチサイズ別メモリプロファイリング ===")
    
    input_name = session.get_inputs()[0].name
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n🔍 バッチサイズ {batch_size} テスト")
        
        # バッチ入力作成
        batch_input = np.random.randn(batch_size, 3, 640, 640).astype(np.float32)
        
        # メモリ監視
        profiler = MemoryProfiler()
        profiler.start_monitoring()
        
        # 推論実行
        try:
            start_time = time.time()
            outputs = session.run(None, {input_name: batch_input})
            end_time = time.time()
            
            profiler.stop_monitoring()
            
            inference_time = (end_time - start_time) * 1000  # ms
            per_sample_time = inference_time / batch_size
            
            stats = profiler.get_stats()
            
            results[batch_size] = {
                'success': True,
                'total_time_ms': inference_time,
                'per_sample_time_ms': per_sample_time,
                'memory_stats': stats,
                'output_shapes': [out.shape for out in outputs]
            }
            
            print(f"✅ 成功: {inference_time:.1f}ms ({per_sample_time:.1f}ms/sample)")
            print(f"📊 最大メモリ: {stats['max_memory_mb']:.1f}MB")
            
        except Exception as e:
            profiler.stop_monitoring()
            results[batch_size] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ 失敗: {e}")
        
        # メモリクリーンアップ
        gc.collect()
        time.sleep(1)
    
    return results

def analyze_memory_optimization():
    """メモリ最適化分析"""
    print(f"\n=== メモリ最適化分析 ===")
    
    # システム情報取得
    system_memory = psutil.virtual_memory()
    process = psutil.Process()
    
    analysis = {
        'system_memory_total_gb': system_memory.total / 1024**3,
        'system_memory_available_gb': system_memory.available / 1024**3,
        'system_memory_used_percent': system_memory.percent,
        'process_memory_mb': process.memory_info().rss / 1024 / 1024,
        'process_memory_percent': process.memory_percent()
    }
    
    print(f"🖥️ システム総メモリ: {analysis['system_memory_total_gb']:.1f}GB")
    print(f"💾 利用可能メモリ: {analysis['system_memory_available_gb']:.1f}GB")
    print(f"📊 メモリ使用率: {analysis['system_memory_used_percent']:.1f}%")
    print(f"🔧 プロセスメモリ: {analysis['process_memory_mb']:.1f}MB")
    
    # 最適化推奨事項
    recommendations = []
    
    if analysis['process_memory_mb'] > 500:
        recommendations.append("メモリ使用量が500MB超 - モデル量子化を検討")
    
    if analysis['system_memory_used_percent'] > 80:
        recommendations.append("システムメモリ使用率が80%超 - メモリ最適化が必要")
    
    if analysis['system_memory_available_gb'] < 2:
        recommendations.append("利用可能メモリが2GB未満 - 軽量化モデルを検討")
    
    analysis['recommendations'] = recommendations
    
    return analysis

def create_memory_visualization(inference_result: Dict, output_dir: Path):
    """メモリ使用量可視化"""
    print(f"\n=== メモリ使用量可視化作成 ===")
    
    try:
        plt.figure(figsize=(12, 8))
        
        # メモリ使用量推移
        plt.subplot(2, 2, 1)
        plt.plot(inference_result['time_samples'], inference_result['memory_samples'], 'b-', linewidth=1)
        plt.title('推論中メモリ使用量推移')
        plt.xlabel('時間 (秒)')
        plt.ylabel('メモリ使用量 (MB)')
        plt.grid(True)
        
        # メモリ使用量分布
        plt.subplot(2, 2, 2)
        plt.hist(inference_result['memory_samples'], bins=30, alpha=0.7, color='green')
        plt.title('メモリ使用量分布')
        plt.xlabel('メモリ使用量 (MB)')
        plt.ylabel('頻度')
        plt.grid(True)
        
        # 統計情報
        plt.subplot(2, 2, 3)
        stats = inference_result['inference_stats']
        labels = ['最大', '平均', '最小']
        values = [stats['max_memory_mb'], stats['avg_memory_mb'], stats['min_memory_mb']]
        plt.bar(labels, values, color=['red', 'blue', 'green'])
        plt.title('メモリ統計')
        plt.ylabel('メモリ使用量 (MB)')
        plt.grid(True)
        
        # 推論時間統計
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f"平均推論時間: {inference_result['avg_inference_time']:.1f}ms", fontsize=12)
        plt.text(0.1, 0.6, f"総実行時間: {inference_result['total_time']:.2f}秒", fontsize=12)
        plt.text(0.1, 0.4, f"メモリ変動: {stats['peak_increase_mb']:.1f}MB", fontsize=12)
        plt.text(0.1, 0.2, f"サンプル数: {stats['samples_count']}", fontsize=12)
        plt.title('実行統計')
        plt.axis('off')
        
        plt.tight_layout()
        
        # 保存
        viz_path = output_dir / 'memory_profile_visualization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可視化保存: {viz_path}")
        return viz_path
        
    except Exception as e:
        print(f"⚠️ 可視化作成エラー: {e}")
        return None

def generate_memory_report(load_result: Dict, inference_result: Dict, batch_results: Dict, optimization_analysis: Dict, output_dir: Path):
    """メモリプロファイリングレポート生成"""
    print(f"\n=== メモリプロファイリングレポート生成 ===")
    
    report_path = output_dir / "memory_profile_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# YOLOv7n ONNX メモリプロファイリングレポート\n\n")
        f.write(f"**生成日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # システム情報
        f.write("## 🖥️ システム情報\n\n")
        f.write(f"- **総メモリ**: {optimization_analysis['system_memory_total_gb']:.1f}GB\n")
        f.write(f"- **利用可能メモリ**: {optimization_analysis['system_memory_available_gb']:.1f}GB\n")
        f.write(f"- **メモリ使用率**: {optimization_analysis['system_memory_used_percent']:.1f}%\n")
        f.write(f"- **プロセスメモリ**: {optimization_analysis['process_memory_mb']:.1f}MB\n\n")
        
        # モデルロード
        f.write("## 📁 モデルロード分析\n\n")
        f.write(f"- **ロード時間**: {load_result['load_time']:.2f}秒\n")
        f.write(f"- **初期メモリ**: {load_result['initial_memory']:.1f}MB\n")
        f.write(f"- **ロード後メモリ**: {load_result['post_load_memory']:.1f}MB\n")
        f.write(f"- **メモリ増加**: {load_result['memory_increase']:.1f}MB\n\n")
        
        # 推論メモリ
        f.write("## ⚡ 推論メモリ分析\n\n")
        stats = inference_result['inference_stats']
        f.write(f"- **平均推論時間**: {inference_result['avg_inference_time']:.1f}ms\n")
        f.write(f"- **最大メモリ**: {stats['max_memory_mb']:.1f}MB\n")
        f.write(f"- **最小メモリ**: {stats['min_memory_mb']:.1f}MB\n")
        f.write(f"- **平均メモリ**: {stats['avg_memory_mb']:.1f}MB\n")
        f.write(f"- **メモリ変動**: {stats['peak_increase_mb']:.1f}MB\n\n")
        
        # バッチサイズ分析
        f.write("## 📊 バッチサイズ別分析\n\n")
        f.write("| バッチサイズ | 推論時間(ms) | サンプル単価(ms) | 最大メモリ(MB) | ステータス |\n")
        f.write("|-------------|-------------|-----------------|-------------|----------|\n")
        
        for batch_size, result in batch_results.items():
            if result['success']:
                f.write(f"| {batch_size} | {result['total_time_ms']:.1f} | {result['per_sample_time_ms']:.1f} | {result['memory_stats']['max_memory_mb']:.1f} | ✅ 成功 |\n")
            else:
                f.write(f"| {batch_size} | - | - | - | ❌ 失敗 |\n")
        
        # 最適化推奨
        f.write("\n## 🎯 最適化推奨事項\n\n")
        if optimization_analysis['recommendations']:
            for rec in optimization_analysis['recommendations']:
                f.write(f"- ⚠️ {rec}\n")
        else:
            f.write("- ✅ 現在のメモリ使用量は適切な範囲内です\n")
        
        f.write("\n### 一般的な最適化手法\n")
        f.write("1. **モデル量子化**: FP32 → FP16 または INT8 変換\n")
        f.write("2. **バッチサイズ調整**: メモリ制約に合わせた最適バッチサイズ\n")
        f.write("3. **プロバイダー最適化**: CUDA、TensorRT等の利用\n")
        f.write("4. **メモリプール設定**: ONNX Runtimeメモリ設定調整\n\n")
    
    print(f"✅ レポート保存: {report_path}")
    return report_path

def main():
    """メイン実行"""
    print("🔍 ONNX メモリプロファイリング開始")
    
    # 出力ディレクトリ作成
    output_dir = project_root / "analysis/memory_profiling"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. モデルロードプロファイリング
        load_result = profile_model_loading()
        
        # 2. 推論メモリプロファイリング
        inference_result = profile_inference_memory(load_result['session'], num_inferences=100)
        
        # 3. バッチサイズ別プロファイリング
        batch_results = profile_batch_inference(load_result['session'])
        
        # 4. メモリ最適化分析
        optimization_analysis = analyze_memory_optimization()
        
        # 5. 可視化作成
        viz_path = create_memory_visualization(inference_result, output_dir)
        
        # 6. レポート生成
        report_path = generate_memory_report(
            load_result, inference_result, batch_results, 
            optimization_analysis, output_dir
        )
        
        print(f"\n🎉 メモリプロファイリング完了!")
        print(f"📁 レポート: {report_path}")
        if viz_path:
            print(f"📊 可視化: {viz_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ メモリプロファイリング失敗: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)