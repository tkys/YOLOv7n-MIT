#!/usr/bin/env python3
"""
ONNX Runtime 推論速度ベンチマーク・最適化スクリプト
YOLOv7n ONNX モデルの性能評価と最適化設定テスト
"""
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import psutil
import gc
from typing import Dict, List, Tuple

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def get_available_providers():
    """利用可能なONNX Runtime プロバイダー取得"""
    print("=== 利用可能プロバイダー ===")
    
    available = ort.get_available_providers()
    print(f"利用可能: {available}")
    
    # 推奨順序で整理
    recommended_order = []
    
    # GPU優先 (利用可能な場合)
    if 'CUDAExecutionProvider' in available:
        recommended_order.append('CUDAExecutionProvider')
        print("✅ CUDA GPU アクセラレーション利用可能")
    
    if 'TensorrtExecutionProvider' in available:
        recommended_order.append('TensorrtExecutionProvider')
        print("✅ TensorRT アクセラレーション利用可能")
    
    # CPU最適化
    if 'CPUExecutionProvider' in available:
        recommended_order.append('CPUExecutionProvider')
        print("✅ CPU実行利用可能")
    
    return recommended_order

def create_session_configs():
    """各種セッション設定を作成"""
    print("\n=== セッション設定作成 ===")
    
    configs = {}
    
    # 1. デフォルト設定
    configs['default'] = {
        'providers': ['CPUExecutionProvider'],
        'session_options': None,
        'description': 'デフォルトCPU実行'
    }
    
    # 2. CPU最適化設定
    cpu_options = ort.SessionOptions()
    cpu_options.intra_op_num_threads = psutil.cpu_count()
    cpu_options.inter_op_num_threads = 1
    cpu_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    cpu_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    configs['cpu_optimized'] = {
        'providers': ['CPUExecutionProvider'],
        'session_options': cpu_options,
        'description': f'CPU最適化 ({psutil.cpu_count()}スレッド)'
    }
    
    # 3. GPU設定 (利用可能な場合)
    available = get_available_providers()
    if 'CUDAExecutionProvider' in available:
        gpu_options = ort.SessionOptions()
        gpu_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        configs['gpu_cuda'] = {
            'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            'session_options': gpu_options,
            'description': 'CUDA GPU アクセラレーション'
        }
    
    # 4. メモリ最適化設定
    memory_options = ort.SessionOptions()
    memory_options.enable_mem_pattern = True
    memory_options.enable_cpu_mem_arena = True
    memory_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    configs['memory_optimized'] = {
        'providers': ['CPUExecutionProvider'],
        'session_options': memory_options,
        'description': 'メモリ使用量最適化'
    }
    
    for name, config in configs.items():
        print(f"✅ {name}: {config['description']}")
    
    return configs

def benchmark_session(session, input_name: str, warmup_runs: int = 5, benchmark_runs: int = 50):
    """セッションのベンチマーク実行"""
    # ダミー入力作成
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    # ウォームアップ
    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: dummy_input})
    
    # ベンチマーク実行
    times = []
    for _ in range(benchmark_runs):
        start_time = time.perf_counter()
        outputs = session.run(None, {input_name: dummy_input})
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # ミリ秒変換
    
    # 統計計算
    times = np.array(times)
    stats = {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99),
        'fps': 1000 / np.mean(times),
        'output_shapes': [out.shape for out in outputs]
    }
    
    return stats

def measure_memory_usage(session, input_name: str):
    """メモリ使用量測定"""
    import psutil
    
    # 実行前メモリ測定
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # ダミー推論実行
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    _ = session.run(None, {input_name: dummy_input})
    
    # 実行後メモリ測定
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory,
        'memory_increase_mb': final_memory - initial_memory
    }

def run_comprehensive_benchmark():
    """包括的ベンチマーク実行"""
    print("🚀 ONNX Runtime 包括ベンチマーク開始")
    
    # ONNX モデルパス
    onnx_path = project_root / "exports/onnx/yolov7n_mezzopiano.onnx"
    if not onnx_path.exists():
        print(f"❌ ONNXファイルが見つかりません: {onnx_path}")
        return False
    
    print(f"📁 モデル: {onnx_path}")
    print(f"📊 ファイルサイズ: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # セッション設定作成
    configs = create_session_configs()
    
    # ベンチマーク結果保存
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n=== {config_name} ベンチマーク ===")
        print(f"説明: {config['description']}")
        
        try:
            # セッション作成
            session = ort.InferenceSession(
                str(onnx_path),
                sess_options=config['session_options'],
                providers=config['providers']
            )
            
            # 入力名取得
            input_name = session.get_inputs()[0].name
            
            # 推論速度ベンチマーク
            print("⏱️ 推論速度測定中...")
            perf_stats = benchmark_session(session, input_name)
            
            # メモリ使用量測定
            print("💾 メモリ使用量測定中...")
            memory_stats = measure_memory_usage(session, input_name)
            
            # 結果保存
            results[config_name] = {
                'config': config,
                'performance': perf_stats,
                'memory': memory_stats,
                'providers_used': session.get_providers()
            }
            
            # 結果表示
            print(f"✅ 平均推論時間: {perf_stats['mean']:.1f}ms")
            print(f"✅ FPS: {perf_stats['fps']:.1f}")
            print(f"✅ P95推論時間: {perf_stats['p95']:.1f}ms")
            print(f"✅ メモリ使用量: {memory_stats['final_memory_mb']:.1f}MB")
            
            # メモリクリーンアップ
            del session
            gc.collect()
            
        except Exception as e:
            print(f"❌ {config_name} ベンチマーク失敗: {e}")
            results[config_name] = {'error': str(e)}
    
    return results

def generate_benchmark_report(results: Dict):
    """ベンチマーク結果レポート生成"""
    print(f"\n=== ベンチマーク結果レポート生成 ===")
    
    output_dir = project_root / "analysis/onnx_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "benchmark_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# YOLOv7n ONNX Runtime ベンチマークレポート\n\n")
        f.write(f"**生成日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**モデル**: YOLOv7n Mezzopiano ONNX\n\n")
        
        # システム情報
        f.write("## 🖥️ システム情報\n\n")
        f.write(f"- **CPU**: {psutil.cpu_count()}コア\n")
        f.write(f"- **メモリ**: {psutil.virtual_memory().total / 1024**3:.1f}GB\n")
        f.write(f"- **ONNX Runtime**: {ort.__version__}\n")
        f.write(f"- **利用可能プロバイダー**: {', '.join(ort.get_available_providers())}\n\n")
        
        # ベンチマーク結果テーブル
        f.write("## 📊 ベンチマーク結果\n\n")
        f.write("| 設定 | 平均推論時間(ms) | FPS | P95時間(ms) | メモリ(MB) | プロバイダー |\n")
        f.write("|------|------------------|-----|-------------|------------|-------------|\n")
        
        for config_name, result in results.items():
            if 'error' in result:
                f.write(f"| {config_name} | ❌ エラー | - | - | - | - |\n")
            else:
                perf = result['performance']
                memory = result['memory']
                providers = ', '.join(result['providers_used'])
                
                f.write(f"| {config_name} | {perf['mean']:.1f} | {perf['fps']:.1f} | {perf['p95']:.1f} | {memory['final_memory_mb']:.1f} | {providers} |\n")
        
        # 詳細統計
        f.write("\n## 📈 詳細統計\n\n")
        
        best_fps = 0
        best_config = None
        
        for config_name, result in results.items():
            if 'error' not in result:
                perf = result['performance']
                memory = result['memory']
                
                f.write(f"### {config_name}\n")
                f.write(f"- **説明**: {result['config']['description']}\n")
                f.write(f"- **平均推論時間**: {perf['mean']:.2f}ms ± {perf['std']:.2f}ms\n")
                f.write(f"- **最小/最大**: {perf['min']:.1f}ms / {perf['max']:.1f}ms\n")
                f.write(f"- **中央値**: {perf['median']:.1f}ms\n")
                f.write(f"- **P95/P99**: {perf['p95']:.1f}ms / {perf['p99']:.1f}ms\n")
                f.write(f"- **FPS**: {perf['fps']:.1f}\n")
                f.write(f"- **メモリ使用量**: {memory['final_memory_mb']:.1f}MB\n")
                f.write(f"- **出力形状**: {perf['output_shapes'][:3]}... (最初の3つ)\n\n")
                
                if perf['fps'] > best_fps:
                    best_fps = perf['fps']
                    best_config = config_name
        
        # 推奨設定
        f.write("## 🎯 推奨設定\n\n")
        if best_config:
            f.write(f"**最高性能**: {best_config} ({best_fps:.1f} FPS)\n\n")
        
        # 用途別推奨
        f.write("### 用途別推奨\n")
        f.write("- **リアルタイム推論**: CPU最適化またはGPU設定\n")
        f.write("- **バッチ処理**: メモリ最適化設定\n")
        f.write("- **組み込み環境**: デフォルト設定\n\n")
    
    print(f"✅ レポート保存: {report_path}")
    return report_path

def main():
    """メイン実行"""
    print("📊 ONNX Runtime ベンチマーク・最適化ツール")
    
    try:
        # 包括的ベンチマーク実行
        results = run_comprehensive_benchmark()
        
        if not results:
            return False
        
        # レポート生成
        report_path = generate_benchmark_report(results)
        
        print(f"\n🎉 ベンチマーク完了!")
        print(f"📁 詳細レポート: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ベンチマーク失敗: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)