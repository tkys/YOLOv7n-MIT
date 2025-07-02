#!/usr/bin/env python3
"""
ONNX クロスプラットフォーム動作検証ツール
異なる環境・設定でのONNXモデル互換性検証
"""
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import platform
import subprocess
import json
from typing import Dict, List, Tuple

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def get_system_info():
    """システム情報収集"""
    print("=== システム情報収集 ===")
    
    info = {
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'onnxruntime_version': ort.__version__,
        'available_providers': ort.get_available_providers(),
    }
    
    # NumPy情報
    info['numpy_version'] = np.__version__
    
    # CPUコア数
    try:
        import psutil
        info['cpu_count'] = psutil.cpu_count()
        info['memory_gb'] = psutil.virtual_memory().total / 1024**3
    except ImportError:
        info['cpu_count'] = 'N/A'
        info['memory_gb'] = 'N/A'
    
    # GPU情報（可能であれば）
    try:
        import torch
        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'
        else:
            info['cuda_available'] = False
    except ImportError:
        info['cuda_available'] = 'Unknown'
    
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return info

def validate_onnx_model_structure():
    """ONNX モデル構造検証"""
    print("\n=== ONNX モデル構造検証 ===")
    
    onnx_path = project_root / "exports/onnx/yolov7n_mezzopiano.onnx"
    
    if not onnx_path.exists():
        print(f"❌ ONNXファイルが見つかりません: {onnx_path}")
        return False
    
    try:
        import onnx
        
        # モデルロード・検証
        print("📁 ONNXモデル構造検証中...")
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        
        # モデル情報取得
        model_info = {
            'file_size_mb': onnx_path.stat().st_size / 1024 / 1024,
            'ir_version': model.ir_version,
            'producer_name': model.producer_name,
            'producer_version': model.producer_version,
            'graph_input_count': len(model.graph.input),
            'graph_output_count': len(model.graph.output),
            'graph_node_count': len(model.graph.node)
        }
        
        print(f"✅ ONNXモデル構造: 正常")
        print(f"   ファイルサイズ: {model_info['file_size_mb']:.1f}MB")
        print(f"   IR Version: {model_info['ir_version']}")
        print(f"   Producer: {model_info['producer_name']} {model_info['producer_version']}")
        print(f"   入力数: {model_info['graph_input_count']}")
        print(f"   出力数: {model_info['graph_output_count']}")
        print(f"   ノード数: {model_info['graph_node_count']}")
        
        return model_info
        
    except ImportError:
        print("⚠️ onnxライブラリが見つかりません。基本検証をスキップします。")
        return {'file_size_mb': onnx_path.stat().st_size / 1024 / 1024}
    except Exception as e:
        print(f"❌ ONNX モデル構造検証失敗: {e}")
        return False

def test_provider_compatibility():
    """各プロバイダー互換性テスト"""
    print("\n=== プロバイダー互換性テスト ===")
    
    onnx_path = project_root / "exports/onnx/yolov7n_mezzopiano.onnx"
    
    # テスト対象プロバイダー
    test_providers = [
        (['CPUExecutionProvider'], 'CPU基本実行'),
        (['AzureExecutionProvider', 'CPUExecutionProvider'], 'Azure実行'),
    ]
    
    # CUDA利用可能であればテスト
    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        test_providers.append((['CUDAExecutionProvider', 'CPUExecutionProvider'], 'CUDA GPU実行'))
    
    if 'TensorrtExecutionProvider' in available:
        test_providers.append((['TensorrtExecutionProvider', 'CPUExecutionProvider'], 'TensorRT実行'))
    
    results = {}
    
    for providers, description in test_providers:
        print(f"\n🔍 テスト中: {description}")
        
        try:
            # セッション作成
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            # 入力準備
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
            
            # 推論テスト
            start_time = time.time()
            outputs = session.run(None, {input_name: dummy_input})
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # ms
            
            results[description] = {
                'success': True,
                'providers_requested': providers,
                'providers_used': session.get_providers(),
                'inference_time_ms': inference_time,
                'output_count': len(outputs),
                'output_shapes': [out.shape for out in outputs]
            }
            
            print(f"✅ 成功: {inference_time:.1f}ms")
            print(f"   使用プロバイダー: {', '.join(session.get_providers())}")
            
        except Exception as e:
            results[description] = {
                'success': False,
                'providers_requested': providers,
                'error': str(e)
            }
            print(f"❌ 失敗: {e}")
    
    return results

def test_input_variations():
    """入力バリエーションテスト"""
    print("\n=== 入力バリエーションテスト ===")
    
    onnx_path = project_root / "exports/onnx/yolov7n_mezzopiano.onnx"
    
    try:
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        
        # テスト入力パターン
        test_cases = [
            ('正常入力', np.random.randn(1, 3, 640, 640).astype(np.float32)),
            ('ゼロ入力', np.zeros((1, 3, 640, 640), dtype=np.float32)),
            ('最大値入力', np.ones((1, 3, 640, 640), dtype=np.float32)),
            ('ランダム範囲入力', np.random.uniform(0, 1, (1, 3, 640, 640)).astype(np.float32)),
            ('正規化済み入力', (np.random.randn(1, 3, 640, 640) * 0.5 + 0.5).astype(np.float32)),
        ]
        
        results = {}
        
        for test_name, test_input in test_cases:
            print(f"🧪 テスト: {test_name}")
            
            try:
                start_time = time.time()
                outputs = session.run(None, {input_name: test_input})
                end_time = time.time()
                
                inference_time = (end_time - start_time) * 1000
                
                # 出力検証
                output_valid = all(
                    not np.isnan(out).any() and not np.isinf(out).any() 
                    for out in outputs
                )
                
                results[test_name] = {
                    'success': True,
                    'inference_time_ms': inference_time,
                    'output_valid': output_valid,
                    'output_shapes': [out.shape for out in outputs],
                    'output_ranges': [(float(np.min(out)), float(np.max(out))) for out in outputs[:3]]  # 最初の3つのみ
                }
                
                status = "✅ 有効" if output_valid else "⚠️ 無効値検出"
                print(f"   {status}: {inference_time:.1f}ms")
                
            except Exception as e:
                results[test_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"   ❌ 失敗: {e}")
        
        return results
        
    except Exception as e:
        print(f"❌ 入力バリエーションテスト失敗: {e}")
        return {}

def test_concurrent_inference():
    """並行推論テスト"""
    print("\n=== 並行推論テスト ===")
    
    onnx_path = project_root / "exports/onnx/yolov7n_mezzopiano.onnx"
    
    try:
        import threading
        import queue
        
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        
        # 並行推論テスト設定
        num_threads = 4
        inferences_per_thread = 10
        
        results_queue = queue.Queue()
        
        def worker():
            """ワーカースレッド"""
            thread_results = []
            
            for i in range(inferences_per_thread):
                dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
                
                try:
                    start_time = time.time()
                    outputs = session.run(None, {input_name: dummy_input})
                    end_time = time.time()
                    
                    thread_results.append({
                        'success': True,
                        'inference_time_ms': (end_time - start_time) * 1000
                    })
                    
                except Exception as e:
                    thread_results.append({
                        'success': False,
                        'error': str(e)
                    })
            
            results_queue.put(thread_results)
        
        # スレッド実行
        print(f"🔄 {num_threads}スレッドで並行推論テスト中...")
        
        threads = []
        start_time = time.time()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # 結果集計
        all_results = []
        while not results_queue.empty():
            all_results.extend(results_queue.get())
        
        successful_inferences = [r for r in all_results if r['success']]
        failed_inferences = [r for r in all_results if not r['success']]
        
        if successful_inferences:
            inference_times = [r['inference_time_ms'] for r in successful_inferences]
            avg_time = np.mean(inference_times)
            max_time = np.max(inference_times)
            min_time = np.min(inference_times)
            
            concurrent_result = {
                'total_inferences': len(all_results),
                'successful_inferences': len(successful_inferences),
                'failed_inferences': len(failed_inferences),
                'total_time_seconds': total_time,
                'avg_inference_time_ms': avg_time,
                'max_inference_time_ms': max_time,
                'min_inference_time_ms': min_time,
                'throughput_fps': len(successful_inferences) / total_time
            }
            
            print(f"✅ 並行推論テスト完了")
            print(f"   成功: {len(successful_inferences)}/{len(all_results)}")
            print(f"   平均推論時間: {avg_time:.1f}ms")
            print(f"   スループット: {concurrent_result['throughput_fps']:.1f} FPS")
            
            return concurrent_result
        else:
            print(f"❌ 全ての並行推論が失敗")
            return {'error': '全推論失敗'}
        
    except ImportError:
        print("⚠️ threadingライブラリが見つかりません。並行テストをスキップします。")
        return {'skipped': 'threading not available'}
    except Exception as e:
        print(f"❌ 並行推論テスト失敗: {e}")
        return {'error': str(e)}

def generate_compatibility_report(system_info: Dict, model_info: Dict, provider_results: Dict, 
                                 input_results: Dict, concurrent_result: Dict, output_dir: Path):
    """互換性検証レポート生成"""
    print(f"\n=== 互換性検証レポート生成 ===")
    
    report_path = output_dir / "cross_platform_compatibility_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# YOLOv7n ONNX クロスプラットフォーム互換性レポート\n\n")
        f.write(f"**生成日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # システム情報
        f.write("## 🖥️ テスト環境\n\n")
        f.write(f"- **プラットフォーム**: {system_info['platform']}\n")
        f.write(f"- **アーキテクチャ**: {system_info['machine']}\n")
        f.write(f"- **Python**: {system_info['python_version']}\n")
        f.write(f"- **ONNX Runtime**: {system_info['onnxruntime_version']}\n")
        f.write(f"- **NumPy**: {system_info['numpy_version']}\n")
        f.write(f"- **CPU**: {system_info['cpu_count']}コア\n")
        f.write(f"- **メモリ**: {system_info['memory_gb']:.1f}GB\n")
        f.write(f"- **CUDA利用可能**: {system_info['cuda_available']}\n")
        f.write(f"- **利用可能プロバイダー**: {', '.join(system_info['available_providers'])}\n\n")
        
        # モデル情報
        if model_info:
            f.write("## 📋 モデル情報\n\n")
            f.write(f"- **ファイルサイズ**: {model_info['file_size_mb']:.1f}MB\n")
            if 'ir_version' in model_info:
                f.write(f"- **IR Version**: {model_info['ir_version']}\n")
                f.write(f"- **Producer**: {model_info['producer_name']} {model_info['producer_version']}\n")
                f.write(f"- **入力数**: {model_info['graph_input_count']}\n")
                f.write(f"- **出力数**: {model_info['graph_output_count']}\n")
                f.write(f"- **ノード数**: {model_info['graph_node_count']}\n")
            f.write("\n")
        
        # プロバイダー互換性
        f.write("## ⚙️ プロバイダー互換性\n\n")
        f.write("| プロバイダー | ステータス | 推論時間(ms) | 実際のプロバイダー |\n")
        f.write("|-------------|-----------|-------------|------------------|\n")
        
        for provider_desc, result in provider_results.items():
            if result['success']:
                actual_providers = ', '.join(result['providers_used'])
                f.write(f"| {provider_desc} | ✅ 成功 | {result['inference_time_ms']:.1f} | {actual_providers} |\n")
            else:
                f.write(f"| {provider_desc} | ❌ 失敗 | - | - |\n")
        
        # 入力バリエーション
        f.write("\n## 🧪 入力バリエーションテスト\n\n")
        f.write("| テストケース | ステータス | 推論時間(ms) | 出力有効性 |\n")
        f.write("|-------------|-----------|-------------|----------|\n")
        
        for test_name, result in input_results.items():
            if result['success']:
                validity = "✅ 有効" if result['output_valid'] else "⚠️ 無効値"
                f.write(f"| {test_name} | ✅ 成功 | {result['inference_time_ms']:.1f} | {validity} |\n")
            else:
                f.write(f"| {test_name} | ❌ 失敗 | - | - |\n")
        
        # 並行推論
        f.write("\n## 🔄 並行推論テスト\n\n")
        if 'error' not in concurrent_result and 'skipped' not in concurrent_result:
            f.write(f"- **総推論数**: {concurrent_result['total_inferences']}\n")
            f.write(f"- **成功推論**: {concurrent_result['successful_inferences']}\n")
            f.write(f"- **失敗推論**: {concurrent_result['failed_inferences']}\n")
            f.write(f"- **総実行時間**: {concurrent_result['total_time_seconds']:.2f}秒\n")
            f.write(f"- **平均推論時間**: {concurrent_result['avg_inference_time_ms']:.1f}ms\n")
            f.write(f"- **最大推論時間**: {concurrent_result['max_inference_time_ms']:.1f}ms\n")
            f.write(f"- **最小推論時間**: {concurrent_result['min_inference_time_ms']:.1f}ms\n")
            f.write(f"- **スループット**: {concurrent_result['throughput_fps']:.1f} FPS\n")
        elif 'skipped' in concurrent_result:
            f.write(f"⚠️ スキップ: {concurrent_result['skipped']}\n")
        else:
            f.write(f"❌ 失敗: {concurrent_result['error']}\n")
        
        # 総合評価
        f.write("\n## 🎯 総合評価\n\n")
        
        success_count = sum(1 for r in provider_results.values() if r['success'])
        total_provider_tests = len(provider_results)
        
        input_success_count = sum(1 for r in input_results.values() if r['success'])
        total_input_tests = len(input_results)
        
        f.write(f"### 互換性スコア\n")
        f.write(f"- **プロバイダー互換性**: {success_count}/{total_provider_tests} ({success_count/total_provider_tests*100:.1f}%)\n")
        f.write(f"- **入力耐性**: {input_success_count}/{total_input_tests} ({input_success_count/total_input_tests*100:.1f}%)\n")
        
        if success_count == total_provider_tests and input_success_count == total_input_tests:
            f.write(f"- **総合評価**: ✅ 優秀 - 全テスト通過\n")
        elif success_count >= total_provider_tests * 0.8:
            f.write(f"- **総合評価**: ⚠️ 良好 - 一部制限あり\n")
        else:
            f.write(f"- **総合評価**: ❌ 要改善 - 重要な問題あり\n")
        
        f.write("\n### 推奨事項\n")
        
        # CPU専用環境での推奨
        cpu_provider_success = any(
            r['success'] and 'CPUExecutionProvider' in r['providers_used'] 
            for r in provider_results.values()
        )
        
        if cpu_provider_success:
            f.write("- ✅ CPU環境での動作確認済み\n")
        else:
            f.write("- ❌ CPU環境での動作に問題あり\n")
        
        # GPU利用推奨
        gpu_provider_success = any(
            r['success'] and any(p in r['providers_used'] for p in ['CUDAExecutionProvider', 'TensorrtExecutionProvider'])
            for r in provider_results.values()
        )
        
        if gpu_provider_success:
            f.write("- ✅ GPU アクセラレーション利用可能\n")
        else:
            f.write("- ⚠️ GPU アクセラレーション未対応または未テスト\n")
        
        f.write("\n")
    
    print(f"✅ レポート保存: {report_path}")
    return report_path

def main():
    """メイン実行"""
    print("🔍 ONNX クロスプラットフォーム互換性検証開始")
    
    # 出力ディレクトリ作成
    output_dir = project_root / "analysis/cross_platform"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. システム情報収集
        system_info = get_system_info()
        
        # 2. ONNX モデル構造検証
        model_info = validate_onnx_model_structure()
        if not model_info:
            return False
        
        # 3. プロバイダー互換性テスト
        provider_results = test_provider_compatibility()
        
        # 4. 入力バリエーションテスト
        input_results = test_input_variations()
        
        # 5. 並行推論テスト
        concurrent_result = test_concurrent_inference()
        
        # 6. レポート生成
        report_path = generate_compatibility_report(
            system_info, model_info, provider_results,
            input_results, concurrent_result, output_dir
        )
        
        print(f"\n🎉 クロスプラットフォーム検証完了!")
        print(f"📁 レポート: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ クロスプラットフォーム検証失敗: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)