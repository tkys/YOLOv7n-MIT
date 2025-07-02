"""
YOLOv7n MIT Edition - 包括的ベンチマークスイート
Docker環境での完全なパフォーマンステスト
"""

import os
import time
from pathlib import Path
from typing import Dict, List

from evaluation_report import ModelEvaluator
from inference.logger import get_logger
from quantize_onnx import ONNXQuantizer

logger = get_logger("benchmark")

class BenchmarkSuite:
    """包括的ベンチマークスイート"""
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        """
        ベンチマークスイート初期化
        
        Args:
            models_dir: モデルディレクトリ
            data_dir: データディレクトリ
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.results = {}
        
        logger.info(
            "ベンチマークスイート初期化",
            models_dir=str(self.models_dir),
            data_dir=str(self.data_dir)
        )
    
    def discover_models(self) -> List[Path]:
        """利用可能なモデルを発見"""
        model_files = []
        
        # ONNXモデル検索
        for pattern in ["*.onnx", "**/*.onnx"]:
            model_files.extend(self.models_dir.glob(pattern))
        
        logger.info(f"発見されたモデル数: {len(model_files)}")
        for model in model_files:
            logger.info(f"  - {model.name}")
        
        return model_files
    
    def discover_test_data(self) -> List[Path]:
        """テストデータを発見"""
        data_dirs = []
        
        # テストデータディレクトリ検索
        for potential_dir in self.data_dir.iterdir():
            if potential_dir.is_dir():
                # 画像ファイルが含まれているかチェック
                image_files = []
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    image_files.extend(potential_dir.glob(f"*{ext}"))
                    image_files.extend(potential_dir.glob(f"*{ext.upper()}"))
                
                if image_files:
                    data_dirs.append(potential_dir)
        
        logger.info(f"発見されたデータセット数: {len(data_dirs)}")
        for data_dir in data_dirs:
            logger.info(f"  - {data_dir.name}")
        
        return data_dirs
    
    def benchmark_model(self, model_path: Path, test_data_dir: Path) -> Dict:
        """単一モデルのベンチマーク"""
        logger.info(
            "モデルベンチマーク開始",
            model=model_path.name,
            test_data=test_data_dir.name
        )
        
        try:
            # モデル評価実行
            evaluator = ModelEvaluator(str(model_path), str(test_data_dir))
            
            # 軽量版評価（Docker環境での高速実行）
            results = {}
            
            # 精度評価
            results["accuracy"] = evaluator.evaluate_accuracy()
            
            # 速度評価（サンプル数削減）
            results["speed"] = evaluator.evaluate_speed(num_samples=20)
            
            # リソース評価（期間短縮）
            results["resource"] = evaluator.evaluate_resource_usage(duration=30)
            
            # 基本情報追加
            results["model_info"] = {
                "name": model_path.name,
                "size_mb": model_path.stat().st_size / (1024 * 1024),
                "test_dataset": test_data_dir.name
            }
            
            logger.info(
                "モデルベンチマーク完了",
                model=model_path.name,
                success_rate=results.get("accuracy", {}).get("success_rate", 0),
                avg_fps=results.get("speed", {}).get("single_inference", {}).get("avg_fps", 0)
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "モデルベンチマークエラー",
                model=model_path.name,
                error=str(e)
            )
            return {
                "error": str(e),
                "model_info": {
                    "name": model_path.name,
                    "size_mb": model_path.stat().st_size / (1024 * 1024),
                    "test_dataset": test_data_dir.name
                }
            }
    
    def run_quantization_benchmark(self, model_path: Path) -> Dict:
        """量子化ベンチマーク"""
        logger.info(f"量子化ベンチマーク開始: {model_path.name}")
        
        try:
            quantizer = ONNXQuantizer(str(model_path))
            
            # 一時出力ディレクトリ
            temp_output = Path("temp_quantized")
            temp_output.mkdir(exist_ok=True)
            
            # 動的量子化のみ実行（高速化）
            dynamic_output = temp_output / f"{model_path.stem}_dynamic.onnx"
            
            result = quantizer.quantize_dynamic(str(dynamic_output))
            
            # 一時ファイル削除
            if dynamic_output.exists():
                dynamic_output.unlink()
            
            logger.info(
                "量子化ベンチマーク完了",
                model=model_path.name,
                compression_ratio=result.get("compression_ratio", 0)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "量子化ベンチマークエラー",
                model=model_path.name,
                error=str(e)
            )
            return {"error": str(e)}
    
    def run_full_benchmark(self) -> Dict:
        """完全ベンチマーク実行"""
        logger.info("完全ベンチマーク開始")
        start_time = time.time()
        
        # モデルとデータ発見
        models = self.discover_models()
        test_datasets = self.discover_test_data()
        
        if not models:
            logger.error("ベンチマーク対象モデルが見つかりません")
            return {"error": "No models found"}
        
        if not test_datasets:
            logger.error("テストデータが見つかりません")
            return {"error": "No test data found"}
        
        benchmark_results = {
            "summary": {
                "start_time": time.time(),
                "models_count": len(models),
                "datasets_count": len(test_datasets)
            },
            "model_benchmarks": {},
            "quantization_benchmarks": {},
            "comparison": {}
        }
        
        # 各モデルでベンチマーク実行
        for model_path in models:
            model_name = model_path.name
            benchmark_results["model_benchmarks"][model_name] = {}
            
            # 各テストデータセットでテスト
            for test_data_dir in test_datasets:
                dataset_name = test_data_dir.name
                
                # モデル評価
                result = self.benchmark_model(model_path, test_data_dir)
                benchmark_results["model_benchmarks"][model_name][dataset_name] = result
            
            # 量子化ベンチマーク
            quant_result = self.run_quantization_benchmark(model_path)
            benchmark_results["quantization_benchmarks"][model_name] = quant_result
        
        # ベンチマーク比較分析
        benchmark_results["comparison"] = self.analyze_benchmarks(
            benchmark_results["model_benchmarks"]
        )
        
        # 完了情報
        total_time = time.time() - start_time
        benchmark_results["summary"]["total_time"] = total_time
        benchmark_results["summary"]["end_time"] = time.time()
        
        logger.info(
            "完全ベンチマーク完了",
            total_time=f"{total_time:.2f}s",
            models_tested=len(models),
            datasets_used=len(test_datasets)
        )
        
        return benchmark_results
    
    def analyze_benchmarks(self, model_benchmarks: Dict) -> Dict:
        """ベンチマーク結果分析"""
        analysis = {
            "best_accuracy": {"model": "", "dataset": "", "success_rate": 0},
            "best_speed": {"model": "", "dataset": "", "fps": 0},
            "best_efficiency": {"model": "", "dataset": "", "efficiency_score": 0},
            "model_rankings": []
        }
        
        model_scores = {}
        
        for model_name, datasets in model_benchmarks.items():
            model_scores[model_name] = {
                "avg_success_rate": 0,
                "avg_fps": 0,
                "avg_cpu_usage": 0,
                "avg_memory_usage": 0,
                "efficiency_score": 0
            }
            
            success_rates = []
            fps_values = []
            cpu_values = []
            memory_values = []
            
            for dataset_name, result in datasets.items():
                if "error" in result:
                    continue
                
                # 精度データ
                if "accuracy" in result and "success_rate" in result["accuracy"]:
                    success_rate = result["accuracy"]["success_rate"]
                    success_rates.append(success_rate)
                    
                    if success_rate > analysis["best_accuracy"]["success_rate"]:
                        analysis["best_accuracy"] = {
                            "model": model_name,
                            "dataset": dataset_name,
                            "success_rate": success_rate
                        }
                
                # 速度データ
                if "speed" in result and "single_inference" in result["speed"]:
                    fps = result["speed"]["single_inference"].get("avg_fps", 0)
                    fps_values.append(fps)
                    
                    if fps > analysis["best_speed"]["fps"]:
                        analysis["best_speed"] = {
                            "model": model_name,
                            "dataset": dataset_name,
                            "fps": fps
                        }
                
                # リソースデータ
                if "resource" in result:
                    cpu_usage = result["resource"].get("cpu_usage", {}).get("avg", 0)
                    memory_usage = result["resource"].get("memory_usage", {}).get("avg", 0)
                    cpu_values.append(cpu_usage)
                    memory_values.append(memory_usage)
            
            # 平均値計算
            if success_rates:
                model_scores[model_name]["avg_success_rate"] = sum(success_rates) / len(success_rates)
            if fps_values:
                model_scores[model_name]["avg_fps"] = sum(fps_values) / len(fps_values)
            if cpu_values:
                model_scores[model_name]["avg_cpu_usage"] = sum(cpu_values) / len(cpu_values)
            if memory_values:
                model_scores[model_name]["avg_memory_usage"] = sum(memory_values) / len(memory_values)
            
            # 効率スコア計算（精度 × 速度 / リソース使用量）
            if (model_scores[model_name]["avg_success_rate"] > 0 and
                model_scores[model_name]["avg_fps"] > 0 and
                model_scores[model_name]["avg_cpu_usage"] > 0):
                
                efficiency = (
                    model_scores[model_name]["avg_success_rate"] * 
                    model_scores[model_name]["avg_fps"]
                ) / (model_scores[model_name]["avg_cpu_usage"] / 100)
                
                model_scores[model_name]["efficiency_score"] = efficiency
                
                if efficiency > analysis["best_efficiency"]["efficiency_score"]:
                    analysis["best_efficiency"] = {
                        "model": model_name,
                        "dataset": "average",
                        "efficiency_score": efficiency
                    }
        
        # モデルランキング作成
        analysis["model_rankings"] = sorted(
            [(name, scores) for name, scores in model_scores.items()],
            key=lambda x: x[1]["efficiency_score"],
            reverse=True
        )
        
        return analysis
    
    def save_benchmark_results(self, results: Dict, output_path: str):
        """ベンチマーク結果保存"""
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"ベンチマーク結果保存: {output_path}")
    
    def print_benchmark_summary(self, results: Dict):
        """ベンチマーク結果サマリー表示"""
        print("\n" + "="*60)
        print("🚀 YOLOv7n MIT Edition - ベンチマーク結果")
        print("="*60)
        
        summary = results.get("summary", {})
        print(f"📊 実行サマリー:")
        print(f"  テストモデル数: {summary.get('models_count', 0)}")
        print(f"  テストデータセット数: {summary.get('datasets_count', 0)}")
        print(f"  総実行時間: {summary.get('total_time', 0):.2f}秒")
        
        comparison = results.get("comparison", {})
        
        if "best_accuracy" in comparison:
            best_acc = comparison["best_accuracy"]
            print(f"\n🎯 最高精度:")
            print(f"  モデル: {best_acc.get('model', 'N/A')}")
            print(f"  成功率: {best_acc.get('success_rate', 0):.1f}%")
        
        if "best_speed" in comparison:
            best_speed = comparison["best_speed"]
            print(f"\n⚡ 最高速度:")
            print(f"  モデル: {best_speed.get('model', 'N/A')}")
            print(f"  FPS: {best_speed.get('fps', 0):.1f}")
        
        if "best_efficiency" in comparison:
            best_eff = comparison["best_efficiency"]
            print(f"\n🏆 最高効率:")
            print(f"  モデル: {best_eff.get('model', 'N/A')}")
            print(f"  効率スコア: {best_eff.get('efficiency_score', 0):.2f}")
        
        # モデルランキング
        rankings = comparison.get("model_rankings", [])
        if rankings:
            print(f"\n📈 モデル効率ランキング:")
            for i, (model_name, scores) in enumerate(rankings[:3], 1):
                print(f"  {i}. {model_name}")
                print(f"     精度: {scores['avg_success_rate']:.1f}%")
                print(f"     速度: {scores['avg_fps']:.1f} FPS")
                print(f"     効率: {scores['efficiency_score']:.2f}")
        
        print("\n" + "="*60)

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv7n包括的ベンチマーク")
    parser.add_argument("--models-dir", default="models", help="モデルディレクトリ")
    parser.add_argument("--data-dir", default="data", help="データディレクトリ")
    parser.add_argument("--output", default="analysis/benchmark_results.json", help="結果出力ファイル")
    
    args = parser.parse_args()
    
    # ベンチマーク実行
    benchmark_suite = BenchmarkSuite(args.models_dir, args.data_dir)
    results = benchmark_suite.run_full_benchmark()
    
    # 結果保存
    benchmark_suite.save_benchmark_results(results, args.output)
    
    # 結果表示
    benchmark_suite.print_benchmark_summary(results)
    
    print(f"\n詳細結果: {args.output}")

if __name__ == "__main__":
    main()