"""
YOLOv7n MIT Edition - 包括的評価レポート生成
学習済みモデルの精度・速度・リソース使用量の自動評価
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
from jinja2 import Template

from inference.batch_images import BatchImageInference
from inference.logger import get_logger
from inference.single_image import SingleImageInference

logger = get_logger("evaluation")

class ModelEvaluator:
    """モデル評価クラス"""
    
    def __init__(self, model_path: str, test_data_dir: str):
        """
        評価器初期化
        
        Args:
            model_path: 評価するONNXモデルパス
            test_data_dir: テストデータディレクトリ
        """
        self.model_path = Path(model_path)
        self.test_data_dir = Path(test_data_dir)
        
        # 推論器初期化
        self.single_inferencer = SingleImageInference(
            model_path=str(self.model_path),
            confidence_threshold=0.5
        )
        
        self.batch_inferencer = BatchImageInference(
            model_path=str(self.model_path),
            confidence_threshold=0.5,
            max_workers=4
        )
        
        logger.info(
            "モデル評価器初期化完了",
            model_path=str(self.model_path),
            test_data_dir=str(self.test_data_dir)
        )
    
    def evaluate_accuracy(self) -> Dict:
        """精度評価"""
        logger.info("精度評価開始")
        
        # テスト画像で推論実行
        test_images = list(self.test_data_dir.glob("*.jpg")) + \
                     list(self.test_data_dir.glob("*.png"))
        
        if not test_images:
            logger.warning("テスト画像が見つかりません")
            return {"error": "No test images found"}
        
        # バッチ推論実行
        batch_result = self.batch_inferencer.predict_directory(
            str(self.test_data_dir),
            parallel=True,
            save_results=False
        )
        
        # 統計情報
        stats = batch_result["statistics"]
        
        accuracy_metrics = {
            "total_images": stats["total_images"],
            "successful_images": stats["successful_images"],
            "success_rate": stats["successful_images"] / stats["total_images"] * 100,
            "total_detections": stats["total_detections"],
            "avg_detections_per_image": stats["avg_detections_per_image"],
            "detection_distribution": self._analyze_detection_distribution(batch_result["results"])
        }
        
        logger.info(
            "精度評価完了",
            success_rate=f"{accuracy_metrics['success_rate']:.1f}%",
            total_detections=accuracy_metrics["total_detections"]
        )
        
        return accuracy_metrics
    
    def _analyze_detection_distribution(self, results: List[Dict]) -> Dict:
        """検出数分布分析"""
        detection_counts = [r.get("num_detections", 0) for r in results if "error" not in r]
        
        if not detection_counts:
            return {}
        
        return {
            "min_detections": min(detection_counts),
            "max_detections": max(detection_counts),
            "median_detections": np.median(detection_counts),
            "std_detections": np.std(detection_counts),
            "detection_counts": detection_counts
        }
    
    def evaluate_speed(self, num_samples: int = 100) -> Dict:
        """速度評価"""
        logger.info(f"速度評価開始 (サンプル数: {num_samples})")
        
        # テスト画像準備
        test_images = list(self.test_data_dir.glob("*.jpg")) + \
                     list(self.test_data_dir.glob("*.png"))
        
        if not test_images:
            return {"error": "No test images found"}
        
        # サンプル選択
        sample_images = (test_images * (num_samples // len(test_images) + 1))[:num_samples]
        
        # 単一画像推論速度測定
        single_times = []
        for image_path in sample_images[:min(50, len(sample_images))]:
            result = self.single_inferencer.predict(str(image_path))
            single_times.append(result["timing"]["total_time"])
        
        # バッチ推論速度測定
        batch_result = self.batch_inferencer.predict_batch_parallel(
            [str(img) for img in sample_images]
        )
        
        batch_stats = self.batch_inferencer.calculate_batch_statistics(
            batch_result, 
            sum(r.get("timing", {}).get("total_time", 0) for r in batch_result)
        )
        
        speed_metrics = {
            "single_inference": {
                "avg_time": np.mean(single_times),
                "min_time": np.min(single_times),
                "max_time": np.max(single_times),
                "std_time": np.std(single_times),
                "avg_fps": 1.0 / np.mean(single_times),
                "times": single_times
            },
            "batch_inference": {
                "total_time": batch_stats["total_time"],
                "avg_time": batch_stats["avg_inference_time"],
                "throughput_fps": batch_stats["throughput_fps"],
                "parallel_efficiency": batch_stats["throughput_fps"] / (1.0 / np.mean(single_times))
            },
            "num_samples": num_samples
        }
        
        logger.info(
            "速度評価完了",
            single_avg_time=f"{speed_metrics['single_inference']['avg_time']:.3f}s",
            single_avg_fps=f"{speed_metrics['single_inference']['avg_fps']:.1f} FPS",
            batch_throughput=f"{speed_metrics['batch_inference']['throughput_fps']:.1f} FPS"
        )
        
        return speed_metrics
    
    def evaluate_resource_usage(self, duration: int = 60) -> Dict:
        """リソース使用量評価"""
        logger.info(f"リソース使用量評価開始 (期間: {duration}秒)")
        
        # テスト画像準備
        test_images = list(self.test_data_dir.glob("*.jpg")) + \
                     list(self.test_data_dir.glob("*.png"))
        
        if not test_images:
            return {"error": "No test images found"}
        
        # リソース監視データ
        cpu_usage = []
        memory_usage = []
        timestamps = []
        
        start_time = time.time()
        
        # 継続的推論とリソース監視
        while time.time() - start_time < duration:
            # 推論実行
            test_image = np.random.choice(test_images)
            self.single_inferencer.predict(str(test_image))
            
            # リソース使用量記録
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            
            cpu_usage.append(cpu_percent)
            memory_usage.append(memory_info.percent)
            timestamps.append(time.time() - start_time)
            
            time.sleep(0.1)  # 100ms間隔
        
        resource_metrics = {
            "duration": duration,
            "cpu_usage": {
                "avg": np.mean(cpu_usage),
                "max": np.max(cpu_usage),
                "min": np.min(cpu_usage),
                "std": np.std(cpu_usage),
                "values": cpu_usage
            },
            "memory_usage": {
                "avg": np.mean(memory_usage),
                "max": np.max(memory_usage), 
                "min": np.min(memory_usage),
                "std": np.std(memory_usage),
                "values": memory_usage
            },
            "timestamps": timestamps,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3)
            }
        }
        
        logger.info(
            "リソース使用量評価完了",
            avg_cpu=f"{resource_metrics['cpu_usage']['avg']:.1f}%",
            max_cpu=f"{resource_metrics['cpu_usage']['max']:.1f}%",
            avg_memory=f"{resource_metrics['memory_usage']['avg']:.1f}%"
        )
        
        return resource_metrics
    
    def generate_visualizations(self, evaluation_results: Dict, output_dir: str):
        """評価結果の可視化"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
        # 1. 検出数分布
        if "accuracy" in evaluation_results and "detection_distribution" in evaluation_results["accuracy"]:
            detection_counts = evaluation_results["accuracy"]["detection_distribution"].get("detection_counts", [])
            if detection_counts:
                plt.figure(figsize=(10, 6))
                plt.hist(detection_counts, bins=20, alpha=0.7, color='skyblue')
                plt.xlabel('検出数')
                plt.ylabel('頻度')
                plt.title('検出数分布')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'detection_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. 推論時間分布
        if "speed" in evaluation_results and "single_inference" in evaluation_results["speed"]:
            times = evaluation_results["speed"]["single_inference"].get("times", [])
            if times:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.hist(times, bins=30, alpha=0.7, color='lightgreen')
                plt.xlabel('推論時間 (秒)')
                plt.ylabel('頻度')
                plt.title('推論時間分布')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.plot(times)
                plt.xlabel('サンプル番号')
                plt.ylabel('推論時間 (秒)')
                plt.title('推論時間の変化')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / 'inference_time_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. リソース使用量
        if "resource" in evaluation_results:
            resource_data = evaluation_results["resource"]
            timestamps = resource_data.get("timestamps", [])
            cpu_values = resource_data.get("cpu_usage", {}).get("values", [])
            memory_values = resource_data.get("memory_usage", {}).get("values", [])
            
            if timestamps and cpu_values and memory_values:
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 1, 1)
                plt.plot(timestamps, cpu_values, color='red', alpha=0.7)
                plt.fill_between(timestamps, cpu_values, alpha=0.3, color='red')
                plt.xlabel('時間 (秒)')
                plt.ylabel('CPU使用率 (%)')
                plt.title('CPU使用率の推移')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 1, 2)
                plt.plot(timestamps, memory_values, color='blue', alpha=0.7)
                plt.fill_between(timestamps, memory_values, alpha=0.3, color='blue')
                plt.xlabel('時間 (秒)')
                plt.ylabel('メモリ使用率 (%)')
                plt.title('メモリ使用率の推移')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / 'resource_usage.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"可視化グラフ生成完了: {output_dir}")
    
    def generate_html_report(
        self, 
        evaluation_results: Dict, 
        output_path: str,
        visualization_dir: Optional[str] = None
    ):
        """HTML評価レポート生成"""
        
        # HTMLテンプレート
        html_template = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv7n MIT Edition - 評価レポート</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #ecf0f1; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }
        .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; font-size: 0.9em; }
        .chart { text-align: center; margin: 20px 0; }
        .chart img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .summary { background: #d5dbdb; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .error { color: #e74c3c; background: #fadbd8; padding: 10px; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLOv7n MIT Edition - 評価レポート</h1>
        
        <div class="summary">
            <h2>評価サマリー</h2>
            <p><strong>モデル:</strong> {{ model_name }}</p>
            <p><strong>評価日時:</strong> {{ evaluation_date }}</p>
            <p><strong>テストデータ:</strong> {{ test_data_info }}</p>
        </div>

        {% if accuracy %}
        <h2>🎯 精度評価</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(accuracy.success_rate) }}%</div>
                <div class="metric-label">成功率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ accuracy.total_detections }}</div>
                <div class="metric-label">総検出数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.2f"|format(accuracy.avg_detections_per_image) }}</div>
                <div class="metric-label">平均検出数/画像</div>
            </div>
        </div>
        
        {% if visualization_dir %}
        <div class="chart">
            <img src="{{ visualization_dir }}/detection_distribution.png" alt="検出数分布">
        </div>
        {% endif %}
        {% endif %}

        {% if speed %}
        <h2>⚡ 速度評価</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ "%.3f"|format(speed.single_inference.avg_time) }}s</div>
                <div class="metric-label">平均推論時間</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(speed.single_inference.avg_fps) }}</div>
                <div class="metric-label">FPS (単一)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(speed.batch_inference.throughput_fps) }}</div>
                <div class="metric-label">スループット (バッチ)</div>
            </div>
        </div>
        
        {% if visualization_dir %}
        <div class="chart">
            <img src="{{ visualization_dir }}/inference_time_analysis.png" alt="推論時間分析">
        </div>
        {% endif %}
        {% endif %}

        {% if resource %}
        <h2>💻 リソース使用量</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(resource.cpu_usage.avg) }}%</div>
                <div class="metric-label">平均CPU使用率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(resource.memory_usage.avg) }}%</div>
                <div class="metric-label">平均メモリ使用率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(resource.system_info.memory_total_gb) }}GB</div>
                <div class="metric-label">総メモリ</div>
            </div>
        </div>
        
        {% if visualization_dir %}
        <div class="chart">
            <img src="{{ visualization_dir }}/resource_usage.png" alt="リソース使用量">
        </div>
        {% endif %}
        {% endif %}

        <h2>📊 詳細データ</h2>
        <table>
            <tr><th>項目</th><th>値</th></tr>
            {% if accuracy %}
            <tr><td>処理成功画像数</td><td>{{ accuracy.successful_images }} / {{ accuracy.total_images }}</td></tr>
            {% endif %}
            {% if speed %}
            <tr><td>評価サンプル数</td><td>{{ speed.num_samples }}</td></tr>
            <tr><td>最短推論時間</td><td>{{ "%.3f"|format(speed.single_inference.min_time) }}s</td></tr>
            <tr><td>最長推論時間</td><td>{{ "%.3f"|format(speed.single_inference.max_time) }}s</td></tr>
            {% endif %}
            {% if resource %}
            <tr><td>CPU最大使用率</td><td>{{ "%.1f"|format(resource.cpu_usage.max) }}%</td></tr>
            <tr><td>メモリ最大使用率</td><td>{{ "%.1f"|format(resource.memory_usage.max) }}%</td></tr>
            <tr><td>評価期間</td><td>{{ resource.duration }}秒</td></tr>
            {% endif %}
        </table>

        <div class="summary">
            <h2>📝 評価まとめ</h2>
            <p>このYOLOv7n MIT Editionモデルは以下の特徴を示しています：</p>
            <ul>
                {% if accuracy and accuracy.success_rate > 90 %}
                <li>✅ 高い推論成功率 ({{ "%.1f"|format(accuracy.success_rate) }}%)</li>
                {% endif %}
                {% if speed and speed.single_inference.avg_fps > 10 %}
                <li>✅ リアルタイム処理可能な推論速度 ({{ "%.1f"|format(speed.single_inference.avg_fps) }} FPS)</li>
                {% endif %}
                {% if resource and resource.cpu_usage.avg < 70 %}
                <li>✅ 効率的なCPU使用率 (平均{{ "%.1f"|format(resource.cpu_usage.avg) }}%)</li>
                {% endif %}
                {% if speed and speed.batch_inference.parallel_efficiency > 2 %}
                <li>✅ バッチ処理による高い並列効率</li>
                {% endif %}
            </ul>
        </div>
        
        <p style="text-align: center; color: #7f8c8d; margin-top: 40px;">
            Generated by YOLOv7n MIT Edition Evaluation System<br>
            {{ evaluation_date }}
        </p>
    </div>
</body>
</html>
        """
        
        # テンプレート変数準備
        template_vars = {
            "model_name": self.model_path.name,
            "evaluation_date": datetime.now().strftime("%Y年%m月%d日 %H:%M:%S"),
            "test_data_info": f"{self.test_data_dir.name} ディレクトリ",
            "accuracy": evaluation_results.get("accuracy"),
            "speed": evaluation_results.get("speed"),
            "resource": evaluation_results.get("resource"),
            "visualization_dir": visualization_dir
        }
        
        # HTMLレンダリング
        template = Template(html_template)
        html_content = template.render(**template_vars)
        
        # ファイル保存
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML評価レポート生成完了: {output_path}")
    
    def run_full_evaluation(self, output_dir: str) -> Dict:
        """完全評価実行"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("完全評価開始")
        
        # 各種評価実行
        evaluation_results = {}
        
        try:
            evaluation_results["accuracy"] = self.evaluate_accuracy()
        except Exception as e:
            logger.error(f"精度評価エラー: {e}")
            evaluation_results["accuracy"] = {"error": str(e)}
        
        try:
            evaluation_results["speed"] = self.evaluate_speed()
        except Exception as e:
            logger.error(f"速度評価エラー: {e}")
            evaluation_results["speed"] = {"error": str(e)}
        
        try:
            evaluation_results["resource"] = self.evaluate_resource_usage()
        except Exception as e:
            logger.error(f"リソース評価エラー: {e}")
            evaluation_results["resource"] = {"error": str(e)}
        
        # 結果保存
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        # 可視化生成
        viz_dir = output_dir / "visualizations"
        self.generate_visualizations(evaluation_results, viz_dir)
        
        # HTMLレポート生成
        html_report = output_dir / "evaluation_report.html"
        self.generate_html_report(
            evaluation_results, 
            html_report,
            visualization_dir="visualizations"
        )
        
        logger.info(
            "完全評価完了",
            output_dir=str(output_dir),
            results_file=str(results_file),
            html_report=str(html_report)
        )
        
        return evaluation_results

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv7n評価レポート生成")
    parser.add_argument("--model", required=True, help="評価するONNXモデルパス")
    parser.add_argument("--test-data", required=True, help="テストデータディレクトリ")
    parser.add_argument("--output", default="analysis/evaluation", help="出力ディレクトリ")
    parser.add_argument("--speed-samples", type=int, default=100, help="速度評価サンプル数")
    parser.add_argument("--resource-duration", type=int, default=60, help="リソース評価期間(秒)")
    
    args = parser.parse_args()
    
    # 評価実行
    evaluator = ModelEvaluator(args.model, args.test_data)
    results = evaluator.run_full_evaluation(args.output)
    
    # 結果表示
    print(f"\n=== 評価結果サマリー ===")
    
    if "accuracy" in results and "error" not in results["accuracy"]:
        acc = results["accuracy"]
        print(f"精度評価:")
        print(f"  成功率: {acc['success_rate']:.1f}%")
        print(f"  総検出数: {acc['total_detections']}")
    
    if "speed" in results and "error" not in results["speed"]:
        speed = results["speed"]
        print(f"速度評価:")
        print(f"  平均推論時間: {speed['single_inference']['avg_time']:.3f}s")
        print(f"  平均FPS: {speed['single_inference']['avg_fps']:.1f}")
    
    if "resource" in results and "error" not in results["resource"]:
        resource = results["resource"]
        print(f"リソース評価:")
        print(f"  平均CPU使用率: {resource['cpu_usage']['avg']:.1f}%")
        print(f"  平均メモリ使用率: {resource['memory_usage']['avg']:.1f}%")
    
    print(f"\n詳細レポート: {args.output}/evaluation_report.html")

if __name__ == "__main__":
    main()