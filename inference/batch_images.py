"""
YOLOv7n MIT Edition - バッチ画像推論
複数画像の並列処理による高速バッチ推論システム
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .logger import inference_logger
from .single_image import SingleImageInference

class BatchImageInference:
    """バッチ画像推論クラス"""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        max_workers: int = 4,
        batch_size: int = 8
    ):
        """
        バッチ推論器初期化
        
        Args:
            model_path: ONNXモデルファイルパス
            confidence_threshold: 信頼度閾値
            nms_threshold: NMS閾値
            input_size: 入力画像サイズ
            max_workers: 並列処理数
            batch_size: バッチサイズ
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # 単一画像推論器を複数作成（並列処理用）
        self.inference_workers = []
        for _ in range(max_workers):
            worker = SingleImageInference(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                input_size=input_size
            )
            self.inference_workers.append(worker)
        
        inference_logger.info(
            "バッチ推論器初期化完了",
            model_path=model_path,
            max_workers=max_workers,
            batch_size=batch_size
        )
    
    def find_images(self, input_path: str) -> List[str]:
        """
        画像ファイル検索
        
        Args:
            input_path: 入力パス（ファイルまたはディレクトリ）
            
        Returns:
            画像ファイルパスリスト
        """
        input_path = Path(input_path)
        
        # サポートする画像拡張子
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        if input_path.is_file():
            if input_path.suffix.lower() in image_extensions:
                return [str(input_path)]
            else:
                raise ValueError(f"サポートされていない画像形式: {input_path.suffix}")
        
        elif input_path.is_dir():
            image_files = []
            for ext in image_extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
            # ファイル名でソート
            image_files = sorted([str(f) for f in image_files])
            
            if not image_files:
                raise ValueError(f"画像ファイルが見つかりません: {input_path}")
            
            return image_files
        
        else:
            raise ValueError(f"存在しないパス: {input_path}")
    
    def predict_batch_sequential(self, image_paths: List[str]) -> List[Dict]:
        """
        順次バッチ推論（メモリ使用量を抑制）
        
        Args:
            image_paths: 画像パスリスト
            
        Returns:
            推論結果リスト
        """
        results = []
        
        # プログレスバー付きで順次処理
        with tqdm(total=len(image_paths), desc="Sequential Inference") as pbar:
            for image_path in image_paths:
                try:
                    result = self.inference_workers[0].predict(image_path)
                    results.append(result)
                    
                    pbar.set_postfix({
                        'Detections': result['num_detections'],
                        'Time': f"{result['timing']['total_time']:.3f}s"
                    })
                    
                except Exception as e:
                    inference_logger.error(
                        "順次推論エラー",
                        image_path=image_path,
                        error=str(e)
                    )
                    # エラー結果も記録
                    results.append({
                        "image_path": image_path,
                        "error": str(e),
                        "detections": [],
                        "num_detections": 0
                    })
                
                pbar.update(1)
        
        return results
    
    def predict_batch_parallel(self, image_paths: List[str]) -> List[Dict]:
        """
        並列バッチ推論（高速処理）
        
        Args:
            image_paths: 画像パスリスト
            
        Returns:
            推論結果リスト
        """
        results = []
        
        # ThreadPoolExecutorで並列処理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 各ワーカーに推論タスクを分散
            futures = {}
            
            for i, image_path in enumerate(image_paths):
                worker_idx = i % self.max_workers
                worker = self.inference_workers[worker_idx]
                
                future = executor.submit(worker.predict, image_path)
                futures[future] = image_path
            
            # 結果収集（完了順）
            completed_results = {}
            
            with tqdm(total=len(image_paths), desc="Parallel Inference") as pbar:
                for future in as_completed(futures):
                    image_path = futures[future]
                    
                    try:
                        result = future.result()
                        completed_results[image_path] = result
                        
                        pbar.set_postfix({
                            'Detections': result['num_detections'],
                            'Time': f"{result['timing']['total_time']:.3f}s"
                        })
                        
                    except Exception as e:
                        inference_logger.error(
                            "並列推論エラー",
                            image_path=image_path,
                            error=str(e)
                        )
                        completed_results[image_path] = {
                            "image_path": image_path,
                            "error": str(e),
                            "detections": [],
                            "num_detections": 0
                        }
                    
                    pbar.update(1)
        
        # 元の順序で結果を整列
        results = [completed_results[path] for path in image_paths]
        
        return results
    
    def predict_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        parallel: bool = True,
        save_results: bool = True
    ) -> Dict:
        """
        ディレクトリ内全画像の推論
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 結果保存ディレクトリ
            parallel: 並列処理使用
            save_results: 結果保存
            
        Returns:
            統計情報付き推論結果
        """
        start_time = time.time()
        
        # 画像ファイル検索
        image_paths = self.find_images(input_dir)
        
        inference_logger.info(
            "ディレクトリ推論開始",
            input_dir=input_dir,
            num_images=len(image_paths),
            parallel=parallel
        )
        
        # バッチ推論実行
        if parallel:
            results = self.predict_batch_parallel(image_paths)
        else:
            results = self.predict_batch_sequential(image_paths)
        
        # 統計情報計算
        total_time = time.time() - start_time
        stats = self.calculate_batch_statistics(results, total_time)
        
        # 結果構築
        batch_result = {
            "input_dir": input_dir,
            "num_images": len(image_paths),
            "results": results,
            "statistics": stats,
            "processing_mode": "parallel" if parallel else "sequential"
        }
        
        # 結果保存
        if save_results and output_dir:
            self.save_batch_results(batch_result, output_dir)
        
        inference_logger.info(
            "ディレクトリ推論完了",
            input_dir=input_dir,
            num_images=len(image_paths),
            total_time=f"{total_time:.2f}s",
            avg_time=f"{stats['avg_inference_time']:.3f}s",
            total_detections=stats['total_detections']
        )
        
        return batch_result
    
    def calculate_batch_statistics(
        self,
        results: List[Dict],
        total_time: float
    ) -> Dict:
        """バッチ推論統計計算"""
        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]
        
        if not successful_results:
            return {
                "total_images": len(results),
                "successful_images": 0,
                "failed_images": len(failed_results),
                "total_detections": 0,
                "avg_detections_per_image": 0,
                "total_time": total_time,
                "avg_inference_time": 0,
                "throughput_fps": 0
            }
        
        # 統計計算
        total_detections = sum(r['num_detections'] for r in successful_results)
        inference_times = [r['timing']['total_time'] for r in successful_results]
        
        stats = {
            "total_images": len(results),
            "successful_images": len(successful_results),
            "failed_images": len(failed_results),
            "total_detections": total_detections,
            "avg_detections_per_image": total_detections / len(successful_results),
            "total_time": total_time,
            "avg_inference_time": np.mean(inference_times),
            "min_inference_time": np.min(inference_times),
            "max_inference_time": np.max(inference_times),
            "throughput_fps": len(successful_results) / total_time
        }
        
        return stats
    
    def save_batch_results(self, batch_result: Dict, output_dir: str):
        """バッチ結果保存"""
        import json
        from datetime import datetime
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # タイムスタンプ付きファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 詳細結果保存
        detailed_file = output_dir / f"batch_results_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(batch_result, f, ensure_ascii=False, indent=2)
        
        # 統計サマリー保存
        summary_file = output_dir / f"batch_summary_{timestamp}.json"
        summary = {
            "input_dir": batch_result["input_dir"],
            "processing_mode": batch_result["processing_mode"],
            "statistics": batch_result["statistics"],
            "timestamp": timestamp
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        inference_logger.info(
            "バッチ結果保存完了",
            detailed_file=str(detailed_file),
            summary_file=str(summary_file)
        )


def main():
    """バッチ推論テスト実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv7nバッチ画像推論")
    parser.add_argument("--model", required=True, help="ONNXモデルパス")
    parser.add_argument("--input", required=True, help="入力ディレクトリ")
    parser.add_argument("--output", help="結果保存ディレクトリ")
    parser.add_argument("--conf", type=float, default=0.5, help="信頼度閾値")
    parser.add_argument("--workers", type=int, default=4, help="並列処理数")
    parser.add_argument("--sequential", action="store_true", help="順次処理モード")
    
    args = parser.parse_args()
    
    # バッチ推論実行
    batch_inferencer = BatchImageInference(
        model_path=args.model,
        confidence_threshold=args.conf,
        max_workers=args.workers
    )
    
    # ディレクトリ推論
    result = batch_inferencer.predict_directory(
        input_dir=args.input,
        output_dir=args.output,
        parallel=not args.sequential,
        save_results=bool(args.output)
    )
    
    # 結果表示
    stats = result["statistics"]
    print(f"\n=== バッチ推論結果 ===")
    print(f"処理画像数: {stats['successful_images']}/{stats['total_images']}")
    print(f"総検出数: {stats['total_detections']}")
    print(f"平均検出数/画像: {stats['avg_detections_per_image']:.2f}")
    print(f"処理時間: {stats['total_time']:.2f}s")
    print(f"平均推論時間: {stats['avg_inference_time']:.3f}s")
    print(f"スループット: {stats['throughput_fps']:.2f} FPS")
    
    if stats['failed_images'] > 0:
        print(f"失敗画像数: {stats['failed_images']}")


if __name__ == "__main__":
    main()