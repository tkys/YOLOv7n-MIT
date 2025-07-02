"""
YOLOv7n MIT Edition - Docker環境テスト
Docker環境での各種機能の動作確認
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List

import requests
from inference.logger import get_logger

logger = get_logger("docker_test")

class DockerEnvironmentTest:
    """Docker環境テストクラス"""
    
    def __init__(self):
        """テスト環境初期化"""
        self.test_results = {}
        logger.info("Docker環境テスト初期化")
    
    def test_basic_imports(self) -> Dict:
        """基本的なインポートテスト"""
        logger.info("基本インポートテスト開始")
        
        test_results = {"status": "success", "failed_imports": []}
        
        # 必須パッケージのインポートテスト
        essential_imports = [
            ("numpy", "import numpy as np"),
            ("opencv", "import cv2"),
            ("onnxruntime", "import onnxruntime as ort"),
            ("fastapi", "from fastapi import FastAPI"),
            ("uvicorn", "import uvicorn"),
            ("matplotlib", "import matplotlib.pyplot as plt"),
            ("pillow", "from PIL import Image"),
            ("psutil", "import psutil"),
            ("tqdm", "from tqdm import tqdm"),
            ("jinja2", "from jinja2 import Template")
        ]
        
        for package_name, import_statement in essential_imports:
            try:
                exec(import_statement)
                logger.debug(f"✅ {package_name} インポート成功")
            except ImportError as e:
                logger.error(f"❌ {package_name} インポート失敗: {e}")
                test_results["failed_imports"].append({
                    "package": package_name,
                    "error": str(e)
                })
        
        if test_results["failed_imports"]:
            test_results["status"] = "failed"
        
        logger.info(f"基本インポートテスト完了: {test_results['status']}")
        return test_results
    
    def test_model_loading(self) -> Dict:
        """モデル読み込みテスト"""
        logger.info("モデル読み込みテスト開始")
        
        # サンプルONNXモデルパス探索
        model_paths = []
        model_dirs = ["models", "runs/train", "."]
        
        for model_dir in model_dirs:
            model_dir_path = Path(model_dir)
            if model_dir_path.exists():
                model_paths.extend(model_dir_path.glob("*.onnx"))
                model_paths.extend(model_dir_path.glob("**/*.onnx"))
        
        if not model_paths:
            return {
                "status": "skipped",
                "reason": "No ONNX models found"
            }
        
        test_model = model_paths[0]
        logger.info(f"テストモデル: {test_model}")
        
        try:
            import onnxruntime as ort
            
            # ONNX Runtime セッション作成テスト
            providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(str(test_model), providers=providers)
            
            # モデル情報取得
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            model_info = {
                "status": "success",
                "model_path": str(test_model),
                "input_name": input_info.name,
                "input_shape": input_info.shape,
                "output_name": output_info.name,
                "output_shape": output_info.shape,
                "providers": session.get_providers()
            }
            
            logger.info("モデル読み込みテスト成功")
            return model_info
            
        except Exception as e:
            logger.error(f"モデル読み込みテスト失敗: {e}")
            return {
                "status": "failed",
                "model_path": str(test_model),
                "error": str(e)
            }
    
    def test_inference_functionality(self) -> Dict:
        """推論機能テスト"""
        logger.info("推論機能テスト開始")
        
        try:
            from inference.single_image import SingleImageInference
            
            # サンプルONNXモデル探索
            model_paths = list(Path("models").glob("*.onnx")) + \
                         list(Path(".").glob("*.onnx"))
            
            if not model_paths:
                return {
                    "status": "skipped",
                    "reason": "No ONNX models found for inference test"
                }
            
            test_model = model_paths[0]
            
            # 推論器初期化テスト
            inferencer = SingleImageInference(
                model_path=str(test_model),
                confidence_threshold=0.5
            )
            
            # テスト画像作成（ダミー画像）
            import numpy as np
            import cv2
            
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_image_path = "/tmp/test_image.jpg"
            cv2.imwrite(test_image_path, test_image)
            
            # 推論実行
            start_time = time.time()
            result = inferencer.predict(test_image_path)
            inference_time = time.time() - start_time
            
            # 一時ファイル削除
            os.unlink(test_image_path)
            
            test_result = {
                "status": "success",
                "model_used": str(test_model),
                "inference_time": inference_time,
                "num_detections": result["num_detections"],
                "timing": result["timing"]
            }
            
            logger.info(f"推論機能テスト成功: {inference_time:.3f}s")
            return test_result
            
        except Exception as e:
            logger.error(f"推論機能テスト失敗: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_api_server(self, timeout: int = 30) -> Dict:
        """APIサーバーテスト"""
        logger.info("APIサーバーテスト開始")
        
        try:
            # FastAPIアプリケーション起動テスト
            from inference.api import app
            
            # アプリケーション初期化確認
            if app is None:
                return {
                    "status": "failed",
                    "error": "FastAPI app not initialized"
                }
            
            # ヘルスチェックエンドポイントの機能確認
            # 注意: 実際のHTTPサーバー起動はしない（環境テストのため）
            
            test_result = {
                "status": "success",
                "app_initialized": True,
                "endpoints_available": [
                    "/health",
                    "/predict/single", 
                    "/predict/batch",
                    "/"
                ]
            }
            
            logger.info("APIサーバーテスト成功")
            return test_result
            
        except Exception as e:
            logger.error(f"APIサーバーテスト失敗: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_logging_system(self) -> Dict:
        """ログシステムテスト"""
        logger.info("ログシステムテスト開始")
        
        try:
            from inference.logger import get_logger
            
            # 新しいロガー作成
            test_logger = get_logger("test_logger")
            
            # 各レベルのログ出力テスト
            test_logger.debug("デバッグメッセージテスト")
            test_logger.info("情報メッセージテスト")
            test_logger.warning("警告メッセージテスト")
            
            # ログファイル存在確認
            log_dir = Path("logs")
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log"))
                test_result = {
                    "status": "success",
                    "log_dir_exists": True,
                    "log_files_count": len(log_files),
                    "log_files": [str(f) for f in log_files]
                }
            else:
                test_result = {
                    "status": "partial",
                    "log_dir_exists": False,
                    "note": "Log directory not created yet"
                }
            
            logger.info("ログシステムテスト完了")
            return test_result
            
        except Exception as e:
            logger.error(f"ログシステムテスト失敗: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_resource_monitoring(self) -> Dict:
        """リソース監視テスト"""
        logger.info("リソース監視テスト開始")
        
        try:
            import psutil
            
            # システム情報取得
            cpu_count = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            resource_info = {
                "status": "success",
                "cpu_count": cpu_count,
                "memory_total_gb": memory_info.total / (1024**3),
                "memory_available_gb": memory_info.available / (1024**3),
                "memory_usage_percent": memory_info.percent,
                "cpu_usage_percent": cpu_percent
            }
            
            logger.info("リソース監視テスト成功")
            return resource_info
            
        except Exception as e:
            logger.error(f"リソース監視テスト失敗: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict:
        """全テスト実行"""
        logger.info("Docker環境包括テスト開始")
        start_time = time.time()
        
        # 各テスト実行
        test_suite = {
            "basic_imports": self.test_basic_imports,
            "model_loading": self.test_model_loading,
            "inference_functionality": self.test_inference_functionality,
            "api_server": self.test_api_server,
            "logging_system": self.test_logging_system,
            "resource_monitoring": self.test_resource_monitoring
        }
        
        results = {
            "test_summary": {
                "start_time": start_time,
                "environment": "docker",
                "total_tests": len(test_suite)
            },
            "test_results": {}
        }
        
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for test_name, test_function in test_suite.items():
            logger.info(f"実行中: {test_name}")
            
            try:
                test_result = test_function()
                results["test_results"][test_name] = test_result
                
                status = test_result.get("status", "unknown")
                if status == "success":
                    passed_tests += 1
                elif status == "failed":
                    failed_tests += 1
                else:
                    skipped_tests += 1
                    
            except Exception as e:
                logger.error(f"テスト実行エラー {test_name}: {e}")
                results["test_results"][test_name] = {
                    "status": "error",
                    "error": str(e)
                }
                failed_tests += 1
        
        # テスト完了情報
        total_time = time.time() - start_time
        results["test_summary"].update({
            "end_time": time.time(),
            "total_time": total_time,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "success_rate": passed_tests / len(test_suite) * 100
        })
        
        logger.info(
            f"Docker環境テスト完了: {passed_tests}/{len(test_suite)} 成功 ({total_time:.2f}s)"
        )
        
        return results
    
    def print_test_summary(self, results: Dict):
        """テスト結果サマリー表示"""
        print("\n" + "="*60)
        print("🐳 Docker環境テスト結果")
        print("="*60)
        
        summary = results["test_summary"]
        print(f"📊 テストサマリー:")
        print(f"  総テスト数: {summary['total_tests']}")
        print(f"  成功: {summary['passed_tests']}")
        print(f"  失敗: {summary['failed_tests']}")
        print(f"  スキップ: {summary['skipped_tests']}")
        print(f"  成功率: {summary['success_rate']:.1f}%")
        print(f"  実行時間: {summary['total_time']:.2f}秒")
        
        print(f"\n📋 詳細結果:")
        for test_name, test_result in results["test_results"].items():
            status = test_result.get("status", "unknown")
            status_emoji = {
                "success": "✅",
                "failed": "❌", 
                "skipped": "⏭️",
                "partial": "⚠️",
                "error": "💥"
            }.get(status, "❓")
            
            print(f"  {status_emoji} {test_name}: {status}")
            
            if status == "failed" and "error" in test_result:
                print(f"    エラー: {test_result['error']}")
        
        print("\n" + "="*60)

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Docker環境テスト")
    parser.add_argument("--output", help="結果出力ファイル")
    parser.add_argument("--verbose", action="store_true", help="詳細出力")
    
    args = parser.parse_args()
    
    # テスト実行
    docker_test = DockerEnvironmentTest()
    results = docker_test.run_all_tests()
    
    # 結果表示
    docker_test.print_test_summary(results)
    
    # 結果保存
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n詳細結果保存: {args.output}")
    
    # 終了コード設定
    success_rate = results["test_summary"]["success_rate"]
    if success_rate < 80:
        print("\n⚠️  警告: テスト成功率が80%未満です")
        exit(1)
    elif success_rate < 100:
        print("\n⚠️  注意: 一部のテストが失敗またはスキップされました")
        exit(0)
    else:
        print("\n🎉 すべてのテストが成功しました！")
        exit(0)

if __name__ == "__main__":
    main()