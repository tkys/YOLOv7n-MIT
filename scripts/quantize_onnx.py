"""
YOLOv7n MIT Edition - ONNX量子化
INT8量子化によるモデルサイズ削減とCPU推論高速化
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
    quantize_dynamic
)
from PIL import Image
import cv2

from inference.logger import get_logger

logger = get_logger("quantization")

class YOLOCalibrationDataReader(CalibrationDataReader):
    """YOLO用キャリブレーションデータリーダー"""
    
    def __init__(
        self,
        calibration_image_folder: str,
        input_name: str,
        input_size: Tuple[int, int] = (640, 640),
        num_samples: int = 100
    ):
        """
        キャリブレーションデータリーダー初期化
        
        Args:
            calibration_image_folder: キャリブレーション用画像フォルダ
            input_name: モデル入力名
            input_size: 入力画像サイズ
            num_samples: キャリブレーションサンプル数
        """
        self.image_folder = Path(calibration_image_folder)
        self.input_name = input_name
        self.input_size = input_size
        self.num_samples = num_samples
        
        # 画像ファイル収集
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = []
        
        for ext in image_extensions:
            self.image_files.extend(self.image_folder.glob(f"*{ext}"))
            self.image_files.extend(self.image_folder.glob(f"*{ext.upper()}"))
        
        # サンプル数制限
        if len(self.image_files) > num_samples:
            self.image_files = self.image_files[:num_samples]
        
        logger.info(
            "キャリブレーションデータ準備完了",
            image_folder=str(self.image_folder),
            num_images=len(self.image_files),
            input_size=input_size
        )
        
        self.enum_data = None
    
    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """次のキャリブレーションデータ取得"""
        if self.enum_data is None:
            self.enum_data = iter(self.image_files)
        
        try:
            image_path = next(self.enum_data)
            input_data = self.preprocess_image(str(image_path))
            return {self.input_name: input_data}
        except StopIteration:
            return None
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """画像前処理（YOLO形式）"""
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像読み込み失敗: {image_path}")
        
        # RGB変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # リサイズ（アスペクト比保持）
        resized_image = self._resize_with_padding(image_rgb, self.input_size)
        
        # 正規化とテンソル変換
        normalized_image = resized_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized_image, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # バッチ次元追加
        
        return input_tensor
    
    def _resize_with_padding(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """アスペクト比保持リサイズ"""
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # スケール計算
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # リサイズ
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # パディング
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        return padded

class ONNXQuantizer:
    """ONNX量子化クラス"""
    
    def __init__(self, model_path: str):
        """
        量子化器初期化
        
        Args:
            model_path: 元ONNXモデルパス
        """
        self.model_path = Path(model_path)
        self.model = onnx.load(str(self.model_path))
        
        # モデル情報取得
        self.input_name = self.model.graph.input[0].name
        
        logger.info(
            "ONNX量子化器初期化完了",
            model_path=str(self.model_path),
            input_name=self.input_name
        )
    
    def quantize_dynamic(
        self,
        output_path: str,
        weight_type: QuantType = QuantType.QUInt8
    ) -> Dict:
        """
        動的量子化実行
        
        Args:
            output_path: 出力ONNXファイルパス
            weight_type: 重み量子化タイプ
            
        Returns:
            量子化結果情報
        """
        start_time = time.time()
        
        logger.info(
            "動的量子化開始",
            input_model=str(self.model_path),
            output_model=output_path,
            weight_type=str(weight_type)
        )
        
        try:
            # 動的量子化実行
            quantize_dynamic(
                model_input=str(self.model_path),
                model_output=output_path,
                weight_type=weight_type,
                op_types_to_quantize=['Conv', 'MatMul']
            )
            
            quantization_time = time.time() - start_time
            
            # ファイルサイズ比較
            original_size = self.model_path.stat().st_size
            quantized_size = Path(output_path).stat().st_size
            compression_ratio = original_size / quantized_size
            
            result = {
                "quantization_type": "dynamic",
                "quantization_time": quantization_time,
                "original_size_mb": original_size / (1024 * 1024),
                "quantized_size_mb": quantized_size / (1024 * 1024),
                "compression_ratio": compression_ratio,
                "size_reduction_percent": (1 - quantized_size / original_size) * 100
            }
            
            logger.info(
                "動的量子化完了",
                output_model=output_path,
                quantization_time=f"{quantization_time:.2f}s",
                compression_ratio=f"{compression_ratio:.2f}x",
                size_reduction=f"{result['size_reduction_percent']:.1f}%"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "動的量子化エラー",
                error=str(e)
            )
            raise
    
    def quantize_static(
        self,
        output_path: str,
        calibration_data_folder: str,
        activation_type: QuantType = QuantType.QUInt8,
        weight_type: QuantType = QuantType.QInt8,
        num_calibration_samples: int = 100
    ) -> Dict:
        """
        静的量子化実行
        
        Args:
            output_path: 出力ONNXファイルパス
            calibration_data_folder: キャリブレーション画像フォルダ
            activation_type: アクティベーション量子化タイプ
            weight_type: 重み量子化タイプ
            num_calibration_samples: キャリブレーションサンプル数
            
        Returns:
            量子化結果情報
        """
        start_time = time.time()
        
        logger.info(
            "静的量子化開始",
            input_model=str(self.model_path),
            output_model=output_path,
            calibration_folder=calibration_data_folder,
            num_samples=num_calibration_samples
        )
        
        try:
            # キャリブレーションデータリーダー作成
            calibration_data_reader = YOLOCalibrationDataReader(
                calibration_image_folder=calibration_data_folder,  
                input_name=self.input_name,
                num_samples=num_calibration_samples
            )
            
            # 静的量子化実行
            quantize_static(
                model_input=str(self.model_path),
                model_output=output_path,
                calibration_data_reader=calibration_data_reader,
                quant_format=QuantFormat.QOperator,
                activation_type=activation_type,
                weight_type=weight_type,
                op_types_to_quantize=['Conv', 'MatMul']
            )
            
            quantization_time = time.time() - start_time
            
            # ファイルサイズ比較
            original_size = self.model_path.stat().st_size
            quantized_size = Path(output_path).stat().st_size
            compression_ratio = original_size / quantized_size
            
            result = {
                "quantization_type": "static",
                "quantization_time": quantization_time,
                "original_size_mb": original_size / (1024 * 1024),
                "quantized_size_mb": quantized_size / (1024 * 1024),
                "compression_ratio": compression_ratio,
                "size_reduction_percent": (1 - quantized_size / original_size) * 100,
                "calibration_samples": num_calibration_samples
            }
            
            logger.info(
                "静的量子化完了",
                output_model=output_path,
                quantization_time=f"{quantization_time:.2f}s",
                compression_ratio=f"{compression_ratio:.2f}x",
                size_reduction=f"{result['size_reduction_percent']:.1f}%"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "静的量子化エラー",
                error=str(e)
            )
            raise
    
    def validate_quantized_model(
        self,
        quantized_model_path: str,
        test_image_path: str
    ) -> Dict:
        """
        量子化モデル検証
        
        Args:
            quantized_model_path: 量子化モデルパス
            test_image_path: テスト画像パス
            
        Returns:
            検証結果
        """
        logger.info(
            "量子化モデル検証開始",
            quantized_model=quantized_model_path,
            test_image=test_image_path
        )
        
        try:
            # CPU最適化設定
            providers = [
                ('CPUExecutionProvider', {
                    'intra_op_num_threads': 4,
                    'execution_mode': ort.ExecutionMode.ORT_SEQUENTIAL,
                    'graph_optimization_level': ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                })
            ]
            
            # 元モデルセッション
            original_session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            
            # 量子化モデルセッション
            quantized_session = ort.InferenceSession(
                quantized_model_path,
                providers=providers
            )
            
            # テスト画像前処理
            calibration_reader = YOLOCalibrationDataReader(
                calibration_image_folder=str(Path(test_image_path).parent),
                input_name=self.input_name,
                num_samples=1
            )
            
            test_input = calibration_reader.preprocess_image(test_image_path)
            
            # 推論時間測定
            # 元モデル
            start_time = time.time()
            original_output = original_session.run(None, {self.input_name: test_input})
            original_time = time.time() - start_time
            
            # 量子化モデル
            start_time = time.time()
            quantized_output = quantized_session.run(None, {self.input_name: test_input})
            quantized_time = time.time() - start_time
            
            # 精度比較（出力の平均絶対誤差）
            mae = np.mean(np.abs(original_output[0] - quantized_output[0]))
            
            result = {
                "original_inference_time": original_time,
                "quantized_inference_time": quantized_time,
                "speedup_ratio": original_time / quantized_time,
                "mean_absolute_error": float(mae),
                "accuracy_retention": 100.0 - (mae * 100)
            }
            
            logger.info(
                "量子化モデル検証完了",
                speedup_ratio=f"{result['speedup_ratio']:.2f}x",
                mae=f"{mae:.6f}",
                accuracy_retention=f"{result['accuracy_retention']:.2f}%"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "量子化モデル検証エラー",
                error=str(e)
            )
            raise

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="YOLOv7n ONNX量子化")
    parser.add_argument("--model", required=True, help="入力ONNXモデルパス")
    parser.add_argument("--output-dir", default="models/quantized", help="出力ディレクトリ")
    parser.add_argument("--calibration-data", help="キャリブレーション画像フォルダ（静的量子化用）")
    parser.add_argument("--test-image", help="検証用テスト画像")
    parser.add_argument("--dynamic-only", action="store_true", help="動的量子化のみ実行")
    parser.add_argument("--static-only", action="store_true", help="静的量子化のみ実行")
    parser.add_argument("--num-samples", type=int, default=100, help="キャリブレーションサンプル数")
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 量子化器初期化
    quantizer = ONNXQuantizer(args.model)
    model_name = Path(args.model).stem
    
    results = {}
    
    # 動的量子化
    if not args.static_only:
        dynamic_output = output_dir / f"{model_name}_dynamic_quantized.onnx"
        logger.info(f"動的量子化実行: {dynamic_output}")
        
        dynamic_result = quantizer.quantize_dynamic(str(dynamic_output))
        results["dynamic"] = dynamic_result
        
        # 検証
        if args.test_image:
            validation_result = quantizer.validate_quantized_model(
                str(dynamic_output),
                args.test_image
            )
            results["dynamic"]["validation"] = validation_result
    
    # 静的量子化
    if not args.dynamic_only and args.calibration_data:
        static_output = output_dir / f"{model_name}_static_quantized.onnx"
        logger.info(f"静的量子化実行: {static_output}")
        
        static_result = quantizer.quantize_static(
            str(static_output),
            args.calibration_data,
            num_calibration_samples=args.num_samples
        )
        results["static"] = static_result
        
        # 検証
        if args.test_image:
            validation_result = quantizer.validate_quantized_model(
                str(static_output),
                args.test_image
            )
            results["static"]["validation"] = validation_result
    
    # 結果保存
    import json
    results_file = output_dir / f"{model_name}_quantization_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 結果表示
    print("\n=== 量子化結果 ===")
    for quant_type, result in results.items():
        print(f"\n{quant_type.upper()}量子化:")
        print(f"  圧縮比: {result['compression_ratio']:.2f}x")
        print(f"  サイズ削減: {result['size_reduction_percent']:.1f}%")
        print(f"  量子化時間: {result['quantization_time']:.2f}s")
        
        if "validation" in result:
            val = result["validation"]
            print(f"  推論高速化: {val['speedup_ratio']:.2f}x")
            print(f"  精度保持: {val['accuracy_retention']:.2f}%")
    
    print(f"\n結果詳細: {results_file}")

if __name__ == "__main__":
    main()