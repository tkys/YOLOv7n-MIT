"""
YOLOv7n MIT Edition - 単一画像推論
CPU最適化ONNX Runtime使用の高速推論システム
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from .logger import inference_logger

class SingleImageInference:
    """単一画像推論クラス"""
    
    def __init__(
        self, 
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        推論器初期化
        
        Args:
            model_path: ONNXモデルファイルパス
            confidence_threshold: 信頼度閾値
            nms_threshold: NMS閾値
            input_size: 入力画像サイズ
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        # CPU最適化設定
        providers = [
            ('CPUExecutionProvider', {
                'intra_op_num_threads': 4,
                'execution_mode': ort.ExecutionMode.ORT_SEQUENTIAL,
                'graph_optimization_level': ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            })
        ]
        
        # ONNX Runtime セッション作成
        self.session = ort.InferenceSession(
            str(self.model_path), 
            providers=providers
        )
        
        # モデル情報取得
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        inference_logger.info(
            "単一画像推論器初期化完了",
            model_path=str(self.model_path),
            input_size=self.input_size,
            providers=[provider[0] for provider in providers]
        )
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """
        画像前処理
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            preprocessed_image: 前処理済み画像
            metadata: 元画像情報
        """
        start_time = time.time()
        
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像読み込み失敗: {image_path}")
        
        original_shape = image.shape[:2]  # (height, width)
        
        # RGB変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # リサイズ（アスペクト比保持）
        resized_image = self._resize_with_padding(image_rgb, self.input_size)
        
        # 正規化とテンソル変換
        normalized_image = resized_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized_image, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # バッチ次元追加
        
        preprocess_time = time.time() - start_time
        
        metadata = {
            "original_shape": original_shape,
            "input_shape": self.input_size,
            "preprocess_time": preprocess_time,
            "scale_ratio": self._calculate_scale_ratio(original_shape, self.input_size)
        }
        
        inference_logger.debug(
            "画像前処理完了",
            image_path=image_path,
            original_shape=original_shape,
            preprocess_time=f"{preprocess_time:.4f}s"
        )
        
        return input_tensor, metadata
    
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
    
    def _calculate_scale_ratio(
        self, 
        original_shape: Tuple[int, int], 
        input_shape: Tuple[int, int]
    ) -> float:
        """スケール比率計算"""
        return min(input_shape[0] / original_shape[0], input_shape[1] / original_shape[1])
    
    def inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        ONNX推論実行
        
        Args:
            input_tensor: 前処理済み入力テンソル
            
        Returns:
            推論結果
        """
        start_time = time.time()
        
        # ONNX Runtime推論
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        
        inference_time = time.time() - start_time
        
        inference_logger.debug(
            "ONNX推論完了",
            inference_time=f"{inference_time:.4f}s",
            output_shape=[output.shape for output in outputs]
        )
        
        return outputs[0], inference_time
    
    def postprocess(
        self, 
        outputs: np.ndarray, 
        metadata: Dict
    ) -> List[Dict]:
        """
        後処理（NMS等による検出結果整理）
        
        Args:
            outputs: 推論結果
            metadata: 画像メタデータ
            
        Returns:
            検出結果リスト
        """
        start_time = time.time()
        
        # バッチ次元除去
        if len(outputs.shape) == 3:
            outputs = outputs[0]
        
        # 信頼度フィルタリング
        valid_detections = outputs[outputs[:, 4] >= self.confidence_threshold]
        
        if len(valid_detections) == 0:
            return []
        
        # 座標変換（入力画像座標 → 元画像座標）
        scale_ratio = metadata["scale_ratio"]
        detections = self._rescale_boxes(valid_detections, scale_ratio, metadata)
        
        # NMS適用
        final_detections = self._apply_nms(detections)
        
        postprocess_time = time.time() - start_time
        
        inference_logger.debug(
            "後処理完了",
            raw_detections=len(valid_detections),
            final_detections=len(final_detections),
            postprocess_time=f"{postprocess_time:.4f}s"
        )
        
        return final_detections
    
    def _rescale_boxes(
        self, 
        detections: np.ndarray, 
        scale_ratio: float,
        metadata: Dict
    ) -> List[Dict]:
        """バウンディングボックス座標変換"""
        results = []
        
        for detection in detections:
            x_center, y_center, width, height, confidence = detection[:5]
            
            # 元画像座標に変換
            x_center /= scale_ratio
            y_center /= scale_ratio
            width /= scale_ratio
            height /= scale_ratio
            
            # xyxy形式に変換
            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(metadata["original_shape"][1], int(x_center + width / 2))
            y2 = min(metadata["original_shape"][0], int(y_center + height / 2))
            
            results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(confidence),
                "class_id": 0  # 単一クラス
            })
        
        return results
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Non-Maximum Suppression適用"""
        if len(detections) <= 1:
            return detections
        
        boxes = np.array([det["bbox"] for det in detections])
        scores = np.array([det["confidence"] for det in detections])
        
        # OpenCV NMS適用
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )
        
        if len(indices) == 0:
            return []
        
        return [detections[i] for i in indices.flatten()]
    
    def predict(self, image_path: str) -> Dict:
        """
        完整预测流程
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            予測結果
        """
        total_start_time = time.time()
        
        try:
            # 前処理
            input_tensor, metadata = self.preprocess_image(image_path)
            
            # 推論
            outputs, inference_time = self.inference(input_tensor)
            
            # 後処理
            detections = self.postprocess(outputs, metadata)
            
            total_time = time.time() - total_start_time
            
            result = {
                "image_path": image_path,
                "detections": detections,
                "num_detections": len(detections),
                "timing": {
                    "preprocess_time": metadata["preprocess_time"],
                    "inference_time": inference_time,
                    "total_time": total_time
                },
                "metadata": metadata
            }
            
            inference_logger.info(
                "単一画像推論完了",
                image_path=image_path,
                num_detections=len(detections),
                total_time=f"{total_time:.4f}s"
            )
            
            return result
            
        except Exception as e:
            inference_logger.error(
                "単一画像推論エラー",
                image_path=image_path,
                error=str(e)
            )
            raise


def main():
    """推論テスト実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv7n単一画像推論")
    parser.add_argument("--model", required=True, help="ONNXモデルパス")
    parser.add_argument("--image", required=True, help="入力画像パス")
    parser.add_argument("--output", help="結果保存パス")
    parser.add_argument("--conf", type=float, default=0.5, help="信頼度閾値")
    
    args = parser.parse_args()
    
    # 推論実行
    inferencer = SingleImageInference(
        model_path=args.model,
        confidence_threshold=args.conf
    )
    
    result = inferencer.predict(args.image)
    
    # 結果表示
    print(f"検出数: {result['num_detections']}")
    print(f"推論時間: {result['timing']['total_time']:.4f}s")
    
    for i, detection in enumerate(result['detections']):
        print(f"検出{i+1}: 信頼度={detection['confidence']:.3f}, "
              f"座標={detection['bbox']}")
    
    # 結果保存
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"結果保存: {args.output}")


if __name__ == "__main__":
    main()