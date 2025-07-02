#!/usr/bin/env python3
"""
YOLOv7n PyTorch → ONNX 変換スクリプト
GPU学習済みモデルをONNX形式に変換して推論最適化
"""
import torch
import os
import sys
import time
from pathlib import Path
import numpy as np

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_trained_model():
    """学習済みモデルロード"""
    print("=== 学習済みモデルロード ===")
    
    # 最新のチェックポイント検索
    checkpoint_dir = project_root / "runs/train/gpu_finetuning/checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"チェックポイントディレクトリが見つかりません: {checkpoint_dir}")
    
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError("チェックポイントファイルが見つかりません")
    
    # 最新のチェックポイント選択
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 使用チェックポイント: {latest_checkpoint}")
    
    try:
        # Lightning形式チェックポイントロード
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        print(f"✅ チェックポイントロード成功")
        print(f"   エポック: {checkpoint.get('epoch', 'N/A')}")
        print(f"   ステップ: {checkpoint.get('global_step', 'N/A')}")
        
        # state_dict取得
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"   パラメータ数: {len(state_dict)}")
        else:
            raise ValueError("state_dictが見つかりません")
            
        return state_dict, latest_checkpoint
        
    except Exception as e:
        print(f"❌ モデルロードエラー: {e}")
        raise

def create_model_for_export():
    """ONNX変換用モデル作成"""
    print("\n=== ONNX変換用モデル作成 ===")
    
    try:
        from yolo.model.yolo import YOLO
        from yolo.config.config import ModelConfig
        from omegaconf import OmegaConf
        
        # v9-s設定ロード
        model_config_path = project_root / "yolo/config/model/v9-s.yaml"
        if not model_config_path.exists():
            raise FileNotFoundError(f"モデル設定ファイルが見つかりません: {model_config_path}")
        
        # 設定ロード
        config_dict = OmegaConf.load(model_config_path)
        model_cfg = OmegaConf.structured(ModelConfig(**config_dict))
        
        print(f"✅ 設定ロード: {model_config_path}")
        
        # モデル初期化 (mezzopiano用1クラス)
        model = YOLO(model_cfg, class_num=1)
        print(f"✅ モデル作成成功: {type(model)}")
        
        # 評価モードに設定
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"❌ モデル作成エラー: {e}")
        raise

def convert_to_onnx(model, state_dict, output_path):
    """PyTorch → ONNX 変換実行"""
    print(f"\n=== ONNX変換実行 ===")
    
    try:
        # 学習済み重み適用
        print("学習済み重み適用中...")
        model.load_state_dict(state_dict, strict=False)
        print("✅ 重み適用完了")
        
        # ダミー入力作成 (YOLOv7n標準サイズ)
        dummy_input = torch.randn(1, 3, 640, 640)
        print(f"🎯 入力サイズ: {dummy_input.shape}")
        
        # ONNX変換パラメータ
        export_params = {
            'input_names': ['input'],
            'output_names': ['output'],
            'dynamic_axes': {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            'opset_version': 11,
            'do_constant_folding': True,
            'export_params': True,
        }
        
        print(f"📤 ONNX変換開始: {output_path}")
        start_time = time.time()
        
        # ONNX変換実行
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            **export_params
        )
        
        conversion_time = time.time() - start_time
        print(f"✅ ONNX変換完了! ({conversion_time:.2f}秒)")
        
        # ファイルサイズ確認
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"📁 ONNXファイルサイズ: {file_size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX変換エラー: {e}")
        return False

def verify_onnx_model(onnx_path):
    """ONNX モデル検証"""
    print(f"\n=== ONNX モデル検証 ===")
    
    try:
        import onnx
        import onnxruntime as ort
        
        # ONNX モデル検証
        print("ONNXモデル構造検証中...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNXモデル構造OK")
        
        # ONNX Runtime セッション作成
        print("ONNX Runtime セッション作成中...")
        session = ort.InferenceSession(str(onnx_path))
        print("✅ ONNX Runtime セッション作成成功")
        
        # 入出力情報表示
        print("\n📊 モデル情報:")
        for input_tensor in session.get_inputs():
            print(f"  入力: {input_tensor.name} {input_tensor.shape} {input_tensor.type}")
        
        for output_tensor in session.get_outputs():
            print(f"  出力: {output_tensor.name} {output_tensor.shape} {output_tensor.type}")
        
        # テスト推論実行
        print("\nテスト推論実行中...")
        test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        start_time = time.time()
        outputs = session.run(None, {'input': test_input})
        inference_time = time.time() - start_time
        
        print(f"✅ テスト推論成功!")
        print(f"   推論時間: {inference_time*1000:.1f}ms")
        print(f"   出力形状: {[out.shape for out in outputs]}")
        
        return True, session
        
    except ImportError as e:
        print(f"⚠️ ONNX/ONNX Runtime未インストール: {e}")
        print("pip install onnx onnxruntime でインストールしてください")
        return False, None
    except Exception as e:
        print(f"❌ ONNX検証エラー: {e}")
        return False, None

def main():
    """メイン実行"""
    print("🚀 YOLOv7n ONNX変換開始")
    
    # 出力ディレクトリ作成
    output_dir = project_root / "exports" / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "yolov7n_mezzopiano.onnx"
    
    try:
        # 1. 学習済みモデルロード
        state_dict, checkpoint_path = load_trained_model()
        
        # 2. 変換用モデル作成
        model = create_model_for_export()
        
        # 3. ONNX変換実行
        success = convert_to_onnx(model, state_dict, output_path)
        if not success:
            return False
        
        # 4. ONNX モデル検証
        verify_success, session = verify_onnx_model(output_path)
        
        if verify_success:
            print(f"\n🎉 ONNX変換完了!")
            print(f"📁 出力ファイル: {output_path}")
            print(f"📊 元チェックポイント: {checkpoint_path}")
            print(f"🔧 次のステップ: ONNX Runtime最適化")
        else:
            print(f"\n⚠️ ONNX変換は完了しましたが、検証に問題があります")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ONNX変換失敗: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)