#!/usr/bin/env python3
"""
Fine-tuning問題解決スクリプト
事前訓練済みモデルの互換性を修正してTransfer Learningを実現
"""
import torch
import os
from pathlib import Path

def analyze_pretrained_model():
    """事前訓練済みモデルの構造解析"""
    print("=== 事前訓練済みモデル解析 ===")
    
    model_path = "weights/v9-s.pt"
    if not Path(model_path).exists():
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return None
    
    try:
        # PyTorchモデルロード
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print(f"✅ モデルロード成功: {model_path}")
        print(f"チェックポイントキー数: {len(checkpoint)}")
        
        # このモデルは直接state_dict形式
        state_dict = checkpoint
        
        # クラス数関連のレイヤー確認
        class_conv_keys = [k for k in state_dict.keys() if 'class_conv' in k]
        print(f"クラス分類レイヤー: {len(class_conv_keys)}個")
        if class_conv_keys:
            for key in class_conv_keys[:3]:  # 最初の3個表示
                print(f"  {key}: {state_dict[key].shape}")
        else:
            # heads検索
            head_keys = [k for k in state_dict.keys() if 'heads' in k and ('weight' in k or 'bias' in k)]
            print(f"ヘッドレイヤー: {len(head_keys)}個")
            for key in head_keys[:5]:
                print(f"  {key}: {state_dict[key].shape}")
                
        return checkpoint
        
    except Exception as e:
        print(f"❌ モデル解析エラー: {e}")
        return None

def create_compatible_model():
    """互換性のある事前訓練済みモデル作成"""
    print("\n=== 互換性モデル作成 ===")
    
    # 元モデルロード
    original_path = "weights/v9-s.pt"
    state_dict = torch.load(original_path, map_location='cpu', weights_only=False)
    
    # クラス数を1に調整（mezzopiano用）
    modified_state_dict = {}
    
    for key, value in state_dict.items():
        if 'class_conv' in key and 'weight' in key:
            # クラス分類レイヤーの出力を1クラスに調整
            if value.shape[0] == 80:  # COCO 80クラス → 1クラス
                print(f"修正: {key} {value.shape} → ({1}, {value.shape[1:]}"[1:])
                modified_state_dict[key] = value[:1]  # 最初の1クラス分のみ
            else:
                modified_state_dict[key] = value
        elif 'class_conv' in key and 'bias' in key:
            # バイアスも1クラス分に調整
            if value.shape[0] == 80:
                print(f"修正: {key} {value.shape} → (1,)")
                modified_state_dict[key] = value[:1]
            else:
                modified_state_dict[key] = value
        else:
            modified_state_dict[key] = value
    
    # 新しいチェックポイント作成（元と同じ形式）
    new_checkpoint = modified_state_dict
    
    # 保存
    output_path = "weights/v9-s-mezzopiano.pt"
    torch.save(new_checkpoint, output_path)
    print(f"✅ 互換性モデル保存: {output_path}")
    
    return output_path

def test_finetuning():
    """Fine-tuning テスト実行"""
    print("\n=== Fine-tuning テスト ===")
    
    import subprocess
    
    cmd = [
        "python", "yolo/lazy.py",
        "task=train",
        "dataset=mezzopiano",
        "model=v9-s",
        "device=cpu",
        "task.epoch=2",  # テスト用短期間
        "task.data.batch_size=4",
        "task.optimizer.args.lr=0.001",  # Fine-tuning用低学習率
        "use_wandb=false",
        "name=finetuning_test",
        "weight=weights/v9-s-mezzopiano.pt"  # 修正済みモデル
    ]
    
    print(f"実行コマンド: {' '.join(cmd)}")
    
    try:
        env = os.environ.copy()
        env['HYDRA_FULL_ERROR'] = '1'
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        
        if result.returncode == 0:
            print("✅ Fine-tuning テスト成功!")
            return True
        else:
            print("❌ Fine-tuning テスト失敗")
            print("エラー:", result.stderr[-1000:])
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ テストタイムアウト")
        return False
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        return False

def main():
    print("🔧 YOLOv7n Fine-tuning問題解決開始")
    
    # Step 1: モデル解析
    checkpoint = analyze_pretrained_model()
    if checkpoint is None:
        return False
    
    # Step 2: 互換性モデル作成
    try:
        compatible_model_path = create_compatible_model()
    except Exception as e:
        print(f"❌ 互換性モデル作成失敗: {e}")
        return False
    
    # Step 3: Fine-tuning テスト
    success = test_finetuning()
    
    if success:
        print("\n🎉 Fine-tuning問題解決完了!")
        print(f"利用可能モデル: {compatible_model_path}")
        print("次回からは weight=weights/v9-s-mezzopiano.pt でFine-tuning可能")
    else:
        print("\n⚠️ さらなる調整が必要")
    
    return success

if __name__ == "__main__":
    main()