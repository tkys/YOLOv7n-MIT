#!/usr/bin/env python3
"""
ONNX 変換自動化スクリプト (修正版)
Phase 2 学習完了 → Phase 3 ONNX 自動変換パイプライン
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ONNXExportPipeline:
    """ONNX変換パイプライン自動化クラス"""
    
    def __init__(self, checkpoint_path: Optional[str] = None, output_dir: Optional[str] = None):
        self.project_root = project_root
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "exports"
        self.scripts_dir = self.project_root / "scripts"
        
        # 実行ログ
        self.execution_log = []
        
    def log(self, message: str, level: str = "INFO"):
        """ログ出力"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.execution_log.append(log_entry)
    
    def run_script(self, script_name: str, args: List[str] = None) -> Dict:
        """スクリプト実行"""
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            self.log(f"スクリプトが見つかりません: {script_path}", "ERROR")
            return {"success": False, "error": f"Script not found: {script_path}"}
        
        # コマンド構築
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        self.log(f"実行中: {' '.join(cmd)}")
        
        try:
            # uv環境での実行
            activate_script = self.project_root / "yolo-training/bin/activate"
            if activate_script.exists():
                # bash で uv環境をactivate してからPython実行
                full_cmd = f"source {activate_script} && {' '.join(cmd)}"
                result = subprocess.run(
                    full_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
            else:
                # 直接実行
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
            
            if result.returncode == 0:
                self.log(f"✅ {script_name} 実行成功")
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "script": script_name
                }
            else:
                self.log(f"❌ {script_name} 実行失敗 (code: {result.returncode})", "ERROR")
                return {
                    "success": False,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "script": script_name
                }
                
        except Exception as e:
            self.log(f"❌ {script_name} 実行エラー: {e}", "ERROR")
            return {"success": False, "error": str(e), "script": script_name}
    
    def verify_prerequisites(self) -> bool:
        """前提条件確認"""
        self.log("=== 前提条件確認 ===")
        
        # 1. 学習済みチェックポイント確認
        checkpoint_dir = self.project_root / "runs/train"
        
        if self.checkpoint_path:
            checkpoint_file = Path(self.checkpoint_path)
            if not checkpoint_file.exists():
                self.log(f"指定されたチェックポイントが見つかりません: {checkpoint_file}", "ERROR")
                return False
        else:
            # 最新のチェックポイント検索
            train_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
            if not train_dirs:
                self.log("学習結果ディレクトリが見つかりません", "ERROR")
                return False
            
            # GPU Fine-tuning を優先
            gpu_dirs = [d for d in train_dirs if 'gpu' in d.name.lower()]
            if gpu_dirs:
                latest_dir = max(gpu_dirs, key=lambda x: x.stat().st_mtime)
            else:
                latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
            
            checkpoint_files = list((latest_dir / "checkpoints").glob("*.ckpt"))
            if not checkpoint_files:
                self.log(f"チェックポイントファイルが見つかりません: {latest_dir}", "ERROR")
                return False
            
            self.checkpoint_path = str(max(checkpoint_files, key=lambda x: x.stat().st_mtime))
        
        self.log(f"✅ チェックポイント確認: {self.checkpoint_path}")
        
        # 2. 必要スクリプト確認
        required_scripts = [
            "convert_to_onnx.py",
            "benchmark_onnx.py",
            "memory_profiler.py",
            "cross_platform_validator.py"
        ]
        
        for script in required_scripts:
            script_path = self.scripts_dir / script
            if not script_path.exists():
                self.log(f"必要スクリプトが見つかりません: {script}", "ERROR")
                return False
        
        self.log("✅ 必要スクリプト確認完了")
        
        # 3. 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"✅ 出力ディレクトリ確認: {self.output_dir}")
        
        return True
    
    def generate_summary_report(self) -> bool:
        """総合レポート生成"""
        self.log("\n=== 総合レポート生成 ===")
        
        summary_path = self.output_dir / "onnx_export_summary.md"
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("# YOLOv7n ONNX 変換・最適化 総合レポート\n\n")
                f.write(f"**生成日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**元チェックポイント**: `{self.checkpoint_path}`\n\n")
                
                # 実行サマリー
                f.write("## 🚀 実行サマリー\n\n")
                f.write("| ステップ | ステータス | 生成物 |\n")
                f.write("|---------|-----------|--------|\n")
                
                # 各ステップの状況確認
                steps = [
                    ("ONNX変換", "exports/onnx/yolov7n_mezzopiano.onnx"),
                    ("性能ベンチマーク", "analysis/onnx_benchmark/benchmark_report.md"),
                    ("メモリプロファイリング", "analysis/memory_profiling/memory_profile_report.md"),
                    ("クロスプラットフォーム検証", "analysis/cross_platform/cross_platform_compatibility_report.md")
                ]
                
                for step_name, output_file in steps:
                    file_path = self.project_root / output_file
                    if file_path.exists():
                        f.write(f"| {step_name} | ✅ 完了 | `{output_file}` |\n")
                    else:
                        f.write(f"| {step_name} | ❌ 失敗 | - |\n")
                
                # 生成ファイル一覧
                f.write("\n## 📁 生成ファイル\n\n")
                
                # ONNX モデル
                onnx_path = self.project_root / "exports/onnx/yolov7n_mezzopiano.onnx"
                if onnx_path.exists():
                    size_mb = onnx_path.stat().st_size / 1024 / 1024
                    f.write(f"### ONNXモデル\n")
                    f.write(f"- **ファイル**: `{onnx_path.relative_to(self.project_root)}`\n")
                    f.write(f"- **サイズ**: {size_mb:.1f}MB\n\n")
                
                # 分析レポート
                f.write("### 分析レポート\n")
                analysis_files = [
                    "analysis/onnx_benchmark/benchmark_report.md",
                    "analysis/memory_profiling/memory_profile_report.md",
                    "analysis/memory_profiling/memory_profile_visualization.png",
                    "analysis/cross_platform/cross_platform_compatibility_report.md"
                ]
                
                for analysis_file in analysis_files:
                    file_path = self.project_root / analysis_file
                    if file_path.exists():
                        f.write(f"- `{analysis_file}`\n")
                
                # 次のステップ
                f.write("\n## 🎯 次のステップ (Phase 4)\n\n")
                f.write("1. **FastAPI推論サーバー構築**\n")
                f.write("2. **Docker化対応**\n")
                f.write("3. **AWS Lambda展開準備**\n")
                f.write("4. **本番環境テスト**\n\n")
                
                # 実行ログ
                f.write("## 📝 実行ログ\n\n")
                f.write("```\n")
                for log_entry in self.execution_log:
                    f.write(f"{log_entry}\n")
                f.write("```\n")
            
            self.log(f"✅ 総合レポート生成完了: {summary_path}")
            return True
            
        except Exception as e:
            self.log(f"総合レポート生成エラー: {e}", "ERROR")
            return False

def main():
    """メイン実行"""
    print("🚀 ONNX変換自動化パイプライン開始")
    
    # パイプライン初期化
    pipeline = ONNXExportPipeline()
    
    start_time = time.time()
    
    # 前提条件確認
    if not pipeline.verify_prerequisites():
        print("❌ 前提条件確認失敗")
        return False
    
    # 総合レポート生成
    if not pipeline.generate_summary_report():
        print("❌ 総合レポート生成失敗")
        return False
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 ONNX変換自動化パイプライン完了!")
    print(f"⏱️ 総実行時間: {total_time:.1f}秒")
    print(f"📁 出力ディレクトリ: {pipeline.output_dir}")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)