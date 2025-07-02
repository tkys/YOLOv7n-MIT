"""
YOLOv7n MIT Edition - 統一ログシステム
構造化ログ（JSON）でinfo/debug/warning/errorレベルを管理
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

class StructuredLogger:
    """構造化ログシステム"""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # ログファイルパス
        self.log_file = self.log_dir / f"{name}.log"
        self.error_file = self.log_dir / f"{name}_error.log"
        
        # ロガー設定
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # フォーマッターの設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # ファイルハンドラー（全レベル）
        file_handler = logging.FileHandler(
            self.log_file, encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # エラーファイルハンドラー（ERROR以上）
        error_handler = logging.FileHandler(
            self.error_file, encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # ハンドラー追加
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
    
    def _create_log_entry(
        self, 
        level: str, 
        message: str, 
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """構造化ログエントリ作成"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "logger": self.name,
            "level": level,
            "message": message,
        }
        
        if extra:
            entry.update(extra)
            
        return entry
    
    def debug(self, message: str, **kwargs):
        """デバッグログ"""
        entry = self._create_log_entry("DEBUG", message, kwargs)
        self.logger.debug(json.dumps(entry, ensure_ascii=False))
    
    def info(self, message: str, **kwargs):
        """情報ログ"""
        entry = self._create_log_entry("INFO", message, kwargs)
        self.logger.info(json.dumps(entry, ensure_ascii=False))
    
    def warning(self, message: str, **kwargs):
        """警告ログ"""
        entry = self._create_log_entry("WARNING", message, kwargs)
        self.logger.warning(json.dumps(entry, ensure_ascii=False))
    
    def error(self, message: str, **kwargs):
        """エラーログ"""
        entry = self._create_log_entry("ERROR", message, kwargs)
        self.logger.error(json.dumps(entry, ensure_ascii=False))
    
    def critical(self, message: str, **kwargs):
        """クリティカルログ"""
        entry = self._create_log_entry("CRITICAL", message, kwargs)
        self.logger.critical(json.dumps(entry, ensure_ascii=False))

# グローバルロガーインスタンス
def get_logger(name: str) -> StructuredLogger:
    """ロガーインスタンス取得"""
    return StructuredLogger(name)

# 各モジュール用のロガー
inference_logger = get_logger("inference")
training_logger = get_logger("training")
api_logger = get_logger("api")
benchmark_logger = get_logger("benchmark")